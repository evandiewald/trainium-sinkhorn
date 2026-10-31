"""Entropic optimal transport (Sinkhorn) in plain PyTorch: the pre-NKI version, and the reference for the kernel.

What it solves
--------------
Two clouds of points, x (n, d) and y (m, d), with weights a (n,) and b (m,), and the cost of moving one unit of
weight from x_i to y_j, C_ij = |x_i - y_j|^2. Find the transport plan P (n x m, how much of x_i goes to y_j)
that minimizes

    <P, C> + eps * KL(P | a b^T)

with row sums a and column sums b ("balanced"). eps blurs the plan: small eps is close to exact (unregularized) OT,
large eps converges faster. "Unbalanced" only encourages the sums, with a penalty rho * KL(row sums | a) (and
the same for columns), so weight can appear or disappear: cells dividing and dying between two time points.

How: log-domain Sinkhorn
------------------------
The best plan always has the form P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps) for two "potential" vectors f
and g. Sinkhorn alternates

    f_i = -eps * LSE_j( log b_j + (g_j - C_ij) / eps )       rows now sum to a
    g_j = -eps * LSE_i( log a_i + (f_i - C_ij) / eps )       columns now sum to b

where LSE(z) = log(sum(exp(z))), computed as max(z) + log(sum(exp(z - max(z)))) so nothing overflows.
Unbalanced: multiply each update by kappa = rho / (rho + eps). Balanced is rho = infinity, kappa = 1.

The attention trick (as in FlashSinkhorn, arXiv 2602.03067)
-----------------------------------------------------------
Expand C_ij = |x_i|^2 + |y_j|^2 - 2 x_i.y_j and carry shifted potentials fh = f - |x|^2, gh = g - |y|^2:

    f_i = |x_i|^2 - eps * LSE_j( log b_j + (gh_j + 2 x_i.y_j) / eps )

That's a matrix multiply (x @ y^T, the Tensor engine's job), a per-column bias, and a row-wise log-sum-exp:
the same shape as softmax in attention. The cost matrix is never stored.

Blocking
--------
The scores for all of x against all of y are n x m: 10 GB in float32 at 50k x 50k, 300 GB at 275k x 275k.
block=None computes every row at once; block=k streams k rows at a time and recomputes their scores each
iteration, so memory is O(k m). Each block is one compiled graph with one fixed shape (Neuron compiles one
program per shape), so x and y are padded up to a multiple of their block. Padded points get a log weight of
-1e9 (weight exp(-1e9) = 0): finite on purpose, because infinities gave silently wrong answers on Neuron in
earlier experiments.

The PyTorch half-step (_half_step) writes each score tile to device memory and reads it back for the exp. On a
Neuron device, kernel="auto" uses the fused NKI kernel instead (nki_sinkhorn.py), which keeps the tiles on chip;
the PyTorch version stays the reference it has to match.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from .backend import Backend

PAD_LOG_WEIGHT = -1e9          # log weight of a padding point: zero weight without an infinity
_DELTA_CAP = 50.0              # cap on (g_old - g_new) / eps in the marginal check, so exp() stays finite


@dataclass
class SinkhornOptions:
    eps: float = 0.05              # blur, in units of the cost (after normalize: of the mean / median squared distance)
    rho: float | tuple | None = None  # None: balanced. Else the unbalanced marginal penalty, in the same units: one
                                   #   value for both sides, or (rho_x, rho_y) (wot's lambda1, lambda2)
    max_iters: int = 2000
    tol: float = 1e-4              # balanced: L1 column-marginal error; unbalanced: max potential change / eps
                                   #   (unbalanced float32: at least 5e-6, its rounding floor is ~1e-6)
    check_every: int = 10          # each check is a device-to-host read, so not every iteration
    block: int | None = None       # rows of x per block in the f update (None: all of them at once)
    block_y: int | None = None     # rows of y per block in the g update (None: same as block)
    eps_start: float | None = None # eps-scaling: start at eps_start, multiply by eps_decay each iteration down to
    eps_decay: float = 0.9         #   eps, then iterate at eps. Fewer iterations at small eps. None: off
    eps_stages: tuple | None = None  # alternative eps-scaling (wot's): run at each of these eps in turn until the
    stage_tol: float = 1e-6        #   max potential change / eps < stage_tol (at least 1e-5 in float32, whose rounding
                                   #   noise sits near 1e-6) or stage_max_iters, then at eps until tol. Overrides eps_start
    stage_max_iters: int = 1000
    normalize: bool | str | tuple = True  # center the points and scale the cost: True / "mean" (mean squared x-y
                                   #   distance = 1), "median" (median = 1, as wot does; exact when n m <= 1e8),
                                   #   False (raw), or a precomputed (center, cost_scale) from normalization()
    fixed_iters: bool = False      # run exactly max_iters, no convergence checks (benchmarks)
    cores: int = 1                 # NeuronCores to split each half-step's rows across (nki kernel only). The halves of
                                   #   the new potential go through the host to reach every core (~0.2 ms each time)
    kernel: str = "auto"           # half-step: "torch" (compiled PyTorch, any device), "nki" (the fused Trainium
                                   #   kernel in nki_sinkhorn.py), "auto" (nki on a Neuron device when d < 128)
    backend: Backend = Backend.preset("cpu")


@dataclass
class SinkhornResult:
    f: np.ndarray                  # potentials, float64, in normalized cost units (times cost_scale: original)
    g: np.ndarray
    iterations: int
    error: float                   # last convergence measure (see SinkhornOptions.tol); nan if never checked
    converged: bool
    cost: float                    # balanced: <a, f> + <b, g>, the entropic OT value (normalized units). nan if unbalanced
    seconds: float                 # wall time of the iteration loop, including any compile on a first call
    eps: float
    rho: float | tuple | None
    center: np.ndarray             # normalization: x_used = (x - center) / sqrt(cost_scale)
    cost_scale: float
    row_mass: np.ndarray | None = None   # unbalanced: the plan's row and column sums (its actual marginals)
    col_mass: np.ndarray | None = None


def _half_step(x_blk, xsq_blk, off_blk, yT, gh, log_b, inv_eps, eps, kappa):
    """One block of rows of the f update, in shifted form: returns fh = f - |x|^2 for these rows.

    x_blk (k, d), xsq_blk (k,), off_blk (k,) = log(target / reference weight) of these rows (0 unless the row
    marginal's target differs from the reference measure), yT (d, m), gh (m,), log_b (m,); inv_eps, eps, kappa
    are 0-d tensors (not Python floats, which torch.compile would bake in as constants and recompile for every
    new eps). The g update is the same function with x and y swapped.
    """
    z = (2.0 * (x_blk @ yT) + gh[None, :]) * inv_eps + log_b[None, :]       # (k, m) scores
    zmax = z.amax(dim=1, keepdim=True)
    lse = zmax[:, 0] + torch.log(torch.exp(z - zmax).sum(dim=1))
    return kappa * (xsq_blk + eps * off_blk - eps * lse) - xsq_blk


NKI_MAX_CHUNKS = 8192          # rows per kernel call: at most this many (128 x 512) score chunks, so the unrolled
                               # kernel stays a few thousand instructions per call (compiles in seconds)


def nki_rows_per_call(m_pad: int) -> int:
    """Rows of the fused kernel per call when the caller gives no block: a multiple of 512, as large as allowed."""
    return max(512, (NKI_MAX_CHUNKS // max(1, m_pad // 512)) * 128 // 512 * 512)


class _Cloud:
    """One point cloud on the device, padded to a whole number of blocks (and of blocks per core).

    With several cores every core holds the whole cloud (it's the column side of the other half-step), and
    computes the rows of its own share of the blocks (my_blocks) in its own half-steps.
    """

    def __init__(self, pts: np.ndarray, log_w: np.ndarray, block: int, be: Backend, nki: bool = False,
                 core: int = 0, cores: int = 1, off: np.ndarray | None = None):
        n, d = pts.shape
        nb = -(-n // (block * cores)) * cores
        P = np.zeros((nb * block, d))
        P[:n] = pts
        L = np.full(nb * block, PAD_LOG_WEIGHT)
        L[:n] = log_w
        self.n, self.block, self.npad = n, block, nb * block
        x = be.tensor(P)
        self.xT = x.T.contiguous()
        sq = be.tensor((P * P).sum(axis=1))            # squared norms in float64, then cast
        O = np.zeros(nb * block)
        if off is not None:
            O[:n] = off
        ot, oz = be.tensor(O), be.tensor(np.zeros(nb * block))
        self.blocks = [(x[s:s + block].contiguous(), sq[s:s + block].contiguous(), ot[s:s + block].contiguous(),
                        oz[s:s + block].contiguous()) for s in range(0, self.npad, block)]
        self.sq = sq
        self.log_w = be.tensor(L)
        self.w = be.tensor(np.exp(L))                    # padding: exactly 0
        if nki:                                          # [x ; 1] per block, (d + 1, block), for the fused kernel
            XA = np.vstack([P.T, np.ones((1, len(P)))])
            self.xa_blocks = [be.tensor(np.ascontiguousarray(XA[:, s:s + block])) for s in range(0, self.npad, block)]
            sqn = (P * P).sum(axis=1)
            self.sq_blocks = [be.tensor(np.ascontiguousarray(sqn[s:s + block])) for s in range(0, self.npad, block)]
            self.off_blocks = [be.tensor(np.ascontiguousarray(O[s:s + block])) for s in range(0, self.npad, block)]
            self.zero_blocks = [be.tensor(np.zeros(block)) for _ in range(0, self.npad, block)]
        per = nb // cores
        self.my_blocks = range(core * per, (core + 1) * per)          # this core's share of the row blocks


def _sweep(step, side: _Cloud, other: _Cloud, other_h, inv_eps, eps, kappa, sc=None, use_off=True):
    if isinstance(step, _NkiStep):
        return step(side, other, other_h, sc, use_off)
    return torch.cat([step(xb, sqb, ob if use_off else zb, other.xT, other_h, other.log_w, inv_eps, eps, kappa)
                      for xb, sqb, ob, zb in side.blocks])


class _NkiStep:
    """A half-step with the fused kernel (nki_sinkhorn.make_half_step_kernel): one kernel call per block of this
    core's rows, nothing else on the device. Returns the new fh for those rows: on the device with one core, on
    the host (to be exchanged) with several."""

    def __init__(self, be: Backend, cores: int):
        self.be, self.cores = be, cores

    def __call__(self, side: _Cloud, other: _Cloud, other_h, sc, use_off=True):
        from .nki_sinkhorn import make_half_step_kernel
        k = make_half_step_kernel(other.xT.shape[0], side.block, other.npad)
        offs = side.off_blocks if use_off else side.zero_blocks
        parts = [k(side.xa_blocks[i], side.sq_blocks[i], offs[i], other.xT, other_h, other.log_w, sc)
                 for i in side.my_blocks]
        if self.cores == 1:
            return parts[0] if len(parts) == 1 else torch.cat(parts)
        return parts


def _use_nki(opts: SinkhornOptions, d: int) -> bool:
    if opts.kernel == "torch":
        return False
    be = opts.backend
    ok = be.device.startswith("neuron") and d + 1 <= 128 and be.dtype == torch.float32
    if opts.kernel == "nki" and not ok:
        raise ValueError("kernel='nki' needs a Neuron device, float32 and d < 128")
    if ok:
        from .nki_sinkhorn import HAVE_NKI
        if not HAVE_NKI and opts.kernel == "nki":
            raise ImportError("kernel='nki' needs the nki package")
        return HAVE_NKI
    return False


def _log_weights(w, n: int, name: str) -> np.ndarray:
    if w is None:
        return np.full(n, -np.log(n))
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (n,) or np.any(w <= 0):
        raise ValueError(f"{name} must be {n} positive weights")
    return np.log(w)


def normalization(x: np.ndarray, y: np.ndarray, samples: int = 20_000, seed: int = 0,
                  how: str = "mean") -> tuple[np.ndarray, float]:
    """(center, cost_scale): the mean of all points, and the typical squared distance between x and y points.
    Dividing coordinates by sqrt(cost_scale) makes typical costs about 1, so eps means the same thing from dataset
    to dataset. Centering also cuts float32 rounding in |x|^2 + |y|^2 - 2 x.y.

    how="mean": mean over random pairs. how="median": the median over all pairs when n m <= 1e8 (what wot does:
    cost / median(cost)), else over random pairs."""
    center = np.concatenate([x, y]).mean(axis=0)
    if how == "median" and len(x) * len(y) <= 100_000_000:
        xc, yc = x - center, y - center
        C = ((xc * xc).sum(1)[:, None] + (yc * yc).sum(1)[None, :] - 2.0 * (xc @ yc.T)).ravel()
        scale = float(np.median(np.maximum(C, 0.0)))
        return center, (scale if scale > 0 else 1.0)
    rng = np.random.default_rng(seed)
    i, j = rng.integers(len(x), size=samples), rng.integers(len(y), size=samples)
    dd = ((x[i] - y[j]) ** 2).sum(axis=1)
    scale = float(np.median(dd) if how == "median" else dd.mean())
    return center, (scale if scale > 0 else 1.0)


def sinkhorn(x, y, a=None, b=None, opts: SinkhornOptions | None = None, a_target=None,
             b_target=None) -> SinkhornResult:
    """Entropic OT between point clouds x (n, d) and y (m, d) with weights a, b (default uniform).

    a, b are the reference measure of the entropy term, eps KL(P | a b^T). Unbalanced (rho given), the marginal
    penalties rho KL(P 1 | a_target) and rho KL(P^T 1 | b_target) pull the plan's row and column sums toward
    a_target and b_target, which default to a and b. (wot's formulation: uniform reference, a_target = the
    cells' growth, b_target = uniform times the mean growth.)
    """
    opts = opts or SinkhornOptions()
    be = opts.backend
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    log_a, log_b = _log_weights(a, len(x), "a"), _log_weights(b, len(y), "b")
    balanced = opts.rho is None
    if balanced and abs(np.exp(log_a).sum() - np.exp(log_b).sum()) > 1e-9 * np.exp(log_a).sum():
        raise ValueError("balanced OT needs sum(a) == sum(b); pass rho for unbalanced")
    if balanced and (a_target is not None or b_target is not None):
        raise ValueError("a_target / b_target are for unbalanced OT (pass rho)")
    rho_x, rho_y = (None, None) if balanced else (tuple(opts.rho) if isinstance(opts.rho, (tuple, list))
                                                    else (opts.rho, opts.rho))
    off_x = None if a_target is None else _log_weights(a_target, len(x), "a_target") - log_a
    off_y = None if b_target is None else _log_weights(b_target, len(y), "b_target") - log_b
    if isinstance(opts.normalize, tuple):                  # precomputed (center, cost_scale)
        center, scale = np.asarray(opts.normalize[0], dtype=np.float64), float(opts.normalize[1])
    elif opts.normalize:
        how = opts.normalize if isinstance(opts.normalize, str) else "mean"
        center, scale = normalization(x, y, how=how)
    else:
        center, scale = np.zeros(x.shape[1]), 1.0
    xs, ys = (x - center) / np.sqrt(scale), (y - center) / np.sqrt(scale)

    nki = _use_nki(opts, x.shape[1])
    cores = max(1, int(opts.cores))
    if cores > 1 and not nki:
        raise ValueError("cores > 1 needs the nki kernel (a Neuron device, float32, d < 128)")
    if nki:
        # Blocks (rows per kernel call) are multiples of 512: each cloud's padded size is then a multiple of the
        # kernel's chunk width when it plays the column role. Default: each core's share of the rows in as few
        # calls as the kernel's size cap allows.
        up = lambda v: -(-v // 512) * 512

        def rows_per_call(n_rows, n_cols, given):
            if given:
                return min(up(given), up(-(-n_rows // cores)))
            cap = nki_rows_per_call(up(n_cols))
            calls = -(-n_rows // (cores * cap))            # calls per core: as few as the size cap allows,
            return up(-(-n_rows // (cores * calls)))       # then rows split evenly (little padding)

        bx = rows_per_call(len(x), len(y), opts.block)
        by = rows_per_call(len(y), len(x), opts.block_y or opts.block)
    else:
        bx = opts.block or len(x)
        by = opts.block_y or opts.block or len(y)
    if cores > 1:
        bes = [replace(be, device=f"neuron:{c}") for c in range(cores)]
    else:
        bes = [be]
    Xs = [_Cloud(xs, log_a, bx, b_, nki, c, cores, off_x) for c, b_ in enumerate(bes)]
    Ys = [_Cloud(ys, log_b, by, b_, nki, c, cores, off_y) for c, b_ in enumerate(bes)]
    X, Y = Xs[0], Ys[0]
    steps = [_NkiStep(b_, cores) if nki else b_.compile_fn(_half_step) for b_ in bes]

    def sweep(sides, others, hs, cs, use_off=True):
        """One half-step on every core; returns the full new shifted potential, one copy per core."""
        if cores == 1:
            return [_sweep(steps[0], sides[0], others[0], hs[0], *cs[0], use_off=use_off)]
        parts = [steps[c](sides[c], others[c], hs[c], cs[c][3], use_off) for c in range(cores)]   # queued on all cores
        # Core-to-core .to() silently leaves the data where it is, so the halves go through the host.
        full = torch.cat([p.cpu() for ps in parts for p in ps])
        return [full.to(b_.device) for b_ in bes]

    # eps schedule: geometric from eps_start down to eps (eps-scaling), then eps. Or wot-style stages.
    sched = []
    stages = [e for e in (opts.eps_stages or ()) if e > opts.eps]
    if not stages and opts.eps_start is not None and opts.eps_start > opts.eps:
        e = opts.eps_start
        while e > opts.eps and len(sched) < opts.max_iters:
            sched.append(e)
            e *= opts.eps_decay
    params = {}

    def consts(e, rho):
        """Per core: (1/eps, eps, kappa) as 0-d tensors on that core, then [2/eps, 1/eps, eps, kappa] as a (1, 4)
        tensor for the fused kernel. rho None: kappa = 1."""
        key = (e, rho)
        if key not in params:
            kappa = 1.0 if rho is None else rho / (rho + e)
            if len(params) > 8:
                params.clear()
            params[key] = [tuple(torch.tensor(v, dtype=be.dtype, device=b_.device) for v in (1.0 / e, e, kappa))
                           + (torch.tensor([[2.0 / e, 1.0 / e, e, kappa]], dtype=torch.float32, device=b_.device),)
                           for b_ in bes]
        return params[key]

    ghs = [torch.zeros(Y.npad, dtype=be.dtype, device=b_.device) for b_ in bes]
    fhs = ghs
    err, converged, it = float("nan"), False, 0
    stage, stage_it = 0, 0                     # wot-style stages: index into stages (len(stages) = the final eps)
    # Intermediate stages are warm starts. In float32 the potentials' change per check stalls around 1e-6 of eps
    # (rounding), so a 1e-6 threshold could hold a stage forever (seen on real single-cell data).
    stage_tol = max(opts.stage_tol, 1e-5 if be.dtype == torch.float32 else 0.0)
    # Same for the final test when unbalanced (max potential change / eps): float32 bottoms out near 1e-6.
    tol = max(opts.tol, 5e-6) if (be.dtype == torch.float32 and not balanced) else opts.tol
    t0 = time.perf_counter()
    while it < opts.max_iters:
        if stages:
            e = stages[stage] if stage < len(stages) else opts.eps
        else:
            e = sched[it] if it < len(sched) else opts.eps
        cx, cy = consts(e, rho_x), consts(e, rho_y)
        inv_eps = cx[0][0]
        fhs = sweep(Xs, Ys, ghs, cx)
        ghs_new = sweep(Ys, Xs, fhs, cy)
        gh, gh_new = ghs[0], ghs_new[0]
        it += 1
        stage_it += 1
        if stages and stage < len(stages):     # an intermediate stage: move on once the potentials settle
            if stage_it % opts.check_every == 0:
                ch = float((((gh_new - gh)[:Y.n].abs().amax()) * inv_eps).item())
                if ch < stage_tol or stage_it >= opts.stage_max_iters:
                    stage, stage_it = stage + 1, 0
            ghs = ghs_new
            continue
        check = (not opts.fixed_iters and (stages or it > len(sched))
                 and (stage_it % opts.check_every == 0 or it == opts.max_iters))
        if check:
            if balanced:
                # Column sums of the plan with (f, old g) are b_j exp((g_old_j - g_new_j) / eps)
                delta = ((gh - gh_new) * inv_eps).clamp(max=_DELTA_CAP)
                err_t = (Y.w * (torch.exp(delta) - 1.0)).abs().sum()
            else:
                err_t = ((gh_new - gh)[:Y.n].abs().amax()) * inv_eps
            err = float(err_t.item())
        ghs = ghs_new
        if check and err < tol:
            converged = True
            break
    f = (fhs[0] + X.sq)[:X.n].detach().to("cpu", torch.float64).numpy()
    g = (ghs[0] + Y.sq)[:Y.n].detach().to("cpu", torch.float64).numpy()
    row_mass = col_mass = None
    if not balanced:
        # The plan's marginals: row sum_i = a_i exp((f_i - fb_i) / eps), where fb is one balanced (kappa = 1,
        # no offset) f update from g. One extra half-step each way.
        c1 = consts(opts.eps, None)
        fb = (sweep(Xs, Ys, ghs, c1, use_off=False)[0] + X.sq)[:X.n].detach().to("cpu", torch.float64).numpy()
        gb = (sweep(Ys, Xs, fhs, c1, use_off=False)[0] + Y.sq)[:Y.n].detach().to("cpu", torch.float64).numpy()
        row_mass = np.exp(log_a + (f - fb) / opts.eps)
        col_mass = np.exp(log_b + (g - gb) / opts.eps)
    seconds = time.perf_counter() - t0
    cost = float(np.exp(log_a) @ f + np.exp(log_b) @ g) if balanced else float("nan")
    return SinkhornResult(f=f, g=g, iterations=it, error=err, converged=converged, cost=cost, seconds=seconds,
                          eps=opts.eps, rho=opts.rho, center=center, cost_scale=scale, row_mass=row_mass,
                          col_mass=col_mass)


def transport_plan(x, y, res: SinkhornResult, a=None, b=None) -> np.ndarray:
    """The dense plan P (n x m, float64) from a result: for small problems and tests only."""
    x = (np.asarray(x, dtype=np.float64) - res.center) / np.sqrt(res.cost_scale)
    y = (np.asarray(y, dtype=np.float64) - res.center) / np.sqrt(res.cost_scale)
    log_a, log_b = _log_weights(a, len(x), "a"), _log_weights(b, len(y), "b")
    C = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    return np.exp(log_a[:, None] + log_b[None, :] + (res.f[:, None] + res.g[None, :] - C) / res.eps)
