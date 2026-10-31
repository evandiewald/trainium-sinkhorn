"""Same unbalanced OT problems on different libraries / hardware: trnopt (Trainium), FlashSinkhorn and OTT-JAX (GPU).

    python examples/sinkhorn_libs_compare.py --lib trnopt --pots results/sc_potentials        # on the trn1
    python examples/sinkhorn_libs_compare.py --lib flash  --pots sc_potentials                # on a CUDA GPU
    python examples/sinkhorn_libs_compare.py --lib ott    --pots sc_potentials                # (OTT-JAX, moscot's engine)

Problems: real consecutive-day pairs from the Waddington-OT data (the coordinates saved by singlecell_wot.py --save:
wot's local PCA, d = 30, normalized so the median cost is 1). Formulation shared by all three libraries (GeomLoss /
FlashSinkhorn convention): P = a b^T exp((f + g - C) / eps), C = |x - y|^2, row weights a = G / I (initial growth),
column weights b = mean(G) / J, KL marginal penalties rho_x = 1, rho_y = 50 (wot's lambda1, lambda2), eps = 0.05
fixed, float32, alternating updates.

Per pair: time for a fixed number of iterations (--iters, after a warm-up solve) -> ms per iteration, and the plan's
row and column masses (saved to --save as npz) so the libraries' answers can be compared with compare_masses().
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import socket
import time
from pathlib import Path

import numpy as np

EPS, RHO_X, RHO_Y = 0.05, 1.0, 50.0
PAIRS = ["3.5_4", "4_4.5", "5.5_6", "3_3.5", "16_16.5", "10_10.5"]


def problem(npz):
    z = np.load(npz, allow_pickle=True)
    s = np.sqrt(float(z["cost_scale"]))
    x = ((z["x0"] - z["center"]) / s).astype(np.float32)
    y = ((z["x1"] - z["center"]) / s).astype(np.float32)
    G = np.asarray(z["learned_growth"][0], dtype=np.float64)
    a = (G / len(x)).astype(np.float32)
    b = np.full(len(y), G.mean() / len(y), dtype=np.float32)
    return x, y, a, b


def masses_from_potentials(x, y, a, b, f, g, chunk=2048):
    """Row and column sums of P = a b^T exp((f + g - C) / eps), on the host in float64."""
    from scipy.special import logsumexp
    x, y, a, b, f, g = (np.asarray(v, dtype=np.float64) for v in (x, y, a, b, f, g))
    y2 = (y * y).sum(1)
    rows = np.empty(len(x))
    cols = np.full(len(y), -np.inf)
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        S = (f[i:i + chunk][:, None] + g[None, :] - ((xb * xb).sum(1)[:, None] + y2[None, :] - 2 * xb @ y.T)) / EPS
        S += np.log(a[i:i + chunk])[:, None] + np.log(b)[None, :]
        rows[i:i + chunk] = np.exp(logsumexp(S, axis=1))
        cols = np.logaddexp(cols, logsumexp(S, axis=0))
    return rows, np.exp(cols)


def run_trnopt(x, y, a, b, iters, cores, backend="neuron"):
    from trnopt import Backend
    from trnopt.sinkhorn import SinkhornOptions, sinkhorn
    o = SinkhornOptions(eps=EPS, rho=(RHO_X, RHO_Y), normalize=False, max_iters=iters, fixed_iters=True,
                        backend=Backend.preset(backend, dtype=__import__("torch").float32), cores=cores)
    sinkhorn(x, y, a, b, o)                                   # warm-up (compile)
    r = sinkhorn(x, y, a, b, o)
    return r.seconds, r.f, r.g


def run_flash(x, y, a, b, iters, tf32=False):
    import torch
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating
    X, Y, A, B = (torch.as_tensor(v, device="cuda") for v in (x, y, a, b))
    kw = dict(eps=EPS, n_iters=iters, rho_x=RHO_X, rho_y=RHO_Y, allow_tf32=tf32, autotune=True)
    sinkhorn_flashstyle_alternating(X, Y, A, B, **kw)          # warm-up (autotune + compile)
    torch.cuda.synchronize()
    t = time.perf_counter()
    f, g = sinkhorn_flashstyle_alternating(X, Y, A, B, **kw)
    torch.cuda.synchronize()
    return time.perf_counter() - t, f.double().cpu().numpy(), g.double().cpu().numpy()


def run_ott(x, y, a, b, iters, tf32=False):
    import jax
    # On Ampere+ JAX's default float32 matmul precision is TF32; "highest" forces full fp32.
    jax.config.update("jax_default_matmul_precision", "default" if tf32 else "highest")
    import jax.numpy as jnp
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn as sk
    geom = pointcloud.PointCloud(jnp.asarray(x), jnp.asarray(y), epsilon=EPS)       # cost |x - y|^2
    prob = linear_problem.LinearProblem(geom, a=jnp.asarray(a), b=jnp.asarray(b),
                                        tau_a=RHO_X / (RHO_X + EPS), tau_b=RHO_Y / (RHO_Y + EPS))
    solver = sk.Sinkhorn(threshold=0.0, min_iterations=iters, max_iterations=iters, inner_iterations=iters,
                         lse_mode=True)
    solve = jax.jit(lambda p: solver(p))
    out = solve(prob)
    jax.block_until_ready(out.f)                               # warm-up (compile)
    t = time.perf_counter()
    out = solve(prob)
    jax.block_until_ready(out.f)
    sec = time.perf_counter() - t
    # OTT's potentials: P = exp((f + g - C) / eps) (no a, b factors) -> GeomLoss convention f - eps log a
    f = np.asarray(out.f, dtype=np.float64) - EPS * np.log(np.asarray(a, dtype=np.float64))
    g = np.asarray(out.g, dtype=np.float64) - EPS * np.log(np.asarray(b, dtype=np.float64))
    return sec, f, g


def compare_masses(dir_a: str, dir_b: str):
    """Max relative difference of row / column masses between two --save directories, per pair."""
    for p in sorted(Path(dir_a).glob("*.npz")):
        q = Path(dir_b) / p.name
        if q.exists():
            A, B = np.load(p), np.load(q)
            print(p.stem, f"rows {np.abs(A['rows'] / B['rows'] - 1).max():.2e}  cols {np.abs(A['cols'] / B['cols'] - 1).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, choices=["trnopt", "flash", "ott"])
    ap.add_argument("--pots", required=True, help="directory of pair_*.npz from singlecell_wot.py --save")
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--backend", default="neuron", help="trnopt only: Backend preset (cpu for a dry run)")
    ap.add_argument("--tf32", action="store_true", help="flash / ott: TF32 matmuls (Ampere+); default strict fp32")
    ap.add_argument("--save", default=None, help="directory for per-pair row/column masses")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/libs_compare_{a.lib}_{socket.gethostname()}_{dt.date.today().isoformat()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(out.open("w", newline=""))
    w.writerow(["lib", "precision", "host", "pair", "n", "m", "iters", "seconds", "ms_per_iter"])
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
    for pair in a.pairs:
        x, y, wa, wb = problem(Path(a.pots) / f"pair_{pair}.npz")
        if a.lib == "trnopt":
            sec, f, g = run_trnopt(x, y, wa, wb, a.iters, a.cores, a.backend)
        elif a.lib == "flash":
            sec, f, g = run_flash(x, y, wa, wb, a.iters, a.tf32)
        else:
            sec, f, g = run_ott(x, y, wa, wb, a.iters, a.tf32)
        prec = "tf32" if a.tf32 else "fp32"
        print(f"{a.lib} {prec} {pair}: {len(x)} x {len(y)}, {a.iters} its in {sec:.3f} s ({sec / a.iters * 1e3:.3f} ms/it)", flush=True)
        w.writerow([a.lib, prec, socket.gethostname(), pair, len(x), len(y), a.iters, round(sec, 4), round(sec / a.iters * 1e3, 4)])
        if a.save:
            rows, cols = masses_from_potentials(x, y, wa, wb, f, g)
            np.savez(Path(a.save) / f"pair_{pair}.npz", rows=rows, cols=cols)


if __name__ == "__main__":
    main()
