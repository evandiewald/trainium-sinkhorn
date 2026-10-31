"""NKI kernels for a Sinkhorn half-step, fused on chip.

What they compute
-----------------
make_row_lse_kernel: the core, a row-wise log-sum-exp of a matrix product,

    lse_i = log sum_j exp( xa_i . ya_j )          i < rows, j < m

for xa_T (K, rows) and ya_T (K, m) with K <= 128 on the partition axis. With

    xa_i = [x_i, 1]          ya_j = [2 y_j / eps, gh_j / eps + log b_j]

xa_i . ya_j is exactly the Sinkhorn score, so the per-column bias rides along in the matmul (one extra
contraction row).

make_half_step_kernel: the whole half-step around that core, so nothing else runs on the device per half-step:
it builds ya_T on chip from y^T, gh, log b and the scalars, and returns the new shifted potential

    fh_i = kappa * (|x_i|^2 + eps * off_i - eps * lse_i) - |x_i|^2

(off_i = log(target / reference weight) of row i: 0 unless the row marginal's target differs from the reference
measure, as in wot's formulation)

directly. (Besides saving the launches of two small PyTorch graphs per half-step, this is what makes two
NeuronCores in one process work: TorchNeuron's compiled graphs and even its eager ops failed on neuron:1 with
"Inconsistent device IDs" once the same op had run on neuron:0, while NKI kernel calls run on either core.)

Why it's shaped this way (from the profile of the plain PyTorch version)
-----------------------------------------------------------------------
* The score tile S (128 rows x CW columns) goes from the Tensor engine to PSUM and is consumed there:
  the Vector engine takes its row max, the Scalar engine computes exp(S - max) and, in the same
  instruction, the row sum. S never touches SBUF or HBM (the compiled PyTorch spilled 0.4-1.4 GB per call).
* Rows sit on partitions from the start (stationary = 128 rows of xa_T), so the row reductions run along
  the free axis. The compiled version spent 4x the useful matmul FLOPs on layout transposes.
* Two instructions per chunk besides the matmul. Each chunk's (-max, sum) lands in its own column of a
  small per-row buffer; the chunks are merged once at the end of the row tile:
      M = max_k max_k,   l = sum_k sum_k * exp(max_k - M),   lse = M + log l
  so there's no running-max rescale (several tiny instructions) per chunk.
* ya_T is streamed once per call in column blocks of CB columns. All the call's row tiles use each block
  before the next one loads, so it's read (or built) once per call. Rows per call = the caller's block size.

Tuning (trn1, 10240 x 10240, d 64, one call): CW = 512 (one PSUM bank, so up to 8 chunks in flight) 1.77 ms,
1024 1.86 ms, 2048 2.19 ms. At CW = 512 the Tensor engine is busy nearly throughout: fp32 matmul is the floor.
bf16 operands ran no faster and lost two digits; tf32 operands fail to compile on NeuronCore-v2.

Loops unroll at trace time (NKI), so the instruction count is about (rows / 128) * (m / CW) * 3: keep rows per
call modest (the caller's block) when m is large.

Requires the standalone `nki` package (Neuron SDK 2.29+).
"""

from __future__ import annotations

import hashlib

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    HAVE_NKI = True
except ImportError:
    HAVE_NKI = False

P = 128                 # partitions
MM = 512                # one matmul's free size (one PSUM bank of fp32)
CW = 512                # columns per reduction chunk
CB = 8192               # columns of ya_T per block
NB = 4                  # rotating per-chunk stat buffers (see _row_lse)
KERNEL_VERSION = 8
_KERNELS: dict = {}


def pad_cols(m: int) -> int:
    """m rounded up to what the kernels need: a multiple of CW."""
    return -(-m // CW) * CW


def _row_lse(xs, K: int, rows: int, m: int, cw: int, cb: int, load_block, nb: int = NB):
    """Trace-time body shared by both kernels. xs: SBUF (P, rows), partitions 0..K-1 = xa_T. load_block(b0, bw)
    returns an SBUF (P, bw) tile whose partitions 0..K-1 are ya_T[:, b0:b0 + bw]. Returns an SBUF (P, R) tile
    with lse of row r * 128 + p at [p, r].

    The per-chunk (-max, sum) columns go to nb buffers in rotation (chunk kr of row tile r -> buffer kr % nb,
    column r * W + kr // nb). Dependencies are tracked per tensor, so with one buffer every chunk's stats
    instructions depend on the previous chunk's.
    """
    R, nch = rows // P, m // cw
    nb = max(1, min(nb, nch))
    W = -(-nch // nb)                          # columns per row tile in each buffer
    negmax, csum = [], []                      # per buffer: -max of each chunk, sum exp(s - max) of each chunk
    for j in range(nb):
        negmax.append(nl.ndarray((P, R * W), dtype=nl.float32, buffer=nl.sbuf))
        csum.append(nl.ndarray((P, R * W), dtype=nl.float32, buffer=nl.sbuf))
        if nch % nb:                           # unused slots: -max = 1e30 (ignored by the min), sum = 0
            nisa.memset(dst=negmax[j], value=1e30)
            nisa.memset(dst=csum[j], value=0.0)
    for b0 in range(0, m, cb):
        bw = min(cb, m - b0)
        yb = load_block(b0, bw)
        for r in range(R):
            for c in range(bw // cw):
                kr = b0 // cw + c
                j, k = kr % nb, r * W + kr // nb
                S = nl.ndarray((P, cw), dtype=nl.float32, buffer=nl.psum)
                for q in range(cw // MM):
                    o = c * cw + q * MM
                    nisa.nc_matmul(dst=S[:, q * MM:(q + 1) * MM], stationary=xs[0:K, r * P:(r + 1) * P],
                                   moving=yb[0:K, o:o + MM])
                nisa.tensor_reduce(dst=negmax[j][:, k:k + 1], op=nl.maximum, data=S, axis=1, negate=True,
                                   keepdims=True)
                junk = nl.ndarray((P, cw), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=junk, op=nl.exp, data=S, bias=negmax[j][:, k:k + 1], scale=1.0,
                                reduce_op=nl.add, reduce_res=csum[j][:, k:k + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
    res = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    for r in range(R):
        cols = slice(r * W, (r + 1) * W)
        mns = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)           # per buffer: -(its max)
        for j in range(nb):
            nisa.tensor_reduce(dst=mns[:, j:j + 1], op=nl.minimum, data=negmax[j][:, cols], axis=1, keepdims=True)
        mn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)              # -M, M = the row's max
        nisa.tensor_reduce(dst=mn, op=nl.minimum, data=mns, axis=1, keepdims=True)
        ls = nl.ndarray((P, nb), dtype=nl.float32, buffer=nl.sbuf)            # per buffer: sum_k s_k exp(m_k - M)
        for j in range(nb):
            w = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)           # exp(m_k - M) = exp(-nm_k + mn)
            nisa.activation(dst=w, op=nl.exp, data=negmax[j][:, cols], scale=-1.0, bias=mn)
            nisa.tensor_tensor(dst=w, data1=w, data2=csum[j][:, cols], op=nl.multiply)
            nisa.tensor_reduce(dst=ls[:, j:j + 1], op=nl.add, data=w, axis=1, keepdims=True)
        l = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=l, op=nl.add, data=ls, axis=1, keepdims=True)
        lg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=lg, op=nl.log, data=l)
        nisa.tensor_tensor(dst=res[:, r:r + 1], data1=lg, data2=mn, op=nl.subtract)   # log l + M
    return res


def _check(K, rows, m, cw, cb):
    if not HAVE_NKI:
        raise ImportError("the nki package is not installed")
    assert K <= P and rows % P == 0 and m % cw == 0 and cw % MM == 0 and cb % cw == 0, (K, rows, m, cw, cb)


def _jit(f, name: str, key: tuple):
    tag = hashlib.sha1(repr((KERNEL_VERSION,) + key).encode()).hexdigest()[:16]
    f.__name__ = f.__qualname__ = f"{name}_{tag}"
    _KERNELS[key] = kernel = nki.jit(f)
    return kernel


def make_row_lse_kernel(K: int, rows: int, m: int, cw: int = CW, cb: int = CB, nb: int = NB):
    """kernel(xa_T (K, rows) f32, ya_T (K, m) f32) -> lse (rows,) f32. rows % 128 == 0, m % cw == 0."""
    _check(K, rows, m, cw, cb)
    key = ("lse", K, rows, m, cw, cb, nb)
    if key in _KERNELS:
        return _KERNELS[key]
    R = rows // P

    def row_lse(xa_T, ya_T):
        out = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)
        xs = nl.ndarray((P, rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs[0:K, :], src=xa_T[0:K, :])

        def load_block(b0, bw):
            yb = nl.ndarray((P, bw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=yb[0:K, :], src=ya_T[0:K, b0:b0 + bw])
            return yb

        res = _row_lse(xs, K, rows, m, cw, cb, load_block, nb)
        nisa.dma_copy(dst=out.ap(pattern=[[1, P], [P, R]], offset=0), src=res)
        return out

    return _jit(row_lse, "sinkhorn_row_lse", key)


def make_half_step_kernel(d: int, rows: int, m: int, cw: int = CW, cb: int = CB, nb: int = NB):
    """kernel(xa_T (d+1, rows), xsq (rows,), off (rows,), yT (d, m), gh (m,), log_b (m,), sc (1, 4)) -> fh (rows,),
    all f32.

    xa_T = [x^T ; 1] for this call's rows, xsq = |x|^2 for them, off = log(target / reference) for them, yT / gh /
    log_b = the other side (points, shifted potential, log weights), sc = [2 / eps, 1 / eps, eps, kappa].
    Returns fh = f - |x|^2 for the rows.
    """
    K = d + 1
    _check(K, rows, m, cw, cb)
    key = ("half", d, rows, m, cw, cb, nb)
    if key in _KERNELS:
        return _KERNELS[key]
    R = rows // P

    def half_step(xa_T, xsq, off, yT, gh, log_b, sc):
        out = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)
        SC = nl.ndarray((P, 4), dtype=nl.float32, buffer=nl.sbuf)              # the scalars on every partition
        nisa.dma_copy(dst=SC, src=sc.ap(pattern=[[0, P], [1, 4]], offset=0))
        xs = nl.ndarray((P, rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs[0:K, :], src=xa_T[0:K, :])

        def load_block(b0, bw):
            # ya_T block: rows 0..d-1 = y^T * (2 / eps), row d = gh / eps + log b. Compute engines work from
            # partition 0, so the bias row is computed on partition 0 and moved to partition d by DMA.
            raw = nl.ndarray((P, bw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=raw[0:d, :], src=yT[0:d, b0:b0 + bw])
            yb = nl.ndarray((P, bw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=yb[0:d, :], data=raw[0:d, :], op0=nl.multiply, operand0=SC[0:d, 0:1])
            g = nl.ndarray((1, bw), dtype=nl.float32, buffer=nl.sbuf)
            lb = nl.ndarray((1, bw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g, src=gh.ap(pattern=[[bw, 1], [1, bw]], offset=b0))
            nisa.dma_copy(dst=lb, src=log_b.ap(pattern=[[bw, 1], [1, bw]], offset=b0))
            nisa.scalar_tensor_tensor(dst=g, data=g, op0=nl.multiply, operand0=SC[0:1, 1:2], op1=nl.add, operand1=lb)
            nisa.dma_copy(dst=yb[d:K, :], src=g)
            return yb

        lse = _row_lse(xs, K, rows, m, cw, cb, load_block, nb)
        xq = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xq, src=xsq.ap(pattern=[[1, P], [P, R]], offset=0))
        oq = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=oq, src=off.ap(pattern=[[1, P], [P, R]], offset=0))
        u = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=u, data=oq, op0=nl.multiply, operand0=SC[:, 2:3],       # eps * off + |x|^2
                                  op1=nl.add, operand1=xq)
        t = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t, data=lse, op0=nl.multiply, operand0=SC[:, 2:3])            # eps * lse
        nisa.tensor_tensor(dst=t, data1=u, data2=t, op=nl.subtract)                           # ... - eps * lse
        nisa.scalar_tensor_tensor(dst=t, data=t, op0=nl.multiply, operand0=SC[:, 3:4],       # * kappa - |x|^2
                                  op1=nl.subtract, operand1=xq)
        nisa.dma_copy(dst=out.ap(pattern=[[1, P], [P, R]], offset=0), src=t)
        return out

    return _jit(half_step, "sinkhorn_half_step", key)
