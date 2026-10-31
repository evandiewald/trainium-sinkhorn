"""Check (and time) the fused row-LSE kernel (trnopt.nki_sinkhorn) against NumPy float64.

  python examples/nki_sinkhorn_check.py --sim                      # NKI CPU simulator, small shapes
  python examples/nki_sinkhorn_check.py --rows 10240 --m 10240     # on the chip (TorchNeuron), with timing

Inputs mimic a real half-step: points from Gaussian blobs, normalized like sinkhorn.py, eps = 0.1, and a
potential vector gh so the scores have realistic spread. Reports the max |lse - lse_ref| (absolute: lse is
multiplied by eps in the potential update, so absolute error in lse is error in units of eps).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.special import logsumexp

from trnopt.nki_sinkhorn import make_half_step_kernel, make_row_lse_kernel, pad_cols


def inputs(rows, m, d, eps, seed=0, pad_from=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, d)) * 0.15
    y = rng.normal(size=(m, d)) * 0.15 + 0.02
    gh = -(y ** 2).sum(1) + rng.normal(scale=0.05, size=m)            # plausible shifted potential
    log_b = np.full(m, -np.log(m))
    if pad_from is not None:                                         # padding columns, as sinkhorn.py makes them
        y[pad_from:] = 0.0
        gh[pad_from:] = 0.0
        log_b[pad_from:] = -1e9
    xa_T = np.vstack([x.T, np.ones((1, rows))]).astype(np.float32)
    ya_T = np.vstack([2.0 / eps * y.T, (gh / eps + log_b)[None, :]]).astype(np.float32)
    ref = logsumexp(xa_T.astype(np.float64).T @ ya_T.astype(np.float64), axis=1)
    return xa_T, ya_T, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--real-m", type=int, default=None, help="columns before padding (default: m - 300)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cw", type=int, default=None, help="columns per reduction chunk (default: the kernel's CW)")
    ap.add_argument("--cb", type=int, default=None, help="columns per ya_T DMA block")
    ap.add_argument("--nb", type=int, default=None, help="rotating stat buffers")
    ap.add_argument("--half", action="store_true", help="the whole half-step kernel (builds ya_T, returns fh)")
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--device", default="neuron", help="e.g. neuron:1 for the second core")
    a = ap.parse_args()
    m = pad_cols(a.m)
    real = a.real_m if a.real_m is not None else m - 300
    xa_T, ya_T, ref = inputs(a.rows, m, a.d, a.eps, pad_from=real)
    kw = {k_: v for k_, v in (("cw", a.cw), ("cb", a.cb), ("nb", a.nb)) if v}
    k = make_row_lse_kernel(a.d + 1, a.rows, m, **kw)
    args = (xa_T, ya_T)
    if a.half:                                   # same scores, built in the kernel from the raw pieces
        d = a.d
        yT = np.ascontiguousarray(ya_T[:d] * (a.eps / 2.0))
        log_b = np.full(m, -np.log(real), dtype=np.float32)
        log_b[real:] = -1e9
        gh = ((ya_T[d] - log_b) * a.eps).astype(np.float32)
        xsq = (xa_T[:d] ** 2).sum(0).astype(np.float32)
        sc = np.array([[2.0 / a.eps, 1.0 / a.eps, a.eps, a.kappa]], dtype=np.float32)
        ya64 = np.vstack([yT.astype(np.float64) * (2.0 / a.eps), (gh.astype(np.float64) / a.eps + log_b)[None]])
        ref = logsumexp(xa_T.astype(np.float64).T @ ya64, axis=1)
        off = np.random.default_rng(3).normal(scale=0.3, size=a.rows).astype(np.float32)    # log target / ref
        ref = a.kappa * (xsq + a.eps * off - a.eps * ref) - xsq
        k = make_half_step_kernel(d, a.rows, m, **kw)
        args = (xa_T, xsq, off, yT, gh, log_b, sc)
    if a.sim:
        import nki
        t = time.perf_counter()
        out = np.asarray(nki.simulate(k)(*args))
        print(f"sim{' half-step' if a.half else ''} {a.rows} x {m} (real {real}), d {a.d}: {time.perf_counter() - t:.1f} s, "
              f"max |err| {np.abs(out - ref).max():.3e}, lse range [{ref.min():.2f}, {ref.max():.2f}]")
        return
    import torch
    import torch_neuronx
    T = [torch.from_numpy(np.ascontiguousarray(v)).to(a.device) for v in args]
    t = time.perf_counter()
    out = k(*T)
    torch_neuronx.synchronize()
    first = time.perf_counter() - t
    err = np.abs(out.cpu().numpy() - ref).max()
    t = time.perf_counter()
    for _ in range(a.reps):
        out = k(*T)
    torch_neuronx.synchronize()
    out.cpu()
    dt = (time.perf_counter() - t) / a.reps
    print(f"chip{' half-step' if a.half else ''} {a.device} {a.rows} x {m} (real {real}), d {a.d}, {kw}: first call {first:.1f} s, {dt * 1e3:.3f} ms/call, "
          f"{a.rows * m / dt:.3g} scores/s, max |err| {err:.3e}, lse range [{ref.min():.2f}, {ref.max():.2f}]")


if __name__ == "__main__":
    main()
