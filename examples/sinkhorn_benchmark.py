"""Time plain-PyTorch Sinkhorn (trnopt.sinkhorn) on a backend: the pre-NKI baseline.

  python examples/sinkhorn_benchmark.py --backend neuron --n 10000 --m 10000 --d 64 --block 2048
  python examples/sinkhorn_benchmark.py --backend neuron --flashsinkhorn          # their Table 2 setup
  python examples/sinkhorn_benchmark.py --backend cpu-float32 --data cells.npz    # x, y (and optional a, b)

Two runs per configuration, both after one untimed warm-up (which compiles):
  * fixed: exactly --iters iterations, no convergence checks. Gives ms per iteration and "pair updates" per
    second (2 n m per iteration: every x-y pair is scored once in the f update and once in the g update).
  * converge: to --tol at --eps (with --eps-start for eps-scaling), reporting iterations and the final error.
--check reruns the converge case in float64 on the host CPU and reports the cost difference.

FlashSinkhorn (arXiv 2602.03067, Table 2, A100): n = m = 10,000, d = 64, eps = 0.1, 10 iterations, forward
only: 8.2 ms with their fused Triton kernel, 54.0 ms for the dense ("tensorized") GeomLoss baseline, 125.5 ms
for KeOps. Their cost isn't normalized, so --flashsinkhorn turns normalization off; timing for a fixed number
of iterations doesn't depend on the data anyway.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import socket
from pathlib import Path

import numpy as np

from trnopt import Backend
from trnopt.sinkhorn import SinkhornOptions, sinkhorn

FIELDS = ["date", "host", "backend", "kernel", "cores", "n", "m", "d", "block", "block_y", "eps", "rho", "eps_start", "fixed_iters",
          "fixed_ms", "ms_per_iter", "pair_updates_per_s", "compile_s", "conv_iters", "conv_s", "conv_error",
          "converged", "cost", "ref_cost", "ref_s", "cost_rel_diff"]
FLASH_A100_MS = {"FlashSinkhorn": 8.2, "GeomLoss tensorized": 54.0, "KeOps": 125.5}   # 10k x 10k, d=64, 10 its


def mixture(n: int, d: int, k: int, seed: int, shift: float = 0.0) -> np.ndarray:
    """n points from k Gaussian blobs in d dimensions (a stand-in for cells in PCA space)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(k, d))
    lab = rng.integers(k, size=n)
    return centers[lab] + rng.normal(size=(n, d)) + shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="neuron", help="Backend preset: neuron, cpu, cpu-float32, emulate-neuron, ...")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--m", type=int, default=None, help="default: n")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--data", help=".npz with x (n, d), y (m, d), optional a, b; overrides --n/--m/--d")
    ap.add_argument("--block", type=int, default=None, help="rows per block (default: all rows at once)")
    ap.add_argument("--block-y", type=int, default=None)
    ap.add_argument("--kernel", default="auto", help="half-step: auto, torch or nki")
    ap.add_argument("--cores", type=int, default=1, help="NeuronCores to split rows across (nki only)")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--eps-start", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None, help="unbalanced marginal penalty (default: balanced)")
    ap.add_argument("--iters", type=int, default=10, help="iterations for the fixed-iteration timing")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--no-converge", action="store_true", help="skip the run to tolerance")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--flashsinkhorn", action="store_true", help="n = m = 10k, d = 64, eps = 0.1, 10 its, raw cost")
    ap.add_argument("--check", action="store_true", help="float64 host-CPU rerun of the converge case")
    ap.add_argument("--out", default=None, help="CSV to append to (default results/sinkhorn_<host>_<date>.csv)")
    a = ap.parse_args()

    if a.flashsinkhorn:
        a.n, a.m, a.d, a.eps, a.iters, a.no_normalize, a.data = 10_000, 10_000, 64, 0.1, 10, True, None
    if a.data:
        z = np.load(a.data)
        x, y = z["x"], z["y"]
        wa, wb = (z["a"] if "a" in z else None), (z["b"] if "b" in z else None)
    else:
        m = a.m or a.n
        x, y = mixture(a.n, a.d, k=20, seed=0), mixture(m, a.d, k=20, seed=1, shift=0.3)
        wa = wb = None
    n, m, d = len(x), len(y), x.shape[1]
    be = Backend.preset(a.backend)
    base = dict(eps=a.eps, rho=a.rho, block=a.block, block_y=a.block_y, normalize=not a.no_normalize,
                kernel=a.kernel, cores=a.cores, backend=be)
    print(f"{a.backend}/{a.kernel}/{a.cores} core(s): n {n:,} m {m:,} d {d}, block {a.block or n}/{a.block_y or a.block or m}, eps {a.eps}"
          f"{'' if a.rho is None else f', rho {a.rho}'}", flush=True)

    fixed = SinkhornOptions(max_iters=a.iters, fixed_iters=True, **base)
    warm = sinkhorn(x, y, wa, wb, fixed)                       # compiles every shape the timed runs use
    print(f"  warm-up (incl. compile): {warm.seconds:.2f} s", flush=True)
    times = [sinkhorn(x, y, wa, wb, fixed).seconds for _ in range(a.reps)]
    t = min(times)
    ms_it = t / a.iters * 1e3
    pairs = 2.0 * n * m * a.iters / t
    print(f"  fixed {a.iters} its: best {t * 1e3:.1f} ms of {a.reps} ({ms_it:.2f} ms/it, {pairs:.3g} pair updates/s)")
    if (n, m, d, a.iters) == (10_000, 10_000, 64, 10):
        for k, v in FLASH_A100_MS.items():
            print(f"    vs A100 {k} {v} ms: {v / (t * 1e3):.3g}x")

    row = dict(date=dt.date.today().isoformat(), host=socket.gethostname(), backend=a.backend, kernel=a.kernel, cores=a.cores, n=n, m=m, d=d,
               block=a.block or n, block_y=a.block_y or a.block or m, eps=a.eps, rho=a.rho, eps_start=a.eps_start,
               fixed_iters=a.iters, fixed_ms=round(t * 1e3, 3), ms_per_iter=round(ms_it, 4),
               pair_updates_per_s=f"{pairs:.4g}", compile_s=round(max(0.0, warm.seconds - t), 2))

    if not a.no_converge:
        conv = SinkhornOptions(max_iters=a.max_iters, tol=a.tol, eps_start=a.eps_start, **base)
        r = sinkhorn(x, y, wa, wb, conv)
        print(f"  to tol {a.tol}: {r.iterations} its, {r.seconds:.2f} s, error {r.error:.2e}, "
              f"converged {r.converged}, cost {r.cost:.6g}")
        row.update(conv_iters=r.iterations, conv_s=round(r.seconds, 3), conv_error=f"{r.error:.3e}",
                   converged=r.converged, cost=r.cost)
        if a.check:
            ref = sinkhorn(x, y, wa, wb, SinkhornOptions(max_iters=a.max_iters, tol=a.tol, eps_start=a.eps_start,
                                                         **{**base, "kernel": "torch", "cores": 1,
                                                            "backend": Backend.preset("cpu")}))
            diff = abs(r.cost - ref.cost) / (1 + abs(ref.cost))
            print(f"  float64 CPU: {ref.iterations} its, {ref.seconds:.2f} s, cost {ref.cost:.6g}, rel diff {diff:.2e}")
            row.update(ref_cost=ref.cost, ref_s=round(ref.seconds, 3), cost_rel_diff=f"{diff:.2e}")

    out = Path(a.out or f"results/sinkhorn_{socket.gethostname()}_{dt.date.today().isoformat()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
