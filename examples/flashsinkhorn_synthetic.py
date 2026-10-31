"""FlashSinkhorn on a CUDA GPU, on the synthetic setup of its paper's Table 2 (the GPU side of the synthetic table).

    python examples/flashsinkhorn_synthetic.py            # strict fp32
    python examples/flashsinkhorn_synthetic.py --tf32     # TF32 matmuls (Ampere+), FlashSinkhorn's default

n = m = 10k, d = 64, eps = 0.1, 10 iterations, balanced, uniform weights; the same Gaussian-blob clouds as
sinkhorn_benchmark.py --flashsinkhorn. Times the alternating and symmetric solvers (best and median of --reps runs,
after a warm-up that autotunes and compiles). Needs flash-sinkhorn (github.com/ot-triton-lab/flash-sinkhorn) on
PYTHONPATH (its torch-ext directory).

Note: results/flashsinkhorn_synth_a100.log was produced by an earlier throwaway version of this script; this one
reconstructs it with the same settings and output format.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinkhorn_benchmark import mixture                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--gpu-name", default=torch.cuda.get_device_name(0).split()[-1] if torch.cuda.is_available() else "GPU")
    a = ap.parse_args()
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating, sinkhorn_flashstyle_symmetric

    x = torch.as_tensor(mixture(a.n, a.d, k=20, seed=0), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(mixture(a.n, a.d, k=20, seed=1, shift=0.3), dtype=torch.float32, device="cuda")
    w = torch.full((a.n,), 1.0 / a.n, device="cuda")
    solvers = {
        "alternating": lambda: sinkhorn_flashstyle_alternating(x, y, w, w, eps=a.eps, n_iters=a.iters,
                                                               allow_tf32=a.tf32, autotune=True),
        "symmetric": lambda: sinkhorn_flashstyle_symmetric(x, y, w, w, use_epsilon_scaling=False, eps=a.eps,
                                                           n_iters=a.iters, allow_tf32=a.tf32),
    }
    prec = "tf32" if a.tf32 else "fp32"
    for name, run in solvers.items():
        run()                                                   # warm-up: autotune + compile
        torch.cuda.synchronize()
        times = []
        for _ in range(a.reps):
            t = time.perf_counter()
            run()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
        times.sort()
        print(f"FlashSinkhorn {name} {a.gpu_name} {prec}, {a.n // 1000}k x {a.n // 1000}k d {a.d} eps {a.eps:g}, "
              f"{a.iters} its: best {times[0]:.1f} ms, median {times[len(times) // 2]:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
