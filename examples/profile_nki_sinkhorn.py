"""Profile one call of the fused row-LSE kernel (trnopt.nki_sinkhorn) with neuron-explorer.

    python examples/profile_nki_sinkhorn.py --rows 10240 --m 10000 --tag v2

Compiles the kernel standalone (NumPy inputs, so the NEFF holds just this kernel), then captures a profile
with real inputs and writes results/profiles/sinkhorn_nki_<rows>x<m>_<tag>.{neff,ntff,pftrace,_summary.txt}.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WORK = os.environ.setdefault("NKI_ARTIFACTS_DIR", tempfile.mkdtemp(prefix="nki_sk_profile_"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nki_sinkhorn_check import inputs                            # noqa: E402
from trnopt.nki_sinkhorn import make_row_lse_kernel, pad_cols   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPLORER = shutil.which("neuron-explorer") or "/opt/aws/neuron/bin/neuron-explorer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10240)
    ap.add_argument("--m", type=int, default=10000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--out", default=str(ROOT / "results" / "profiles"))
    a = ap.parse_args()
    m = pad_cols(a.m)
    xa_T, ya_T, _ = inputs(a.rows, m, a.d, 0.1, pad_from=a.m)
    kernel = make_row_lse_kernel(a.d + 1, a.rows, m)
    try:
        kernel(xa_T, ya_T)                       # NumPy inputs: standalone compile; writes the NEFF
    except ImportError:
        pass
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"sinkhorn_nki_{a.rows}x{m}_{a.tag}"
    neff = out / f"{name}.neff"
    shutil.copy(max(Path(WORK).rglob("*.neff"), key=lambda p: p.stat().st_mtime), neff)
    ifmaps = []
    for arg, v in (("xa_T", xa_T), ("ya_T", ya_T)):
        np.save(Path(WORK) / f"{arg}.npy", np.ascontiguousarray(v))
        ifmaps += [arg, str(Path(WORK) / f"{arg}.npy")]
    ntff = out / f"{name}.ntff"
    run = lambda *x: subprocess.run([EXPLORER, *x], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run("capture", "-n", str(neff), "-s", str(ntff), *ifmaps)
    view = ["view", "-n", str(neff), "-s", str(ntff), "--disable-ui"]
    with open(out / f"{name}_summary.txt", "w") as f:
        subprocess.run([EXPLORER, *view, "--output-format", "summary-text"], check=True, stdout=f,
                       stderr=subprocess.DEVNULL)
    run(*view, "--output-format", "perfetto", "--output-file", str(out / f"{name}.pftrace"))
    print(f"Wrote {out / name}.*", flush=True)


if __name__ == "__main__":
    main()
