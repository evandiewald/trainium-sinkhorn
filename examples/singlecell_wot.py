"""Waddington-OT transport maps for every consecutive pair of time points, trnopt vs wot.

    python examples/singlecell_wot.py --data ~/wotdata/data --backend neuron --cores 2 --growth-iters 3
    python examples/singlecell_wot.py --data ~/wotdata/data --backend neuron --pairs 3.5:4.0 16:16.5 --wot

Data: the wot tutorial bundle (Google Drive id 1E494DhIx5RLy0qv_6eWa9426Bfmq28po; `gdown` it and unzip), serum
cells only (as the tutorial). Per pair (t0, t1): wot's local PCA (30 components), growth = cell_growth_rate^(t1-t0),
eps 0.05, lambda1 1, lambda2 50, `--growth-iters` passes (tutorial: 3). Writes per-pair rows to a CSV:
cells, PCA time, solve time (and first-call compile separately), Sinkhorn iterations, and with --wot the wot time
(its whole compute_transport_map: PCA + cost + solve) and the agreement of the two maps (relative Frobenius error,
learned-growth error, max total-variation distance between the cells' descendant distributions).
--save DIR writes each pair's potentials and learned growth (npz) for the ancestor / fate figures.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import socket
import time
from pathlib import Path

import numpy as np

from trnopt import Backend
from trnopt.singlecell import local_pca_coords, wot_tmap, wot_transport_map
from trnopt.sinkhorn import SinkhornOptions

FIELDS = ["date", "host", "backend", "cores", "t0", "t1", "n0", "n1", "growth_iters", "pca_s", "solve_s",
          "iterations", "wot_s", "rel_frobenius", "growth_max_rel", "descendant_max_tv"]


def load(data: str):
    import anndata
    import pandas as pd
    ad = anndata.read_h5ad(f"{data}/ExprMatrix.var.genes.h5ad")
    days = pd.read_csv(f"{data}/cell_days.txt", sep="\t", index_col=0)["day"]
    growth = pd.read_csv(f"{data}/growth_gs_init.txt", sep=",", index_col=0)["cell_growth_rate"]
    serum = pd.read_csv(f"{data}/serum_cell_ids.txt", header=None)[0]
    keep = ad.obs_names.isin(serum) & ad.obs_names.isin(days.index)
    ad = ad[keep].copy()
    ad.obs["day"] = days.loc[ad.obs_names].values
    ad.obs["cell_growth_rate"] = growth.reindex(ad.obs_names).fillna(1.0).values
    return ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with the wot tutorial files")
    ap.add_argument("--backend", default="neuron")
    ap.add_argument("--kernel", default="auto")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--growth-iters", type=int, default=3)
    ap.add_argument("--pairs", nargs="*", help="t0:t1 pairs (default: every consecutive pair)")
    ap.add_argument("--wot", action="store_true", help="also run wot on the host CPU and compare the maps")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--save", default=None, help="directory for per-pair potentials / learned growth (npz)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ad = load(a.data)
    days = np.sort(ad.obs.day.unique())
    pairs = ([tuple(float(v) for v in p.split(":")) for p in a.pairs] if a.pairs
             else list(zip(days[:-1], days[1:])))
    be = Backend.preset(a.backend)
    opts = SinkhornOptions(backend=be, kernel=a.kernel, cores=a.cores, max_iters=50_000, check_every=5)
    out = Path(a.out or f"results/singlecell_wot_{socket.gethostname()}_{dt.date.today().isoformat()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
    if new:
        w.writeheader()
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
    wot_model = None
    if a.wot:
        import anndata
        anndata.read = anndata.read_h5ad                 # wot 1.0.8 calls the removed anndata.read
        import wot
        wot_model = wot.ot.OTModel(ad, epsilon=0.05, lambda1=1, lambda2=50, growth_iters=a.growth_iters)
    total = 0.0
    for t0, t1 in pairs:
        s0, s1 = ad.obs.day.values == t0, ad.obs.day.values == t1
        t = time.perf_counter()
        x0, x1 = local_pca_coords(ad.X[s0], ad.X[s1], 30)
        pca_s = time.perf_counter() - t
        tm = wot_transport_map(x0, x1, ad.obs.cell_growth_rate.values[s0], t1 - t0,
                               growth_iters=a.growth_iters, tol=a.tol, opts=opts)
        total += pca_s + tm.seconds
        row = dict(date=dt.date.today().isoformat(), host=socket.gethostname(), backend=a.backend, cores=a.cores,
                   t0=t0, t1=t1, n0=int(s0.sum()), n1=int(s1.sum()), growth_iters=a.growth_iters,
                   pca_s=round(pca_s, 3), solve_s=round(tm.seconds, 3), iterations=tm.iterations)
        msg = (f"{t0:5.2f} -> {t1:5.2f}: {row['n0']:5d} x {row['n1']:5d}  PCA {pca_s:5.2f} s  solve {tm.seconds:7.2f} s"
               f"  {tm.iterations:5d} its")
        if wot_model is not None:
            t = time.perf_counter()
            ref = wot_model.compute_transport_map(t0, t1)
            row["wot_s"] = round(time.perf_counter() - t, 2)
            R = np.asarray(ref.X)
            T = wot_tmap(x0, x1, tm)
            D_w, D_o = R / R.sum(1, keepdims=True), T / T.sum(1, keepdims=True)
            row.update(rel_frobenius=f"{np.linalg.norm(T - R) / np.linalg.norm(R):.2e}",
                       growth_max_rel=f"{np.abs(T.sum(1) / R.sum(1) - 1).max():.2e}",
                       descendant_max_tv=f"{0.5 * np.abs(D_o - D_w).sum(1).max():.2e}")
            msg += (f"  | wot {row['wot_s']:7.1f} s ({row['wot_s'] / (pca_s + tm.seconds):5.1f}x)  map err "
                    f"{row['rel_frobenius']}  growth err {row['growth_max_rel']}  TV {row['descendant_max_tv']}")
        print(msg, flush=True)
        w.writerow(row)
        fh.flush()
        if a.save:
            r = tm.result
            np.savez(Path(a.save) / f"pair_{t0:g}_{t1:g}.npz", f=r.f, g=r.g, eps=r.eps, center=r.center,
                     cost_scale=r.cost_scale, row_mass=r.row_mass, col_mass=r.col_mass,
                     learned_growth=np.array(tm.learned_growth), cells0=ad.obs_names[s0].values,
                     cells1=ad.obs_names[s1].values, x0=x0, x1=x1)
    print(f"total (PCA + solve, incl. compile): {total:.1f} s over {len(pairs)} pairs  -> {out}")


if __name__ == "__main__":
    main()
