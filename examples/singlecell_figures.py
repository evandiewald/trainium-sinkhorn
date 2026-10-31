"""Ancestor figure for the Waddington-OT reprogramming data: where did the day-18 iPSCs come from?

    python examples/singlecell_figures.py --data ~/wotdata/data --potentials ~/sc_potentials \
        [--wot-tmaps ~/wotdata/tmaps] --out results/figures/ancestors_ips.png

Starts from the cells of one set (default IPS) at the last day and pulls the distribution back through the chain
of transport maps, one consecutive pair at a time (wot's ancestor computation: p_{t0} ∝ T_{t0,t1} p_{t1}, renormalized).
trnopt's maps come from the saved potentials of examples/singlecell_wot.py --save (never materialized: each pull-back
is a log-sum-exp over the pair's cost, on the host in float64). With --wot-tmaps (wot's own transport maps, .h5ad per
pair, e.g. the tutorial's precomputed ones), a second row shows wot's ancestors and each panel reports the total-variation
distance between the two.

Panels: the force-directed layout of all serum cells in light grey, that day's cells in darker grey, and the cells
carrying ancestral mass (over 2% of the day's maximum) colored by ancestor probability.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def read_sets(gmt: str) -> dict:
    sets = {}
    for line in open(gmt):
        f = line.rstrip("\n").split("\t")
        sets[f[0]] = set(f[2:])
    return sets


def pull_back_potentials(npz, log_v1: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """log of (P v) up to a constant, P_ij ∝ exp((f_i + g_j - C_ij) / eps), from the saved potentials."""
    z = np.load(npz, allow_pickle=True)
    s = np.sqrt(float(z["cost_scale"]))
    x0, x1 = (z["x0"] - z["center"]) / s, (z["x1"] - z["center"]) / s
    f, g, eps = z["f"], z["g"], float(z["eps"])
    y2 = (x1 * x1).sum(1)
    out = np.empty(len(x0))
    for i in range(0, len(x0), chunk):
        xb = x0[i:i + chunk]
        C = (xb * xb).sum(1)[:, None] + y2[None, :] - 2.0 * xb @ x1.T
        out[i:i + chunk] = logsumexp((g[None, :] - C) / eps + log_v1[None, :], axis=1) + f[i:i + chunk] / eps
    return out - logsumexp(out)


def wot_tmap_file(d: str, t0: float, t1: float):
    for pat in (f"*_{t0:g}_{t1:g}.h5ad", f"*_{t0}_{t1}.h5ad", f"*{t0:g}_{t1:g}*.h5ad"):
        hits = glob.glob(str(Path(d) / pat))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--potentials", required=True, help="directory of pair_<t0>_<t1>.npz from singlecell_wot.py --save")
    ap.add_argument("--wot-tmaps", default=None, help="directory of wot transport maps (.h5ad per pair)")
    ap.add_argument("--set", default="IPS", help="cell set in major_cell_sets.gmt (IPS, Stromal, Neural, ...)")
    ap.add_argument("--show-days", default="2,4,6,8,10,12,14,16")
    ap.add_argument("--out", default="results/figures/ancestors_ips.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    fle = pd.read_csv(f"{a.data}/fle_coords.txt", sep="\t", index_col=0)
    serum = set(pd.read_csv(f"{a.data}/serum_cell_ids.txt", header=None)[0])
    fle = fle[fle.index.isin(serum)]
    target = read_sets(f"{a.data}/major_cell_sets.gmt")[a.set]
    pairs = sorted((float(p.stem.split("_")[1]), float(p.stem.split("_")[2]), p)
                   for p in Path(a.potentials).glob("pair_*.npz"))
    last = pairs[-1]
    z = np.load(last[2], allow_pickle=True)
    cells = {last[1]: z["cells1"]}
    in_set = np.array([c in target for c in z["cells1"]])
    logv = {last[1]: np.where(in_set, -np.log(in_set.sum()), -np.inf)}
    wv = {last[1]: np.exp(logv[last[1]])} if a.wot_tmaps else None
    for t0, t1, path in reversed(pairs):
        zz = np.load(path, allow_pickle=True)
        assert np.array_equal(zz["cells1"], cells[t1]), f"cell order mismatch at {t1}"
        logv[t0] = pull_back_potentials(path, logv[t1])
        cells[t0] = zz["cells0"]
        if wv is not None:
            fpath = wot_tmap_file(a.wot_tmaps, t0, t1)
            if fpath is None or t1 not in wv:
                continue
            import anndata
            T = anndata.read_h5ad(fpath)
            T = T[[str(c) for c in cells[t0]], [str(c) for c in cells[t1]]]
            v = np.asarray(T.X @ wv[t1]).ravel()
            wv[t0] = v / v.sum()

    show = [float(d) for d in a.show_days.split(",") if float(d) in logv]
    rows = 2 if wv else 1
    fig, axes = plt.subplots(rows, len(show), figsize=(2.6 * len(show), 2.8 * rows), squeeze=False)
    for k, day in enumerate(show):
        p = np.exp(logv[day])
        xy = fle.loc[[str(c) for c in cells[day]]].values
        for r in range(rows):
            ax = axes[r, k]
            ax.scatter(fle.x, fle.y, s=0.2, c="#d9d9d9", linewidths=0, rasterized=True)
            q = p if r == 0 else wv.get(day)
            if q is None:
                ax.set_axis_off()
                continue
            ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="#9e9e9e", linewidths=0, rasterized=True)   # this day's cells
            rel = q / q.max()
            keep = rel > 0.02                       # only cells carrying real ancestral mass get a color
            order = np.flatnonzero(keep)[np.argsort(rel[keep])]
            ax.scatter(xy[order, 0], xy[order, 1], s=1.0, c=rel[order], cmap="plasma", vmin=0, vmax=1,
                       linewidths=0, rasterized=True)
            title = f"day {day:g}"
            if r == 1:
                title += f"\nTV vs trnopt {0.5 * np.abs(q - p).sum():.1e}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, 0].set_ylabel("trnopt (Trainium)", fontsize=9)
        if rows == 2:
            axes[1, 0].set_ylabel("wot (reference)", fontsize=9)
    fig.suptitle(f"Ancestors of day-{last[1]:g} {a.set} cells, pulled back through the transport maps", fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"-> {a.out}")
    np.savez(Path(a.out).with_suffix(".npz"), **{f"trnopt_{d:g}": np.exp(v) for d, v in logv.items()},
             **({f"wot_{d:g}": v for d, v in wv.items()} if wv else {}))


if __name__ == "__main__":
    main()
