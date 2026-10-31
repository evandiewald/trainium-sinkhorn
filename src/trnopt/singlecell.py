"""Waddington-OT-style transport maps between consecutive time points, solved with trnopt's Sinkhorn.

Reproduces wot (Schiebinger et al., Cell 2019; pip `wot`, OTModel defaults) so its transport maps, and the
ancestor / fate analyses built on them, can be compared one to one:

* Coordinates: a local PCA of the two time points' cells (30 components), scaled by the singular values, the
  same projection as wot.ot.compute_pca + compute_default_cost_matrix.
* Cost: squared Euclidean distance divided by its median (normalize="median").
* Formulation: uniform reference measure (1/I, 1/J); rows pulled toward the cells' growth G = g^(t1 - t0)
  (wot's p), columns toward the mean growth (wot's q), with KL penalties lambda1 (rows) and lambda2 (columns);
  entropy eps. In our terms: a = 1/I, b = 1/J, a_target = G / I, b_target = mean(G) / J, rho = (lambda1, lambda2).
* eps-scaling as wot's default solver (optimal_transport_duality_gap): eps0 = 1, then 5 geometric steps down to
  eps; each intermediate stage until the potentials settle, the last until tol.
* growth_iters: re-solve with the row target set to the previous plan's row sums (wot's learned growth).

wot returns tmap = I * P (P = our plan, which sums to the transported mass), so tmap rows sum to wot's
learned growth G_next = I * row_mass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from .sinkhorn import SinkhornOptions, SinkhornResult, normalization, sinkhorn, transport_plan


def local_pca_coords(X0, X1, n_components: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """wot's per-pair embedding: PCA of the two time points' (centered) expression together, as wot.ot.compute_pca
    does it (sklearn PCA fit on the transpose), then scaled by the singular values (compute_default_cost_matrix's
    eigenvals). Returns (cells0 x k, cells1 x k)."""
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    m1 = X0.toarray() if sp.issparse(X0) else np.asarray(X0)
    m2 = X1.toarray() if sp.issparse(X1) else np.asarray(X1)
    x = np.vstack([m1, m2]).astype(np.float64)
    x = x - x.mean(axis=0)
    k = min(n_components, x.shape[0])
    pca = PCA(n_components=k, random_state=58951).fit(x.T)
    comp = pca.components_.T * pca.singular_values_[None, :]
    return comp[:len(m1)], comp[len(m1):]


def wot_eps_stages(eps: float, eps0: float = 1.0, scalings: int = 5) -> tuple:
    """The intermediate eps values of wot's duality-gap solver: eps0 * sf^-e for e = 0 .. scalings - 1, with
    sf = eps^(-1 / scalings). (Its last stage runs at eps0 * eps, which is eps for the default eps0 = 1.)"""
    sf = np.exp(-np.log(eps) / scalings)
    return tuple(float(eps0 / sf ** e) for e in range(scalings))


@dataclass
class TransportMap:
    result: SinkhornResult             # the last growth iteration's solve
    learned_growth: list               # per growth iteration, the row target used (wot's G, density w.r.t. 1/I)
    seconds: float                     # cost normalization (exact median, host) + all solves, including any compile
    iterations: int                    # Sinkhorn iterations, all growth iterations


def wot_transport_map(x0: np.ndarray, x1: np.ndarray, growth: np.ndarray | None = None, delta_days: float = 1.0,
                      eps: float = 0.05, lambda1: float = 1.0, lambda2: float = 50.0, growth_iters: int = 1,
                      tol: float = 1e-6, stage_tol: float = 1e-6, opts: SinkhornOptions | None = None) -> TransportMap:
    """One wot transport map between coordinates x0 (I cells at t0) and x1 (J cells at t1).

    growth: per-cell growth rates at t0 (wot's cell_growth_rate), raised to delta_days. opts: backend, kernel,
    cores, check_every, max_iters (eps, rho, normalize and the eps stages are set here).
    """
    I, J = len(x0), len(x1)
    G = np.ones(I) if growth is None else np.power(np.asarray(growth, dtype=np.float64), delta_days)
    t = time.perf_counter()
    norm = normalization(x0, x1, how="median")          # once per pair: the cost doesn't change across growth passes
    seconds = time.perf_counter() - t
    base = replace(opts or SinkhornOptions(), eps=eps, rho=(lambda1, lambda2), normalize=norm,
                   eps_stages=wot_eps_stages(eps), stage_tol=stage_tol, tol=tol, eps_start=None, fixed_iters=False)
    a, b = np.full(I, 1.0 / I), np.full(J, 1.0 / J)
    learned, iters, res = [], 0, None
    for i in range(growth_iters):
        if i > 0:
            G = I * res.row_mass                       # wot: row sums of the previous tmap (= I * P)
        learned.append(G)
        t = time.perf_counter()
        res = sinkhorn(x0, x1, a, b, base, a_target=G / I, b_target=np.full(J, G.mean() / J))
        seconds += time.perf_counter() - t
        iters += res.iterations
    return TransportMap(result=res, learned_growth=learned, seconds=seconds, iterations=iters)


def wot_tmap(x0, x1, tm: TransportMap) -> np.ndarray:
    """The dense transport map in wot's normalization (I * P), for comparison with wot's output. Small pairs only."""
    I, J = len(x0), len(x1)
    return I * transport_plan(x0, x1, tm.result, np.full(I, 1.0 / I), np.full(J, 1.0 / J))
