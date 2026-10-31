"""Sinkhorn (trnopt.sinkhorn) against an independent float64 reference, and against POT when it's installed.

The reference below works on the full cost matrix C directly, with no |x|^2 + |y|^2 - 2 x.y expansion and no
shifted potentials, so it checks the attention-style rewrite as well as the iteration. Plans are compared, not
balanced potentials: those are only defined up to a constant (f + c, g - c give the same plan).
"""

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from trnopt import Backend
from trnopt.sinkhorn import SinkhornOptions, normalization, sinkhorn, transport_plan

# name -> (backend, allowed max plan error relative to max(P))
BACKENDS = {
    "cpu": (Backend.preset("cpu"), 1e-6),
    "cpu-float32": (Backend.preset("cpu", dtype=torch.float32), 2e-3),
    "emulate-neuron": (Backend.preset("emulate-neuron", compile=False), 2e-3),
}


def stop_tol(be):
    """Stopping tolerance: float32 bottoms out around 1e-6 marginal error, so ask for less."""
    return 1e-5 if be.dtype == torch.float32 else 1e-7


def _neuron_device_present() -> bool:
    try:
        import torch_neuronx  # noqa: F401
        return torch.neuron.device_count() > 0
    except Exception:
        return False


HAVE_NEURON = _neuron_device_present()
if HAVE_NEURON:
    BACKENDS["neuron"] = (Backend.preset("neuron"), 2e-3)      # kernel="auto": the fused NKI kernel


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def backend(request):
    return BACKENDS[request.param]


def clouds(n=137, m=91, d=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = rng.normal(size=(m, d)) * 0.8 + 0.5
    return x, y


def reference(x, y, eps, rho=None, a=None, b=None, iters=20_000, tol=1e-12):
    """Plain log-domain Sinkhorn on the full C, float64. Returns the plan, in the normalized units sinkhorn uses."""
    center, scale = normalization(x, y)
    x, y = (x - center) / np.sqrt(scale), (y - center) / np.sqrt(scale)
    n, m = len(x), len(y)
    la = np.log(a) if a is not None else np.full(n, -np.log(n))
    lb = np.log(b) if b is not None else np.full(m, -np.log(m))
    C = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    kappa = 1.0 if rho is None else rho / (rho + eps)
    f, g = np.zeros(n), np.zeros(m)
    for _ in range(iters):
        f = -kappa * eps * logsumexp(lb[None, :] + (g[None, :] - C) / eps, axis=1)
        g_new = -kappa * eps * logsumexp(la[:, None] + (f[:, None] - C) / eps, axis=0)
        done = np.abs(g_new - g).max() < tol
        g = g_new
        if done:
            break
    return np.exp(la[:, None] + lb[None, :] + (f[:, None] + g[None, :] - C) / eps), f, g


def plan_err(P, Q):
    return np.abs(P - Q).max() / Q.max()


def test_balanced_matches_reference(backend):
    be, tol = backend
    x, y = clouds()
    res = sinkhorn(x, y, opts=SinkhornOptions(eps=0.1, tol=stop_tol(be), max_iters=5000, backend=be))
    assert res.converged
    P_ref, _, _ = reference(x, y, eps=0.1)
    assert plan_err(transport_plan(x, y, res), P_ref) < tol
    # rows sum to a exactly after an f update; columns to b within the stopping tolerance
    P = transport_plan(x, y, res)
    np.testing.assert_allclose(P.sum(axis=1), 1 / len(x), rtol=tol * 10)
    np.testing.assert_allclose(P.sum(axis=0), 1 / len(y), rtol=tol * 10 + 1e-5)


def test_blocked_matches_unblocked(backend):
    """Blocks that don't divide n or m: exercises the padding."""
    be, tol = backend
    x, y = clouds()
    o = dict(eps=0.1, tol=stop_tol(be), max_iters=5000, backend=be)
    full = sinkhorn(x, y, opts=SinkhornOptions(**o))
    blk = sinkhorn(x, y, opts=SinkhornOptions(block=32, block_y=40, **o))
    assert blk.converged
    assert plan_err(transport_plan(x, y, blk), transport_plan(x, y, full)) < tol
    assert abs(blk.cost - full.cost) < tol * (1 + abs(full.cost))


def test_unbalanced_matches_reference(backend):
    be, tol = backend
    x, y = clouds()
    a = np.random.default_rng(1).uniform(0.5, 1.5, len(x)) / len(x)   # total mass != 1: allowed when unbalanced
    res = sinkhorn(x, y, a=a, opts=SinkhornOptions(eps=0.1, rho=1.0, tol=stop_tol(be), max_iters=5000, block=50, backend=be))
    assert res.converged
    P_ref, f_ref, g_ref = reference(x, y, eps=0.1, rho=1.0, a=a)
    assert plan_err(transport_plan(x, y, res, a=a), P_ref) < tol
    # unbalanced potentials are unique: compare them directly
    np.testing.assert_allclose(res.f, f_ref, atol=tol * 10)


def test_weighted_balanced(backend):
    be, tol = backend
    x, y = clouds()
    rng = np.random.default_rng(2)
    a = rng.uniform(0.2, 1.0, len(x)); a /= a.sum()
    b = rng.uniform(0.2, 1.0, len(y)); b /= b.sum()
    res = sinkhorn(x, y, a, b, SinkhornOptions(eps=0.1, tol=stop_tol(be), max_iters=5000, backend=be))
    P_ref, _, _ = reference(x, y, eps=0.1, a=a, b=b)
    assert plan_err(transport_plan(x, y, res, a, b), P_ref) < tol


def test_eps_scaling_same_answer():
    """Annealing eps from 2.0 down changes the path, not the answer, and needs fewer iterations at small eps."""
    x, y = clouds(n=300, m=250, d=10, seed=3)
    o = dict(eps=0.01, tol=1e-6, max_iters=20_000)
    plain = sinkhorn(x, y, opts=SinkhornOptions(**o))
    scaled = sinkhorn(x, y, opts=SinkhornOptions(eps_start=2.0, eps_decay=0.8, **o))
    assert plain.converged and scaled.converged
    assert scaled.iterations < plain.iterations
    assert plan_err(transport_plan(x, y, scaled), transport_plan(x, y, plain)) < 1e-4


def test_fixed_iters_runs_exactly():
    x, y = clouds()
    res = sinkhorn(x, y, opts=SinkhornOptions(max_iters=7, fixed_iters=True))
    assert res.iterations == 7 and not res.converged


def test_matches_pot():
    """Cross-check against POT's log-domain solver on the raw (unnormalized) cost."""
    ot = pytest.importorskip("ot")
    x, y = clouds(n=120, m=80, d=4, seed=5)
    eps = 0.5
    res = sinkhorn(x, y, opts=SinkhornOptions(eps=eps, normalize=False, tol=1e-9, max_iters=10_000))
    M = ot.dist(x, y)                      # squared Euclidean
    a, b = np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y))
    P_pot = ot.bregman.sinkhorn_log(a, b, M, reg=eps, numItermax=10_000, stopThr=1e-12)
    assert plan_err(transport_plan(x, y, res), P_pot) < 1e-6


def test_rejects_unequal_mass_when_balanced():
    x, y = clouds()
    with pytest.raises(ValueError):
        sinkhorn(x, y, a=np.full(len(x), 2.0 / len(x)))


@pytest.mark.skipif(not HAVE_NEURON, reason="no NeuronCore on this machine")
@pytest.mark.parametrize("rho", [None, 1.0])
def test_nki_kernel_matches_torch_on_chip(rho):
    """The fused kernel against the compiled PyTorch half-step on the chip, at a size with padding on both sides
    and several kernel calls per half-step (block 512)."""
    x, y = clouds(n=1300, m=1700, d=50, seed=4)
    o = dict(eps=0.05, rho=rho, tol=1e-5, max_iters=3000, backend=Backend.preset("neuron"))
    ref = sinkhorn(x, y, opts=SinkhornOptions(kernel="torch", **o))
    fused = sinkhorn(x, y, opts=SinkhornOptions(kernel="nki", block=512, **o))
    assert fused.converged and abs(fused.iterations - ref.iterations) <= 10
    assert plan_err(transport_plan(x, y, fused), transport_plan(x, y, ref)) < 2e-3


@pytest.mark.skipif(not HAVE_NEURON, reason="no NeuronCore on this machine")
def test_two_cores_match_one_on_chip():
    """Rows split across both NeuronCores (halves exchanged through the host) give the same answer as one core."""
    x, y = clouds(n=2100, m=1500, d=30, seed=6)
    o = dict(eps=0.05, tol=1e-5, max_iters=3000, kernel="nki", backend=Backend.preset("neuron"))
    one = sinkhorn(x, y, opts=SinkhornOptions(**o))
    two = sinkhorn(x, y, opts=SinkhornOptions(cores=2, **o))
    assert two.converged and two.iterations == one.iterations
    np.testing.assert_allclose(two.f, one.f, atol=1e-5)
    np.testing.assert_allclose(two.g, one.g, atol=1e-5)


def wot_scaling_reference(x, y, eps, lam1, lam2, G, iters=50_000, tol=1e-13):
    """wot's own update (scaling form, as in wot.ot.optimal_transport_duality_gap without its stabilization):
    uniform reference, row target G (density), column target mean(G); cost / median. float64. Returns I * P (wot's
    tmap normalization)."""
    C = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    C = C / np.median(C)
    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J
    p, q = G, np.ones(J) * G.mean()
    K = np.exp(-C / eps)
    a1, a2 = lam1 / (lam1 + eps), lam2 / (lam2 + eps)
    a, b = np.ones(I), np.ones(J)
    for _ in range(iters):
        a = (p / K.dot(b * dy)) ** a1
        b_new = (q / K.T.dot(a * dx)) ** a2
        done = np.abs(b_new / b - 1).max() < tol
        b = b_new
        if done:
            break
    return (K.T * a).T * b / J


@pytest.mark.parametrize("staged", [False, True])
def test_wot_formulation_matches_wot_scaling(backend, staged):
    """Targets != reference, a rho per side, median normalization (and wot's eps stages): the same plan as wot's
    scaling iteration."""
    from trnopt.singlecell import wot_eps_stages
    be, tol = backend
    x, y = clouds(n=90, m=70, d=4, seed=7)
    G = np.random.default_rng(8).uniform(0.5, 2.0, len(x))
    eps, lam1, lam2 = 0.2, 1.0, 50.0                # eps big enough for the unstabilized reference
    I, J = len(x), len(y)
    o = SinkhornOptions(eps=eps, rho=(lam1, lam2), normalize="median", tol=stop_tol(be) / 10, max_iters=20_000,
                        check_every=5, backend=be, eps_stages=wot_eps_stages(eps) if staged else None)
    res = sinkhorn(x, y, np.full(I, 1 / I), np.full(J, 1 / J), o, a_target=G / I, b_target=np.full(J, G.mean() / J))
    assert res.converged
    T = I * transport_plan(x, y, res, np.full(I, 1 / I), np.full(J, 1 / J))
    R = wot_scaling_reference(x, y, eps, lam1, lam2, G)
    assert plan_err(T, R) < tol * 5
    np.testing.assert_allclose(I * res.row_mass, R.sum(1), rtol=tol * 5)      # learned growth
    np.testing.assert_allclose(I * res.col_mass, R.sum(0), rtol=tol * 5)
