# trnopt: Sinkhorn optimal transport on AWS Trainium

Entropic optimal transport (log-domain Sinkhorn) on AWS Trainium, with a fused NKI kernel, applied to single-cell
trajectory inference (Waddington-OT). This is the companion code for the article *[title / link TBD]*.

What's here:

- A plain-PyTorch Sinkhorn that runs on CPU, GPU or Trainium (the baseline).
- A fused NKI kernel that computes each half-step without materializing the cost matrix (the FlashAttention /
  FlashSinkhorn idea, re-planned for the NeuronCore's engines), with an optional split across both NeuronCores.
- A reproduction of [wot](https://github.com/broadinstitute/wot)'s transport maps on the Schiebinger et al. (Cell
  2019) reprogramming data, and benchmarks against wot (CPU), FlashSinkhorn and OTT-JAX (T4, A100).

## Results

All numbers measured on trn1.2xlarge (TorchNeuron, torch-neuronx 2.14), g4dn (T4), and Lambda 1x A100 40GB.
Raw results are in [`results/`](results/).

**Single-cell, full pipeline** (38 consecutive day pairs, 175k serum cells, wot's settings, run to convergence):

| | precision | time | |
|---|---|---|---|
| wot 1.0.8 on the trn1 host CPU (8 vCPU) | fp64 | 2,156 s | reference |
| trnopt, 1 NeuronCore | fp32 | **142 s** | **15.2x**; maps match wot to ~1.5e-5 |

**Single-cell, per iteration** (ms; d = 30, eps = 0.05, unbalanced, largest pair 7,355 x 8,962 cells):

| | fp32 | TF32 |
|---|---|---|
| trnopt, trn1 (1 NeuronCore) | 2.61 | n/a (no TF32 on trn1) |
| FlashSinkhorn, T4 | 3.90 | n/a |
| FlashSinkhorn, A100 | 1.59 | 0.43 (plan off by 2-3.5%) |
| OTT-JAX, A100 | 1.56 | 1.14 |

**Synthetic** (FlashSinkhorn paper's Table 2 setup: 10k x 10k, d = 64, eps = 0.1, ms per iteration):

| | ms / it |
|---|---|
| trn1, naive PyTorch (1 core) | 10.2 |
| trn1, fused NKI kernel (1 core) | 3.67 |
| trn1, fused NKI kernel (2 cores) | **2.88** |
| A100 FlashSinkhorn, fp32 | 3.21 |
| A100 FlashSinkhorn, TF32 (paper: 0.82) | 0.71 |
| T4 FlashSinkhorn, fp32 | 10.12 |

At the prices paid (trn1.2xlarge $1.34/hr, A100 $1.99/hr), trn1 and the A100 are roughly even per dollar in fp32
on the cell data, and trn1 is ~1.65x better per dollar on the d = 64 synthetic problem. The A100's TF32 mode is
~4x better per dollar, at the cost of a 2-3.5% different transport plan.

## Layout

```
src/trnopt/
  sinkhorn.py        log-domain Sinkhorn: balanced/unbalanced, eps-scaling, blocking, 1 or 2 NeuronCores
  nki_sinkhorn.py    the fused NKI kernels (row log-sum-exp; whole half-step)
  singlecell.py      wot's formulation: local PCA, median-normalized cost, growth iterations
  backend.py         device presets (cpu, cpu-float32, cuda, neuron, emulate-neuron, mps)
  compile_cache.py   persistent compile cache (TRNOPT_CACHE_DIR, default ~/.cache/trnopt)
tests/test_sinkhorn.py   against an independent float64 reference, POT, and wot's own update
examples/
  sinkhorn_benchmark.py      synthetic benchmark (--flashsinkhorn for the paper's setup; --kernel, --cores)
  nki_sinkhorn_check.py      kernel vs NumPy float64 (--sim for NKI's CPU simulator)
  profile_nki_sinkhorn.py    profile one kernel call with neuron-explorer
  singlecell_wot.py          transport maps for every day pair (--wot to run wot alongside and compare)
  singlecell_figures.py      ancestor figure (results/figures/ancestors_ips.png)
  sinkhorn_libs_compare.py   the same cell problems on trnopt / FlashSinkhorn / OTT-JAX
  flashsinkhorn_synthetic.py FlashSinkhorn on a GPU, synthetic setup
results/
```

## How it works (short version)

Each Sinkhorn half-step is a log-sum-exp over one row of scores: `f_i = -eps * LSE_j(log b_j + (g_j - C_ij)/eps)`.
Expanding `C_ij = |x_i|^2 - 2 x_i.y_j + |y_j|^2` makes the scores a matrix multiply plus a per-column term, the
same shape as attention. The kernel streams 128 x 512 tiles through the NeuronCore's three engines:

1. **Tensor engine**: scores for 128 rows x 512 columns in one `nc_matmul` (the per-column term rides in the
   matmul as an extra contraction row: `[x, 1] . [2y/eps, g/eps + log b]`).
2. **Vector engine**: the tile's row max.
3. **Scalar engine**: `exp(score - max)` and the row sum in one `activation` instruction.

Only each tile's (max, sum) is kept, in four rotating buffers so the engines overlap; they're merged into one
log-sum-exp per row at the end. Scores never leave the chip (5 MB of HBM traffic per 10k x 10k pass vs 0.94 GB for
the naive version). See the docstrings in `sinkhorn.py` and `nki_sinkhorn.py` for details.

## Running it

**Locally (CPU)**, to run the tests or try the solver:

```bash
uv sync
uv run pytest                      # CPU, float32 and emulate-neuron backends; NeuronCore tests skip
```

**On Trainium** (trn1.2xlarge, Neuron DLAMI with TorchNeuron):

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
pip install -e .
pytest                                                         # adds the on-chip kernel tests
python examples/sinkhorn_benchmark.py --flashsinkhorn --kernel torch --no-converge   # naive baseline
python examples/sinkhorn_benchmark.py --flashsinkhorn --kernel nki --cores 2         # fused kernel, 2 cores
```

TorchNeuron officially targets trn2/trn3; everything here was run on trn1.

**Single-cell**:

```bash
pip install -e ".[singlecell]" && pip install --no-deps wot
gdown 1E494DhIx5RLy0qv_6eWa9426Bfmq28po -O wotdata.zip && unzip wotdata.zip -d ~/wotdata   # wot tutorial data
python examples/singlecell_wot.py --data ~/wotdata/data --save results/sc_potentials      # all 38 pairs
python examples/singlecell_wot.py --data ~/wotdata/data --wot                             # also run wot (slow)
python examples/singlecell_figures.py --data ~/wotdata/data --potentials results/sc_potentials
```

wot 1.0.8 calls the removed `anndata.read`; `singlecell_wot.py` patches it.

**GPU comparisons** (FlashSinkhorn from github.com/ot-triton-lab/flash-sinkhorn, OTT-JAX):

```bash
python examples/sinkhorn_libs_compare.py --lib flash --pots results/sc_potentials [--tf32]
python examples/sinkhorn_libs_compare.py --lib ott   --pots results/sc_potentials [--tf32]
python examples/flashsinkhorn_synthetic.py [--tf32]
```

FlashSinkhorn defaults to TF32 on Ampere and newer; the fp32 numbers above use `allow_tf32=False`. JAX also defaults
to TF32 matmuls there; `--lib ott` sets `jax_default_matmul_precision="highest"` unless `--tf32`.

## Notes and caveats

- **Precision.** trn1 has no TF32, and bf16 gave ~6e-2 error in the log-sum-exp, so everything on trn1 is fp32.
  Results agree with wot's float64 maps to ~1.5e-5 and with FlashSinkhorn fp32 to ~1e-5.
- **Small d.** The Tensor engine's cost per tile doesn't depend on the contraction length (up to 128), so at the
  cell data's d = 30 it's ~1/4 utilized. That's why the A100 is faster on the cell data but not at d = 64.
- **Two cores** split the rows and exchange potentials through the host (~0.5 ms per iteration): 1.95x at 100k,
  1.27x at 10k, slower than one core at single-cell pair sizes.
- **Compile time.** Every new problem shape compiles (~10 s per pair the first time); the cache makes reruns fast.
- `results/sc_potentials/` (98 MB) and raw profiler traces are not in git; regenerate them with the commands above.
