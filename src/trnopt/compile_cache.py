"""One persistent home for everything trnopt compiles, so a new process doesn't compile it again.

* torch.compile with the "neuron" backend (TorchNeuron): each graph is lowered to HLO and compiled by neuronx-cc
  into a NEFF, kept in TORCH_NEURONX_NEFF_CACHE_DIR (eager ops' HLO in TORCH_NEURONX_HLO_CACHE_DIR). Both default
  to /tmp, which is wiped whenever the instance stops.
* torch.compile on a CPU or GPU (emulate-neuron, cuda): Inductor's cache, TORCHINDUCTOR_CACHE_DIR.
* NKI kernels keep their own trace cache in /var/tmp, which survives reboots; left alone.

configure() points them at one directory, TRNOPT_CACHE_DIR (default ~/.cache/trnopt), so they survive reboots and
can be copied to the next instance. It never overrides a variable you set yourself; TRNOPT_CACHE_DIR=off leaves
PyTorch's and Neuron's defaults alone. It runs when trnopt is imported: torch-neuronx reads its cache locations as
it loads, so import trnopt before torch (or set the variables in your shell).

Caveat (two NeuronCores): the NEFF cache is keyed on the HLO, not on the core a graph was compiled for, so a graph
cached from neuron:1 can be reused on neuron:0 and fail ("Inconsistent device IDs"). That's why the multi-core path
in sinkhorn.py runs everything through NKI kernels, which work on either core.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ENV = "TRNOPT_CACHE_DIR"
_OFF = ("", "0", "off", "none", "false")


def cache_dir() -> Path | None:
    """The cache's root directory, or None if TRNOPT_CACHE_DIR turns caching off."""
    v = os.environ.get(ENV)
    if v is not None:
        return None if v.strip().lower() in _OFF else Path(v).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "trnopt"


def configure() -> Path | None:
    """Point the Inductor and Neuron compile caches at cache_dir(). Returns the root, or None if off."""
    root = cache_dir()
    if root is None:
        return None
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_DIR", str(root / "neff"))
    os.environ.setdefault("TORCH_NEURONX_HLO_CACHE_DIR", str(root / "hlo"))
    # neuronx-cc also takes the location as --cache_dir in NEURON_CC_FLAGS, which wins; leave that alone.
    if "--cache_dir" not in os.environ.get("NEURON_CC_FLAGS", ""):
        os.environ.setdefault("NEURON_COMPILE_CACHE_URL", str(root / "neuron"))
    return root


def clear() -> None:
    """Delete trnopt's cache directory (the next run compiles everything again)."""
    root = cache_dir()
    if root is not None and root.exists():
        shutil.rmtree(root)
