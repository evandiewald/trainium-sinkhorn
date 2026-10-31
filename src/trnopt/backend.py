"""Where device differences live: one settings object, chosen once.

The solver never asks "am I on Trainium?". It reads these settings instead.

* dtype   -- Trainium has no float64. Its widest number type is float32 (about 7 significant digits instead
             of 16), so tight tolerances get harder (see the float32 floors in sinkhorn.py).
* compile -- On an accelerator, every separate tensor operation is a trip from the host to the chip.
             torch.compile fuses a whole half-step into one program. On Neuron this needs TorchNeuron's own
             "neuron" compiler backend and fixed shapes: the Neuron compiler can't handle the symbolic shapes
             torch.compile switches to after it sees a second problem size.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

_COMPILED: dict = {}          # compile_fn's cache: one compiled function per (function, compiler, shape mode)


@dataclass(frozen=True)
class Backend:
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    compile: bool = False
    compile_backend: str | None = None       # None = PyTorch's default compiler for the device

    @classmethod
    def preset(cls, name: str, **overrides) -> "Backend":
        """Named starting points. Any field can be overridden, e.g. Backend.preset("cpu", dtype=torch.float32)."""
        presets = {
            "cpu": cls(),
            # float32 alone: isolates the effect of Trainium's precision.
            "cpu-float32": cls(dtype=torch.float32),
            "cuda": cls(device="cuda"),
            # Trainium with TorchNeuron, PyTorch's native Neuron device (torch-neuronx 2.10+). Verified on trn1.
            # sinkhorn(kernel="auto") uses the fused NKI kernel here; kernel="torch" the compiled PyTorch half-step.
            "neuron": cls(device="neuron", dtype=torch.float32, compile=True, compile_backend="neuron"),
            # Trainium's restrictions (float32, compiled, fixed shapes) on a laptop CPU.
            "emulate-neuron": cls(device="cpu", dtype=torch.float32, compile=True),
            # Apple silicon GPU via PyTorch's Metal backend (no float64 there).
            "mps": cls(device="mps", dtype=torch.float32),
        }
        if name not in presets:
            raise ValueError(f"unknown backend preset {name!r}; choose from {sorted(presets)}")
        return replace(presets[name], **overrides)

    def compile_fn(self, f):
        """torch.compile `f` if this backend compiles, else return it unchanged.

        Cached per (function, compiler, shape mode) for the life of the process, so every solve reuses the same
        compiled function; pass per-problem data in as arguments, never by closure.
        """
        if not self.compile:
            return f
        # dynamic=False on Neuron: always specialize to concrete shapes (one compiled program per shape).
        dynamic = False if self.device.startswith("neuron") else None
        if dynamic is False:
            # Dynamo's default of 8 versions per function would silently fall back to uncompiled code in a run
            # over many problem sizes (e.g. the 38 single-cell pairs).
            import torch._dynamo as dynamo
            dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 256)
        key = (f, self.compile_backend or "inductor", dynamic)
        if key not in _COMPILED:
            _COMPILED[key] = torch.compile(f, backend=key[1], dynamic=dynamic)
        return _COMPILED[key]

    def tensor(self, a) -> torch.Tensor:
        # Convert on the CPU first: some devices (Apple's MPS) can't hold float64 even briefly.
        return torch.as_tensor(np.asarray(a), dtype=self.dtype).to(self.device)
