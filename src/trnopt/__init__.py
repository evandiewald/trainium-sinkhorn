"""trnopt: entropic optimal transport (Sinkhorn) on AWS Trainium."""

from . import compile_cache

compile_cache.configure()            # before torch is imported: see compile_cache.py

from .backend import Backend  # noqa: E402
from .sinkhorn import SinkhornOptions, SinkhornResult, sinkhorn, transport_plan  # noqa: E402

__all__ = ["Backend", "SinkhornOptions", "SinkhornResult", "sinkhorn", "transport_plan"]
