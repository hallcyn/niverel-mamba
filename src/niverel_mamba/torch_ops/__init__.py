"""Pure-PyTorch Mamba2 operations.

Importing this package imports torch. Callers that must stay framework-free
should go through :mod:`niverel_mamba.registry` instead.
"""

from __future__ import annotations

from .causal_conv import causal_conv1d, validate_seq_idx
from .gated_rmsnorm import gated_rmsnorm
from .mamba2 import SSD_IMPLEMENTATIONS, Mamba2
from .ssd_chunked import ssd_chunked
from .ssd_sequential import ssd_sequential
from .state import Mamba2State

__all__ = [
    "SSD_IMPLEMENTATIONS",
    "Mamba2",
    "Mamba2State",
    "causal_conv1d",
    "gated_rmsnorm",
    "ssd_chunked",
    "ssd_sequential",
    "validate_seq_idx",
]
