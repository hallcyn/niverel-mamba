"""``niverel_mamba.mlx`` -- the MLX-facing public API.

    from niverel_mamba.mlx import Mamba2

    model = Mamba2(config)
    model.load_canonical_weights(weights)
    y = model(x, seq_idx=seq_idx)

Deliberately not a ``torch.nn.Module`` lookalike.
"""

from __future__ import annotations

from .config import Mamba2Config
from .mlx_ops.mamba2 import Mamba2, from_mlx_weights, to_mlx_weights
from .mlx_ops.state import Mamba2State

__all__ = ["Mamba2", "Mamba2Config", "Mamba2State", "from_mlx_weights", "to_mlx_weights"]
