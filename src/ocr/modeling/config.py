from typing import Any
from dataclasses import dataclass


@dataclass
class MHAConfig:
    """Config for MultiheadAttention blocks."""
    embed_dim: int
    num_heads: int
    dropout: bool = False
    vdim: int | None = None
    kdim: int | None = None


@dataclass
class MLPConfig:
    """Config for FFN layers."""
    hidden_channels: list[int]
    activation_layer: None = None
    bias: bool = True
    dropout: float = 0


@dataclass
class NormConfig:
    """Config for LayerNorm."""