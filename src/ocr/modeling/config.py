from typing import Callable

import torch
import torch.nn.functional as F

from msgspec import Struct
from msgspec.structs import asdict


class BaseConfig(Struct):

    def as_dict(self):
        return asdict(self)


class MHAConfig(BaseConfig):
    """Config for MultiheadAttention blocks."""
    embed_dim: int
    num_heads: int
    dropout: bool = False
    vdim: int | None = None
    kdim: int | None = None


class MLPConfig(BaseConfig):
    """Config for FFN layers."""
    hidden_channels: list[int]
    activation_layer: Callable[[torch.Tensor, str], torch.Tensor] = F.gelu
    bias: bool = True
    dropout: float = 0


class NormConfig(BaseConfig):
    """Config for LayerNorm."""
    normalized_shape: int | list | torch.Size