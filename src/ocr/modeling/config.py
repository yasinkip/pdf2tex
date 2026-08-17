from dataclasses import asdict, dataclass

import torch


class BaseConfig:

    def as_dict(self):
        return asdict(self)


@dataclass
class MHAConfig(BaseConfig):
    """Config for MultiheadAttention blocks."""
    embed_dim: int
    num_heads: int
    dropout: bool = False
    vdim: int | None = None
    kdim: int | None = None

    def as_dict(self):
        return asdict(self)


@dataclass
class MLPConfig(BaseConfig):
    """Config for FFN layers."""
    hidden_channels: list[int]
    activation_layer: None = None
    bias: bool = True
    dropout: float = 0


@dataclass
class NormConfig(BaseConfig):
    """Config for LayerNorm."""
    normalized_shape: int | list | torch.Size