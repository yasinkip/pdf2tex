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
    causal: bool = False


class MLPConfig(BaseConfig):
    """Config for FFN layers."""
    hidden_channels: list[int]
    activation_layer: Callable[[torch.Tensor, str], torch.Tensor] = F.gelu
    bias: bool = True
    dropout: float = 0


class NormConfig(BaseConfig):
    """Config for LayerNorm."""
    normalized_shape: int | list | torch.Size


class TransformerBlockConfig(BaseConfig):
    attn: MHAConfig
    mlp: MLPConfig
    norm: NormConfig

    @property
    def embed_dim(self):
        return self.attn.embed_dim


class TransformerStackConfig(BaseConfig):
    num_blocks: int
    block: TransformerBlockConfig


class OCRConfig(BaseConfig):
    encoder_cfg: TransformerStackConfig
    decoder_cfg: TransformerStackConfig
    num_layers: int

    @property
    def encoder_emb_dim(self):
        return self.encoder_cfg.attn.embed_dim

    @property
    def decoder_emb_dim(self):
        return self.decoder_cfg.attn.embed_dim