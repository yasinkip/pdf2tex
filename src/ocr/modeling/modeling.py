"""
Building off of: https://arxiv.org/pdf/2010.11929.

The original paper uses a classification MLP head to classify the image embedded patches into a class using an encoder-only architecture.

Here we are going to use an encoder backbone and a decoder text generator.
"""

import math

from typing import Literal

import torch

from torch import nn
from torchvision.ops import MLP
from torch.nn.functional import scaled_dot_product_attention

from .config import OCRConfig, TransformerStackConfig


class OCR(nn.Module):
    def __init__(self, cfg: OCRConfig):
        super().__init__()
        encoder = VisionEncoder(cfg.encoder_cfg)

        pass

    def forward(x: torch.Tensor):
        pass


class TextDecoder(nn.Module):
    def __init__(
        self, 
        stack_cfg: TransformerStackConfig,
    ):
        """
        Decoder Stack. Currently implemented exactly as in the original "Attention is All You Need" paper, with more explicit parameterization compared to `nn.TransformerDecoder`, allowing for more customisability.

        Args:
            num_blocks (int): Number of transformer blocks.

            block_cfg (TransformerBlockConfig): Block config.
        """
        assert stack_cfg.block.attn.causal, "Must use causal attention for decoder stack."

        self.stack = nn.Sequential(*build_blocks(stack_cfg))
        pass

    def forward(x: torch.Tensor):
        pass



class VisionEncoder(nn.Module):
    def __init__(
        self,
        stack_cfg: TransformerStackConfig,
        base_dims: tuple[int, int, int],
        p_dims: tuple[int, int],
    ):
        """
        Encoder stack. Currently implemented exactly as in the original Vision Transformers paper. 

        Args:
            num_blocks (int): Number of transformer blocks.

            block_cfg (TransformerBlock): Config for neural net layers.

            base_dims (tuple[int, int, int]): The base image dimensions (H, W, C) to interpolate into.

            p_dims (tuple[int, int]): Dimensions of each patch. i.e. 14x14 for a 196x196 image.
        """
        super().__init__()
        self.base_dims = base_dims
        self.p_dims = p_dims

        block_cfg = stack_cfg.block
        self.embed_dim = block_cfg.attn.embed_dim

        assert not block_cfg.attn.causal, "Cannot use causal attention for encoder stack."

        H, W, C = base_dims
        pH, pW = p_dims
        # ensure patch dimensions fit within the base dimensions given
        assert H // pH == W // pW, f"Patches don't fit exactly in base dimensions: {H} // {pH} =/= {W} // {pW}"
        assert H % pH == 0 and W % pW == 0, (
            f"Patch dimensions aren't divisible by base dimensions: one of ({H}, {pH}) and ({W}, {pW}) do not divide."
        )

        N = H * W // (p_dims[0] * p_dims[1])
        self.flattened_dims = (N, p_dims[0] * p_dims[1] * C)

        self.stack = nn.Sequential(*build_blocks(stack_cfg))

        # Initial projector to embedding dimension
        self.embedder = nn.Linear(self.flattened_dims[1], self.embed_dim)

        # Initialisation I just set arbitrarily to be a random float in [-N-1, N+1]
        self.cls = nn.Parameter((N + 1) * torch.rand(size=(self.embed_dim,), requires_grad=True)) 
        self.positional = nn.Parameter((N + 1) * torch.rand(size=(N + 1, self.embed_dim), requires_grad=True))

    def forward(self, x: torch.Tensor):
        """
        Forward pass is initially going to be implemented with absolute positional embeddings and linear interpolation. We assume at this stage that `x` has been resized to `base_dims` dimensions.

        TODO: We currently "perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image" but we can replace this w/ something like RoPE.

        Args:
            x (torch.Tensor): The input image of size (B, H, W, C).
        """
        assert len(x.shape) == 4, f"Input tensor needs to be of dimension 4, instead its of dimension {len(x.shape)}."
        assert x.shape[1:] == self.base_dims, (
            f"Please resize x (which has dimensions {x.shape[1:]}) to {self.base_dims})"
        )

        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, *self.flattened_dims))

        # Run through embedder
        z = self.embedder(x)

        # add [CLS] token and positional embedding
        cls = self.cls.data
        cls = cls.expand((batch_size, 1, self.embed_dim))
        z = torch.cat((cls, z), dim=1)
        z += self.positional
        return self.stack(z)


class MHALayer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: bool = False, vdim: int | None = None, kdim: int | None = None, causal: bool = False
    ):
        """
        Implementation of a standard multi head attention layer using `scaled_dot_product_attention`.

        Args:
            embed_dim (int): Dimension of input embeddings.

            num_heads (int | None): Number of attention heads.

            dropout (bool): Whether to include dropout.

            vdim (int | None): Dimension of the value embeddings.

            kdim (int | None): Dimension of the key embeddings.

            causal (bool): Whether or not to use causal attention.
        """
        super().__init__()
        assert not embed_dim % num_heads, "The embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.head_dim = embed_dim // num_heads

        self.Wq = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.Wk = nn.Linear(self.kdim, self.embed_dim, bias=False)
        self.Wv = nn.Linear(self.vdim, self.embed_dim, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor fo size (batch_size, num_tokens, emb_dim) == (B, N, D).
        """
        assert len(x.shape) == 3, "Input tensor must be of dimension 3: (batch_size, num_tokens, emb_dim)."
        B, N, D = x.shape

        # All of size (B, N, D)
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)

        # sliced per-head (D -> (num_heads, D_head))
        query = torch.reshape(query, (B, N, self.num_heads, self.head_dim))
        key = torch.reshape(query, (B, N, self.num_heads, self.head_dim))
        value = torch.reshape(query, (B, N, self.num_heads, self.head_dim))

        # Compute attention, concatenate across heads, then output projection
        attn = scaled_dot_product_attention(query, key, value, is_causal=self.causal)
        attn = torch.reshape(attn, (B, N, D))
        attn = self.out_proj(attn)

        # Apply residual connection and return
        return x + attn


def build_blocks(block_cfg: TransformerStackConfig) -> list[nn.Module]:
    layers = []
    for _ in range(block_cfg.num_blocks):
        layers.append(nn.LayerNorm(**block_cfg.norm.as_dict()))
        layers.append(MHALayer(**block_cfg.attn.as_dict()))
        layers.append(MLP(block_cfg.embed_dim, **block_cfg.mlp.as_dict()))
    return layers
