"""
Building off of: https://arxiv.org/pdf/2010.11929.

The original paper uses a classification MLP head to classify the image embedded patches into a class using an encoder-only architecture.

Here we are going to use an encoder backbone and a decoder text generator.
"""

from dataclasses import asdict

import torch

from torch import nn
from torchvision.ops import MLP
from torch.nn.functional import scaled_dot_product_attention

from config import MHAConfig, MLPConfig, NormConfig


class ViTransformer(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int):
        super().__init__()
        pass

    def forward(x: torch.Tensor):
        pass


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        attn_cfg: MHAConfig,
        mlp_cfg: MLPConfig,
        base_dims: tuple[int, int, int],
        p_dims: tuple[int, int], 
        *,
        norm_cfg: NormConfig | None = None,
    ):
        """
        Encoder block. If attention layers are given, 

        Args:
            num_blocks (int): Number of attention blocks.

            attn_cfg (MHAConfig): Self attention head config.

            mlp (MLPConfig): Feed-forward neural network config.
           
            base_dims (tuple[int, int, int]): The base image dimensions (H, W, C) to interpolate into.

            p_dims (tuple[int, int]): Dimensions of each patch. i.e. 14x14 for a 196x196 image.

            norm_cfg (NormConfig | None): Norm layer config.
        """
        super().__init__()
        self.base_dims = base_dims
        self.p_dims = p_dims
        self.embed_dim = attn_cfg.embed_dim

        H, W, C = base_dims
        pH, pW = p_dims
        # ensure patch dimensions fit within the base dimensions given
        assert H // pH == W // pW 

        N = H * W // (p_dims[0] * p_dims[1])
        self.flattened_dims = (N, p_dims[0] * p_dims[1] * C)

        layers = []
        # Project to embedding dimension first
        layers.append(nn.Linear(self.flattened_dims[1], self.embed_dim)) 
        for _ in range(num_blocks):
            layers.append(**norm_cfg.as_dict())
            layers.append(**attn_cfg.as_dict())
            layers.append(self.embed_dim, **mlp_cfg.as_dict())
        self.stack = nn.Sequential(layers)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass is initially going to be implemented with absolute positional embeddings and linear interpolation. 

        TODO: We currently "perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image" but we can replace this w/ something like RoPE.

        Args: 
            x (torch.Tensor): The input image of size (B, H, W, C).
        """
        assert len(x.shape) == 4
        B = x.shape[0]
        flattened = torch.reshape(x, (B, *self.flattened_dims))
        return self.stack(flattened)


class MHALayer(nn.MultiheadAttention):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: bool = False, vdim: int | None = None, kdim: int | None = None
    ):
        """
        Implementation of a standard multi head attention layer.

        Args:
            embed_dim (int): Dimension of input embeddings.

            num_heads (int | None): Number of attention heads.

            dropout (bool): Whether to include dropout.

            vdim (int | None): Dimension of the value embeddings.

            kdim (int | None): Dimension of the key embeddings.
        """
        super().__init__()
        assert not embed_dim % num_heads, "The embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        if kdim is None:
            self.kdim = embed_dim

        if vdim is None:
            self.vdim = embed_dim

        self.head_dim = embed_dim // num_heads

        self.Wq = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.Wk = nn.Linear(self.kdim, self.embed_dim, bias=False)
        self.Wv = nn.Linear(self.vdim, self.embed_dim, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor, causal: bool = False):
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
        attn = scaled_dot_product_attention(query, key, value, is_causal=causal)
        attn = torch.reshape(attn, (B, N, D))
        attn = self.out_proj(attn)

        # Apply residual connection and return
        return x + attn