"""
Building off of: https://arxiv.org/pdf/2010.11929
"""

import torch

from torch import nn
from torch.nn.functional import scaled_dot_product_attention


class MHALayer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: bool = False, vdim: int | None = None, kdim: int | None = None
    ):
        """
        Implementation of a standard multi head attention layer.

        Args:
            embed_dim (int): Dimension of input embeddings.
            num_heads (int): Number of attention heads.
            qk_dim (int): Dimension of the query-key vectors.
            dropout (bool): Whether to include dropout, which randomly zeroes some neurons (regularisation).
            vdim (int | None): Dimension of the value embeddings.
            kdim (int | None): Dimension of the key embeddings.
        """
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("The embeding dimension must be divisible by the number of heads.")

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
        try:
            B, N, D = x.shape
        except ValueError as e:
            raise e("Input tensor must be of dimension 3: (batch_size, num_tokens, emb_dim).")

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
