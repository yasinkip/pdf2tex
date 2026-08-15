import torch

from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, qk_dim):
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        pass

    def forward(x: torch.Tensor):
        # x is of size (num_batches, batch_size, emb_dim)
        try:
            num_batches, batch_size, emb_dim = x.shape
        except ValueError as e:
            raise e("Input tensor must be of dimension 3.")
