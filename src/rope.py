import torch
import torch.nn as nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta: theta value for RoPE
            d_k: dimension of query and key vectors
            max_seq_len: maxinum sequence length that will be inputted
        """
        super().__init__()

        # TODO: handle odd d_k. The last dimension is not rotated and left untouched.
        assert d_k % 2 == 0

        # TODO: precompute cosines and sines for all positions in [max_seq_len]

        self.d_k = d_k
        num_pairs: int = d_k // 2

        # Rotation angle:
        # \theta_{i, k} = \frac{i}{\theta{2k/d}}
        # where i is the token position, k is the pair position and d is d_k

        # Compute base theta. The shape is (num_pairs,)
        # Before rotation, scale base_thetas by position

        # [0, 0, 2, 2, ...] / d_k, Shape=(num_pairs,)
        exponent = torch.arange(start=0, end=num_pairs) * 2 / d_k
        base_thetas = 1 / torch.pow(theta, exponent=exponent)
        self.register_buffer("base_thetas", base_thetas)

        # torch.register_buffer specifies that the weights is part of the model's
        # state, but NOT trainable parameters. Implications:
        # 1. When moving the module to device (e.g. .to('cuda')), the weights
        # will be moved too.
        # 2. The weights will be included in `state_dict()`

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        Args
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        """
        assert x.shape[-1] == self.d_k

        num_pairs = self.d_k // 2

        # A naive approach is to construct the full block-diagonal matrix consisting
        # of 2x2 rotation matrices. However, the matrix is very sparse and the
        # operation is inefficent.
        #
        # Instead, directly apply the 2x2 rotation matrices to pairs of
        # embedding elements. There are two ways to apply rotation:
        #
        # 1. Using matrix multplication.
        #    - Reshape input (..., seq_len, d) into (..., seq_len, d/2, 2)
        #    - Construct rotation matrices (..., seq_len, 2, 2)
        #    - Apply matmul to get (..., seq_len, d/2, 2)
        #    - Reshape batck to (..., seq_len, d)
        #
        # 2. Using element-wise operation
        #
        #    For a pair x_{2k}, x_{2k+1},
        #      x’_{2k}   = x_{2k} * cosθ - x_{2k+1} * sinθ
        #      x’_{2k+1} = x_{2k} * sinθ + x_{2k+1} * cosθ
        #
        #    where θ(i, k) = i * base_theta
        #
        #  Here we implement the 2nd approach.

        # Separate into x_even and x_odd
        even_end = 2 * num_pairs + 1
        x_even = x[..., 0:even_end:2]  # 0, 2, ..., 2*num_pairs

        odd_end = even_end + 1
        x_odd = x[..., 1:odd_end:2]  # 1, 3, ..., 2*num_pairs + 1

        # Scale rotation angle by token position
        self.base_thetas: torch.Tensor
        betas: torch.Tensor = einsum(
            self.base_thetas,
            token_positions,
            "num_pairs, ... seq_len -> ... seq_len num_pairs",
        )

        # Apply rotation using element-wise operation
        cosines = torch.cos(betas)
        sines = torch.sin(betas)
        x_even_rotated = x_even * cosines - x_odd * sines
        x_odd_rotated = x_even * sines + x_odd * cosines

        # Interleave x_even_rotated and x_odd_rotated into final result
        x_even_and_odd_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        # Einops pattern doesn't support literal, so use `d2=2`
        x_rotated = rearrange(
            x_even_and_odd_rotated,
            "... seq_len num_pairs d2 -> ... seq_len (num_pairs d2)",
            d2=2,
        )

        return x_rotated
