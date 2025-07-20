import torch
from torch import nn
from einops import einsum, reduce, rearrange


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.W = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x: (batch_size, sequence_length, d_model)
        """

        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_square_mean = reduce(x * x, "b seq_len d_model-> b seq_len", "mean")
        rms = (x_square_mean + self.eps).sqrt()
        rms = rearrange(rms, "batch seq_len -> batch seq_len 1")

        scaled_x = x / rms

        # Broadcasted element-wise multiplication
        result = scaled_x * self.W

        return result.to(in_dtype)
