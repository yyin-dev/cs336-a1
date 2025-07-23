import torch
import torch.nn as nn
from rms_norm import RMSNorm
from attention import MultiHeadSelfAttention
from swiglu import SwiGLU


class Transformer_block(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()

        self.pre_mhsa_rmsnorm = RMSNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)

        self.pre_ffn_rmsnorm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        # RMS Norm + Multi-Head Self-Attention
        attention = self.mhsa(self.pre_mhsa_rmsnorm(x))
        x = x + attention

        # RMS Norm + FFN
        result = x + self.ffn(self.pre_ffn_rmsnorm(x))
        return result
