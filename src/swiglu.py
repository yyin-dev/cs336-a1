import torch
import torch.nn as nn
import math
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        std = math.sqrt((d_model + d_ff) / 2)

        w1_init = nn.init.trunc_normal_(
            torch.zeros(d_ff, d_model), mean=0, std=std, a=-3 * std, b=3 * std
        )
        self.W1 = nn.Parameter(w1_init)

        w2_init = nn.init.trunc_normal_(
            torch.zeros(d_model, d_ff), mean=0, std=std, a=-3 * std, b=3 * std
        )
        self.W2 = nn.Parameter(w2_init)

        w3_init = nn.init.trunc_normal_(
            torch.zeros(d_ff, d_model), mean=0, std=std, a=-3 * std, b=3 * std
        )
        self.W3 = nn.Parameter(w3_init)

    def forward(self, x: torch.Tensor):
        W1x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = W1x * torch.sigmoid(W1x)
        W3x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        gated = einsum(silu, W3x, "... d_ff, ... d_ff -> ... d_ff")
        result = einsum(self.W2, gated, "d_model d_ff, ... d_ff -> ... d_model")
        return result
