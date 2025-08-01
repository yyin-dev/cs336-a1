#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import math
from einops import einsum

# Temporarily create the buggy SwiGLU for testing
class BuggySwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        std = math.sqrt((d_model + d_ff) / 2)  # BUGGY: too large

        w1_init = torch.nn.init.trunc_normal_(
            torch.zeros(d_ff, d_model), mean=0, std=std, a=-3 * std, b=3 * std
        )
        self.W1 = torch.nn.Parameter(w1_init)

        w2_init = torch.nn.init.trunc_normal_(
            torch.zeros(d_model, d_ff), mean=0, std=std, a=-3 * std, b=3 * std
        )
        self.W2 = torch.nn.Parameter(w2_init)

        w3_init = torch.nn.init.trunc_normal_(
            torch.zeros(d_ff, d_model), mean=0, std=std, a=-3 * std, b=3 * std
        )
        self.W3 = torch.nn.Parameter(w3_init)

    def forward(self, x: torch.Tensor):
        W1x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = W1x * torch.sigmoid(W1x)
        W3x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        gated = einsum(silu, W3x, "... d_ff, ... d_ff -> ... d_ff")  # BUGGY: inner product instead of element-wise
        result = einsum(self.W2, gated, "d_model d_ff, ... d_ff -> ... d_model")
        return result

# Test what the buggy einsum actually produces
x = torch.randn(1, 4, 512)  # batch=1, seq_len=4, d_model=512
silu = torch.randn(1, 4, 1344)  # batch=1, seq_len=4, d_ff=1344
W3x = torch.randn(1, 4, 1344)

print("Testing buggy einsum operation:")
print(f"silu shape: {silu.shape}")
print(f"W3x shape: {W3x.shape}")

# The buggy operation
gated_buggy = einsum(silu, W3x, "... d_ff, ... d_ff -> ... d_ff")
print(f"Buggy gated shape: {gated_buggy.shape}")
print(f"Buggy gated value (should be scalar per batch/seq): {gated_buggy[0, 0]}")

# What it should be  
gated_correct = silu * W3x
print(f"Correct gated shape: {gated_correct.shape}")
print(f"Correct gated values [0, 0, :5]: {gated_correct[0, 0, :5]}")

# The buggy einsum pattern is actually computing dot product over d_ff dimension
manual_buggy = torch.sum(silu * W3x, dim=-1, keepdim=True)
print(f"Manual buggy computation shape: {manual_buggy.shape}")
print(f"Manual buggy matches einsum: {torch.allclose(gated_buggy.unsqueeze(-1), manual_buggy)}")

# So the buggy implementation was doing:
# gated = einsum("batch seq d_ff, batch seq d_ff -> batch seq") -> scalar per (batch, seq)
# But then projects this scalar through W2 which expects (batch, seq, d_ff) input

print(f"\nSo the buggy implementation reduces {silu.shape} to {gated_buggy.shape}")
print("Then tries to project this through W2 which expects d_ff dimension!")