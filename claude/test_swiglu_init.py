#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import math
from src.swiglu import SwiGLU

# Test the current SwiGLU initialization
d_model = 512
d_ff = 1344

print("Testing current SwiGLU initialization...")
swiglu = SwiGLU(d_model, d_ff)

print(f"d_model: {d_model}, d_ff: {d_ff}")

# Check the standard deviation used
expected_std = math.sqrt(2 / (d_model + d_ff))
print(f"Expected std: {expected_std:.6f}")

# Check actual weight statistics
print(f"W1 std: {swiglu.W1.std().item():.6f}")
print(f"W2 std: {swiglu.W2.std().item():.6f}")
print(f"W3 std: {swiglu.W3.std().item():.6f}")

# Test forward pass with small input
x = torch.randn(1, 4, d_model) * 0.3  # Reasonable input
output = swiglu(x)

print(f"\nInput: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
print(f"Output: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
print(f"Output extreme values: min={output.min().item():.2f}, max={output.max().item():.2f}")

# Compare with the buggy version
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

    def forward(self, x):
        from einops import einsum
        W1x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = W1x * torch.sigmoid(W1x)
        W3x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        gated = silu * W3x  # Fixed gating operation
        result = einsum(self.W2, gated, "d_model d_ff, ... d_ff -> ... d_model")
        return result

print("\n" + "="*50)
print("Comparing with buggy initialization...")
buggy_swiglu = BuggySwiGLU(d_model, d_ff)

buggy_std = math.sqrt((d_model + d_ff) / 2)
print(f"Buggy std: {buggy_std:.6f}")
print(f"Buggy W1 std: {buggy_swiglu.W1.std().item():.6f}")

buggy_output = buggy_swiglu(x)
print(f"Buggy output: mean={buggy_output.mean().item():.6f}, std={buggy_output.std().item():.6f}")
print(f"Buggy extreme values: min={buggy_output.min().item():.2f}, max={buggy_output.max().item():.2f}")