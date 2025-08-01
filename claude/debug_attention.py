#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
from src.checkpointing import load_checkpoint
from src.transformer import Transformer
import math

# Load the model
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iteration = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model, optimizer=None)

print('Attention weight statistics:')
for i, block in enumerate(model.transformer_blocks):
    attention = block.mhsa
    
    # Expected initialization std
    hd = 16 * 32  # num_heads * d
    d_model = 512
    expected_std = math.sqrt(2 / (hd + d_model))
    
    print(f'\nBlock {i}:')
    print(f'  Expected init std: {expected_std:.6f}')
    print(f'  W_Q: mean={attention.W_Q.mean().item():.6f}, std={attention.W_Q.std().item():.6f}')
    print(f'  W_K: mean={attention.W_K.mean().item():.6f}, std={attention.W_K.std().item():.6f}')
    print(f'  W_V: mean={attention.W_V.mean().item():.6f}, std={attention.W_V.std().item():.6f}')
    print(f'  W_O: mean={attention.W_O.mean().item():.6f}, std={attention.W_O.std().item():.6f}')

# Test attention computation step by step
print('\nTesting attention computation:')
input_ids = torch.tensor([[430, 439, 259, 398]])  # "Once upon a time" tokens

# Get input embeddings
x = model.input_embedding(input_ids)
print(f'Input embeddings: mean={x.mean().item():.6f}, std={x.std().item():.6f}')

# Test first transformer block step by step
block = model.transformer_blocks[0]

# Pre-attention norm
x_norm = block.pre_mhsa_rmsnorm(x)
print(f'After pre-attention norm: mean={x_norm.mean().item():.6f}, std={x_norm.std().item():.6f}')

# Attention projection  
attention = block.mhsa
from einops import einsum
Q = einsum(attention.W_Q, x_norm, "hd d_model, batch seq_len d_model -> batch seq_len hd")
K = einsum(attention.W_K, x_norm, "hd d_model, batch seq_len d_model -> batch seq_len hd") 
V = einsum(attention.W_V, x_norm, "hd d_model, batch seq_len d_model -> batch seq_len hd")

print(f'Q projection: mean={Q.mean().item():.6f}, std={Q.std().item():.6f}, max={Q.max().item():.2f}')
print(f'K projection: mean={K.mean().item():.6f}, std={K.std().item():.6f}, max={K.max().item():.2f}')
print(f'V projection: mean={V.mean().item():.6f}, std={V.std().item():.6f}, max={V.max().item():.2f}')

# Check for extreme values
print(f'Q extreme values: min={Q.min().item():.2f}, max={Q.max().item():.2f}')
print(f'K extreme values: min={K.min().item():.2f}, max={K.max().item():.2f}')
print(f'V extreme values: min={V.min().item():.2f}, max={V.max().item():.2f}')

# Full attention forward
attention_out = block.mhsa(x_norm)
print(f'Attention output: mean={attention_out.mean().item():.6f}, std={attention_out.std().item():.6f}')

# Residual connection
x_after_attn = x + attention_out
print(f'After attention residual: mean={x_after_attn.mean().item():.6f}, std={x_after_attn.std().item():.6f}')

# Pre-FFN norm
x_ffn_norm = block.pre_ffn_rmsnorm(x_after_attn) 
print(f'After pre-FFN norm: mean={x_ffn_norm.mean().item():.6f}, std={x_ffn_norm.std().item():.6f}')

# FFN
ffn_out = block.ffn(x_ffn_norm)
print(f'FFN output: mean={ffn_out.mean().item():.6f}, std={ffn_out.std().item():.6f}')
print(f'FFN extreme values: min={ffn_out.min().item():.2f}, max={ffn_out.max().item():.2f}')

# Final residual
final_out = x_after_attn + ffn_out
print(f'Final block output: mean={final_out.mean().item():.6f}, std={final_out.std().item():.6f}')
print(f'Final extreme values: min={final_out.min().item():.2f}, max={final_out.max().item():.2f}')