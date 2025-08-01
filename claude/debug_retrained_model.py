#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
from src.checkpointing import load_checkpoint
from src.transformer import Transformer
import math

# Load the retrained model
print("Loading retrained model with fixed SwiGLU...")
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iteration = load_checkpoint('../a1-checkpoints/fix_swiglu_iter17999.pt', model, optimizer=None)
print(f"Loaded checkpoint from iteration: {iteration}")

# Test input
input_ids = torch.tensor([[430, 439, 259, 398]])  # "Once upon a time" tokens
print(f"Input tokens: {input_ids}")

# Get input embeddings
x = model.input_embedding(input_ids)
print(f"\nInput embeddings: mean={x.mean().item():.6f}, std={x.std().item():.6f}")

# Test each transformer block
for i, block in enumerate(model.transformer_blocks):
    print(f"\n=== Transformer Block {i} ===")
    
    # Pre-attention norm
    x_norm = block.pre_mhsa_rmsnorm(x)
    print(f"After pre-attention norm: mean={x_norm.mean().item():.6f}, std={x_norm.std().item():.6f}")
    
    # Attention
    attention_out = block.mhsa(x_norm)
    print(f"Attention output: mean={attention_out.mean().item():.6f}, std={attention_out.std().item():.6f}")
    
    # Residual connection
    x_after_attn = x + attention_out
    print(f"After attention residual: mean={x_after_attn.mean().item():.6f}, std={x_after_attn.std().item():.6f}")
    
    # Pre-FFN norm
    x_ffn_norm = block.pre_ffn_rmsnorm(x_after_attn) 
    print(f"After pre-FFN norm: mean={x_ffn_norm.mean().item():.6f}, std={x_ffn_norm.std().item():.6f}")
    
    # FFN (SwiGLU)
    ffn_out = block.ffn(x_ffn_norm)
    print(f"FFN output: mean={ffn_out.mean().item():.6f}, std={ffn_out.std().item():.6f}")
    print(f"FFN extreme values: min={ffn_out.min().item():.2f}, max={ffn_out.max().item():.2f}")
    
    # Final residual
    x = x_after_attn + ffn_out
    print(f"Final block output: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
    print(f"Final extreme values: min={x.min().item():.2f}, max={x.max().item():.2f}")

# Final layer norm
x_final = model.norm(x)
print(f"\nAfter final layer norm: mean={x_final.mean().item():.6f}, std={x_final.std().item():.6f}")

# Output embedding (LM head)
logits = model.output_embedding(x_final)
print(f"Final logits: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
print(f"Logits shape: {logits.shape}")

# Check top predictions
probs = torch.softmax(logits[0, -1], dim=-1)
top_k = torch.topk(probs, k=10)
print(f"\nTop 10 predictions for next token:")
for i, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices)):
    print(f"  {i+1}. Token {token_id.item()}: prob={prob.item():.4f}")

# Check if model is predicting low token IDs
low_token_mass = probs[:100].sum().item()
print(f"\nProbability mass on tokens 0-99: {low_token_mass:.4f}")
print(f"Probability mass on tokens 100+: {1-low_token_mass:.4f}")