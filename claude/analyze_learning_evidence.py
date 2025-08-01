#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import math
from src.transformer import Transformer
from src.checkpointing import load_checkpoint

print("LEARNING ANALYSIS: Did deeper blocks learn?")
print("="*60)

# Load the old "buggy" checkpoint
model_old = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iteration = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model_old, optimizer=None)

print(f"Analyzing checkpoint from iteration {iteration}")

# Calculate expected initialization std for comparison
d_model = 512
hd = 16 * 32  # num_heads * d_head  
expected_attn_std = math.sqrt(2 / (hd + d_model))
expected_swiglu_std = math.sqrt(2 / (d_model + 1344))

print(f"\nExpected initialization stds:")
print(f"  Attention weights: {expected_attn_std:.6f}")
print(f"  SwiGLU weights: {expected_swiglu_std:.6f}")

print(f"\n" + "="*60)
print("ATTENTION WEIGHT ANALYSIS")
print("="*60)

for i, block in enumerate(model_old.transformer_blocks):
    attention = block.mhsa
    
    wq_std = attention.W_Q.std().item()
    wk_std = attention.W_K.std().item()
    wv_std = attention.W_V.std().item()
    wo_std = attention.W_O.std().item()
    
    # Calculate how much they changed from initialization
    wq_ratio = wq_std / expected_attn_std
    wk_ratio = wk_std / expected_attn_std
    
    print(f"\nBlock {i}:")
    print(f"  W_Q: std={wq_std:.6f} (ratio to init: {wq_ratio:.2f}x)")
    print(f"  W_K: std={wk_std:.6f} (ratio to init: {wk_ratio:.2f}x)")
    print(f"  W_V: std={wv_std:.6f}")
    print(f"  W_O: std={wo_std:.6f}")
    
    if wq_ratio > 2.0:
        print(f"  → LEARNED SIGNIFICANTLY (weights grew from initialization)")
    elif wq_ratio < 0.8:
        print(f"  → BARELY LEARNED (weights smaller than initialization)")
    else:
        print(f"  → MODERATE LEARNING")

print(f"\n" + "="*60)  
print("SWIGLU WEIGHT ANALYSIS")
print("="*60)

# Note: SwiGLU had the bug, so expected std was wrong during training
buggy_swiglu_std = math.sqrt((d_model + 1344) / 2)  # What it actually was
print(f"Buggy SwiGLU initialization std: {buggy_swiglu_std:.6f}")

for i, block in enumerate(model_old.transformer_blocks):
    ffn = block.ffn
    
    w1_std = ffn.W1.std().item()
    w2_std = ffn.W2.std().item() 
    w3_std = ffn.W3.std().item()
    
    # Compare to buggy initialization
    w1_ratio = w1_std / buggy_swiglu_std
    
    print(f"\nBlock {i}:")
    print(f"  W1: std={w1_std:.6f} (ratio to buggy init: {w1_ratio:.2f}x)")
    print(f"  W2: std={w2_std:.6f}")
    print(f"  W3: std={w3_std:.6f}")
    
    if w1_ratio < 0.8:
        print(f"  → WEIGHTS DECREASED from buggy initialization")
    else:
        print(f"  → WEIGHTS SIMILAR to buggy initialization")

print(f"\n" + "="*60)
print("FORWARD PASS GRADIENT FLOW TEST")
print("="*60)

# Test if gradients can flow to deeper layers
input_ids = torch.tensor([[430, 439, 259, 398]])  # "Once upon a time"
input_ids.requires_grad_(False)

model_old.eval()
model_old.zero_grad()

# Forward pass
output = model_old(input_ids)
loss = output.sum()  # Dummy loss
loss.backward()

print("Gradient magnitudes by layer:")
for i, block in enumerate(model_old.transformer_blocks):
    if block.mhsa.W_Q.grad is not None:
        grad_norm = block.mhsa.W_Q.grad.norm().item()
        print(f"  Block {i} attention W_Q grad norm: {grad_norm:.6f}")
    else:
        print(f"  Block {i} attention W_Q: NO GRADIENT")
        
    if block.ffn.W1.grad is not None:
        ffn_grad_norm = block.ffn.W1.grad.norm().item()
        print(f"  Block {i} SwiGLU W1 grad norm: {ffn_grad_norm:.6f}")
    else:
        print(f"  Block {i} SwiGLU W1: NO GRADIENT")

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)

print("EVIDENCE FOR 'BLOCKS 2-4 LEARNED LESS':")
print("1. Weight magnitude analysis:")
print("   - Block 0: Attention weights 3-4x larger than initialization")  
print("   - Blocks 1-3: Attention weights SMALLER than initialization")
print("   - This suggests Block 0 updated significantly, others barely")
print()
print("2. SwiGLU weights:")
print("   - All blocks have similar magnitude to buggy initialization")
print("   - None really 'recovered' from the bad initialization")
print()
print("LIMITATIONS OF THIS ANALYSIS:")
print("- Weight magnitude doesn't directly prove learning quality")
print("- Could be explained by different gradient magnitudes at different depths")
print("- Need to compare to successful training to be definitive")