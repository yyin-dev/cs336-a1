#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import math
import numpy as np
from src.transformer import Transformer
from src.checkpointing import load_checkpoint
from src.data_loading import get_batch

print("PHASE 3 LEARNING ANALYSIS: Post SwiGLU + Embedding Fixes")
print("="*70)

# Load the Phase 3 checkpoint (post both fixes)
model_phase3 = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iteration = load_checkpoint('../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt', model_phase3, optimizer=None)

print(f"Analyzing Phase 3 checkpoint from iteration {iteration}")

# Calculate expected initialization std for comparison
d_model = 512
hd = 16 * 32  # num_heads * d_head  
expected_attn_std = math.sqrt(2 / (hd + d_model))
expected_swiglu_std = math.sqrt(2 / (d_model + 1344))

print(f"\nExpected initialization stds (CORRECTED):")
print(f"  Attention weights: {expected_attn_std:.6f}")
print(f"  SwiGLU weights: {expected_swiglu_std:.6f}")

print("\n" + "="*70)
print("ATTENTION WEIGHT ANALYSIS")
print("="*70)

# Analyze each block
for i in range(4):
    block = model_phase3.transformer_blocks[i]
    
    W_Q_std = block.mhsa.W_Q.std().item()
    W_K_std = block.mhsa.W_K.std().item()  
    W_V_std = block.mhsa.W_V.std().item()
    W_O_std = block.mhsa.W_O.std().item()
    
    # Compare to expected initialization
    W_Q_ratio = W_Q_std / expected_attn_std
    W_K_ratio = W_K_std / expected_attn_std
    
    print(f"\nBlock {i}:")
    print(f"  W_Q: std={W_Q_std:.6f} (ratio to init: {W_Q_ratio:.2f}x)")
    print(f"  W_K: std={W_K_std:.6f} (ratio to init: {W_K_ratio:.2f}x)")
    print(f"  W_V: std={W_V_std:.6f}")
    print(f"  W_O: std={W_O_std:.6f}")
    
    if W_Q_ratio > 1.5 and W_K_ratio > 1.5:
        print(f"  → LEARNED SIGNIFICANTLY (weights grew from initialization)")
    elif W_Q_ratio > 0.8 and W_K_ratio > 0.8:
        print(f"  → LEARNED MODERATELY")
    else:
        print(f"  → BARELY LEARNED (weights smaller than initialization)")

print("\n" + "="*70)
print("SWIGLU WEIGHT ANALYSIS")
print("="*70)

for i in range(4):
    block = model_phase3.transformer_blocks[i]
    
    W1_std = block.ffn.W1.std().item()
    W2_std = block.ffn.W2.std().item()
    W3_std = block.ffn.W3.std().item()
    
    # Compare to expected (corrected) initialization
    W1_ratio = W1_std / expected_swiglu_std
    
    print(f"\nBlock {i}:")
    print(f"  W_gate: std={W1_std:.6f} (ratio to corrected init: {W1_ratio:.2f}x)")
    print(f"  W_up: std={W2_std:.6f}")
    print(f"  W_down: std={W3_std:.6f}")
    
    if W1_ratio > 1.5:
        print(f"  → LEARNED SIGNIFICANTLY")
    elif W1_ratio > 0.8:
        print(f"  → LEARNED MODERATELY")
    else:
        print(f"  → WEIGHTS CLOSE TO INITIALIZATION")

print("\n" + "="*70)
print("GRADIENT FLOW TEST")
print("="*70)

# Load training data with CORRECT dtype
train_dataset = np.memmap("../a1-data/ts-train-encoded-tiktoken.npy", mode="r", dtype=np.uint16)

# Get a small batch
batch_size = 16
context_length = 256
device = "cpu"
inputs, targets = get_batch(train_dataset, batch_size, context_length, device)

# Convert to long for loss calculation
inputs = inputs.long()
targets = targets.long()

# Forward pass
model_phase3.train()
model_phase3.zero_grad()
outputs = model_phase3(inputs)

# Calculate loss
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(outputs.view(-1, 10000), targets.view(-1))

# Backward pass  
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"Input token range: {inputs.min().item()} to {inputs.max().item()}")

print("\nGradient magnitudes by layer:")
for i in range(4):
    block = model_phase3.transformer_blocks[i]
    
    if block.mhsa.W_Q.grad is not None:
        attn_grad = block.mhsa.W_Q.grad.norm().item()
        print(f"  Block {i} attention W_Q grad norm: {attn_grad:.6f}")
    
    if block.ffn.W1.grad is not None:
        ffn_grad = block.ffn.W1.grad.norm().item()  
        print(f"  Block {i} SwiGLU W1 grad norm: {ffn_grad:.6f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("PHASE 3 MODEL STATUS:")
print("- Post SwiGLU initialization fix (std=0.033 instead of 30.46)")
print("- Post embedding initialization fix (std=0.044 instead of 1.0)")
print("- Training dynamics should be healthy across all blocks")
print("- But model was still learning corrupted data (uint8 instead of uint16)")

if __name__ == "__main__":
    pass