#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import math
from src.transformer import Transformer

print("WEIGHT INITIALIZATION ANALYSIS")
print("="*60)

# Create a fresh model to analyze initialization
vocab_size = 10000
num_heads = 16
d_model = 512
d_ff = 1344
model = Transformer(vocab_size=vocab_size, num_heads=num_heads, d_model=d_model, d_ff=d_ff, rope_theta=10_000, context_length=256, num_layers=4)

print(f"Model configuration:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  d_ff: {d_ff}")
print(f"  num_heads: {num_heads}")
print(f"  num_layers: 4")

print(f"\n" + "="*60)
print("INITIALIZATION ANALYSIS")
print("="*60)

def analyze_tensor(name, tensor, expected_std=None):
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    print(f"{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
    print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    
    if expected_std is not None:
        ratio = std / expected_std
        status = "✓" if 0.8 <= ratio <= 1.2 else "❌"
        print(f"  Expected std: {expected_std:.6f}, Ratio: {ratio:.3f} {status}")
    print()

# 1. Input Embedding
print("1. INPUT EMBEDDING")
print("-" * 30)
embedding = model.input_embedding
analyze_tensor("Input Embedding W", embedding.W)
print(f"  NOTE: Uses std=1.0 (too large for embeddings)")
print(f"  BETTER: std = 1/√d_model = {1/math.sqrt(d_model):.6f}")
print()

# 2. Transformer Blocks
for i, block in enumerate(model.transformer_blocks):
    print(f"2.{i+1}. TRANSFORMER BLOCK {i}")
    print("-" * 30)
    
    # Attention weights
    attention = block.mhsa
    hd = num_heads * (d_model // num_heads)  # Should be d_model
    expected_attn_std = math.sqrt(2 / (hd + d_model))
    
    analyze_tensor(f"  Attention W_Q", attention.W_Q, expected_attn_std)
    analyze_tensor(f"  Attention W_K", attention.W_K, expected_attn_std)
    analyze_tensor(f"  Attention W_V", attention.W_V, expected_attn_std)
    analyze_tensor(f"  Attention W_O", attention.W_O, expected_attn_std)
    
    # RMS Norm weights
    analyze_tensor(f"  Pre-MHSA RMSNorm W", block.pre_mhsa_rmsnorm.W)
    analyze_tensor(f"  Pre-FFN RMSNorm W", block.pre_ffn_rmsnorm.W)
    print(f"  NOTE: RMSNorm initialized to ones (correct)")
    
    # SwiGLU weights  
    ffn = block.ffn
    expected_swiglu_std = math.sqrt(2 / (d_model + d_ff))
    
    analyze_tensor(f"  SwiGLU W1", ffn.W1, expected_swiglu_std)
    analyze_tensor(f"  SwiGLU W2", ffn.W2, expected_swiglu_std)
    analyze_tensor(f"  SwiGLU W3", ffn.W3, expected_swiglu_std)
    print()

# 3. Final Layer Norm and Output Embedding
print("3. FINAL LAYERS")
print("-" * 30)
analyze_tensor("Final RMSNorm W", model.norm.W)
print(f"  NOTE: RMSNorm initialized to ones (correct)")

expected_output_std = math.sqrt(2 / (d_model + vocab_size))
analyze_tensor("Output Embedding W", model.output_embedding.W, expected_output_std)

print("="*60)
print("SUMMARY AND RECOMMENDATIONS")
print("="*60)

print("✅ CORRECT INITIALIZATIONS:")
print("  - SwiGLU: Fixed! Now uses proper Xavier initialization")
print("  - Linear (output embedding): Uses Xavier initialization")
print("  - Attention: Uses Xavier initialization") 
print("  - RMSNorm: Initialized to ones (standard)")

print("\n❌ PROBLEMATIC INITIALIZATIONS:")
print("  - Input Embedding: std=1.0 is TOO LARGE")
print(f"    Current: std=1.0")
print(f"    Recommended: std=1/√d_model = {1/math.sqrt(d_model):.6f}")
print(f"    Ratio: {1.0/(1/math.sqrt(d_model)):.1f}x too large!")

print("\n🔍 WHY THIS MATTERS:")
print("  - Large input embeddings create large initial activations")
print("  - Even with proper SwiGLU init, large inputs can cause instability")
print("  - The √d_model scaling is standard for transformer embeddings")

print("\n🔧 RECOMMENDED FIX:")
print("  Change src/embedding.py line 21:")
print("  FROM: std=1")
print(f"  TO:   std=1/math.sqrt(embedding_dim)  # = {1/math.sqrt(d_model):.6f}")

# Test the impact
print("\n📊 ACTIVATION SCALE ANALYSIS:")
sample_input = torch.randint(0, vocab_size, (1, 4))
with torch.no_grad():
    embeddings = model.input_embedding(sample_input)
    print(f"Input embedding output: mean={embeddings.mean():.6f}, std={embeddings.std():.6f}")
    print(f"Expected for proper init: std ≈ {1/math.sqrt(d_model):.6f}")
    print(f"Current is {embeddings.std().item()/(1/math.sqrt(d_model)):.1f}x larger than expected")