#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import numpy as np
from src.transformer import Transformer

print("FRESH MODEL INITIALIZATION TEST")
print("="*60)
print("Testing if current initialization code has bias")

# Create a completely fresh model with current "fixed" initialization
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)

print("Analyzing fresh model embedding initialization...")

# Get all embedding norms
embedding_norms = []
for i in range(10000):
    norm = model.input_embedding.W[i].norm().item()
    embedding_norms.append(norm)

embedding_norms = np.array(embedding_norms)

print(f"\nFresh model embedding statistics:")
print(f"Overall mean: {embedding_norms.mean():.6f}")
print(f"Overall std: {embedding_norms.std():.6f}")
print(f"Min: {embedding_norms.min():.6f}")
print(f"Max: {embedding_norms.max():.6f}")

# Analyze by token ranges
token_ranges = [
    ("Bytes 0-255", slice(0, 256)),
    ("Low tokens 256-999", slice(256, 1000)), 
    ("Mid tokens 1000-4999", slice(1000, 5000)),
    ("High tokens 5000-9999", slice(5000, 10000))
]

print(f"\nFresh initialization by token range:")
for name, token_slice in token_ranges:
    norms = embedding_norms[token_slice]
    print(f"{name:20}: mean={norms.mean():.6f}, std={norms.std():.6f}")

# Check first and last tokens
low_token_norms = embedding_norms[:100]
high_token_norms = embedding_norms[9000:]

ratio = low_token_norms.mean() / high_token_norms.mean()
print(f"\nBias ratio (first 100 / last 1000): {ratio:.6f}")

if abs(ratio - 1.0) < 0.05:
    print("✅ FRESH INITIALIZATION IS BALANCED")
    print("   → The bias develops during training!")
else:
    print("❌ FRESH INITIALIZATION IS BIASED")
    print("   → There's a bug in the initialization code itself!")

print(f"\nFirst 10 fresh embedding norms:")
for i in range(10):
    print(f"Token {i}: {embedding_norms[i]:.6f}")

print(f"\nLast 10 fresh embedding norms:")
for i in range(9990, 10000):
    print(f"Token {i}: {embedding_norms[i]:.6f}")

# Expected values
expected_std = 1 / np.sqrt(512)  # ~0.044
print(f"\nExpected embedding norm (approx): {expected_std * np.sqrt(512):.6f}")
print(f"Actual mean norm: {embedding_norms.mean():.6f}")

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)

if abs(ratio - 1.0) < 0.05:
    print("✅ INITIALIZATION IS CORRECT")
    print("🚨 THE BIAS DEVELOPS DURING TRAINING!")
    print()
    print("This means there's a systematic bug in the training process")
    print("that makes low token IDs easier to learn/predict.")
    print()
    print("Possible causes:")
    print("1. Bug in gradient computation for embeddings")
    print("2. Data loading order bias")
    print("3. Optimizer update mechanism bias")
    print("4. Loss computation numerical precision issues")
    print()
    print("RECOMMENDATION: Investigate training loop carefully")
    
else:
    print("❌ INITIALIZATION ITSELF IS BIASED")
    print("The bug is in the embedding initialization code.")
    print("Even 'fixed' initialization creates unequal embeddings.")
    print()
    print("RECOMMENDATION: Fix initialization first, then retrain")

# Test if we can create truly uniform embeddings
print(f"\n" + "="*60)
print("TESTING MANUAL UNIFORM INITIALIZATION") 
print("="*60)

# Manually set all embeddings to same std
std = 1 / np.sqrt(512)
with torch.no_grad():
    for i in range(10000):
        # Initialize each embedding independently with same parameters
        model.input_embedding.W[i] = torch.randn(512) * std

# Check the result
manual_norms = []
for i in range(10000):
    norm = model.input_embedding.W[i].norm().item()
    manual_norms.append(norm)

manual_norms = np.array(manual_norms)
manual_low = manual_norms[:100].mean()
manual_high = manual_norms[9000:].mean()
manual_ratio = manual_low / manual_high

print(f"Manual uniform initialization:")
print(f"First 100 tokens mean norm: {manual_low:.6f}")
print(f"Last 1000 tokens mean norm: {manual_high:.6f}")
print(f"Ratio: {manual_ratio:.6f}")

if abs(manual_ratio - 1.0) < 0.05:
    print("✅ Manual initialization is balanced")
    print("→ The issue is in the automatic initialization code")
else:
    print("~ Even manual init has some variance (expected due to randomness)")