#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import numpy as np
from src.transformer import Transformer
from src.checkpointing import load_checkpoint

print("EMBEDDING BIAS INVESTIGATION")
print("="*60)

# Load the fully fixed model
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
load_checkpoint('../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt', model, optimizer=None)

print("Analyzing embedding bias patterns...")

# Get all embedding norms
embedding_norms = []
for i in range(10000):
    norm = model.input_embedding.W[i].norm().item()
    embedding_norms.append(norm)

embedding_norms = np.array(embedding_norms)

print(f"\nEmbedding statistics:")
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

print(f"\nAnalysis by token range:")
for name, token_slice in token_ranges:
    norms = embedding_norms[token_slice]
    print(f"{name:20}: mean={norms.mean():.6f}, std={norms.std():.6f}, max={norms.max():.6f}")

# Check if there's a clear pattern
print(f"\nFirst 20 token norms:")
for i in range(20):
    print(f"Token {i:2d}: {embedding_norms[i]:.6f}")

print(f"\nLast 20 token norms:")
for i in range(9980, 10000):
    print(f"Token {i:4d}: {embedding_norms[i]:.6f}")

# Look for the bias pattern
low_token_norms = embedding_norms[:100]  # First 100 tokens
high_token_norms = embedding_norms[1000:]  # Skip middle range

print(f"\n" + "="*60)
print("BIAS ANALYSIS")
print("="*60)

print(f"First 100 tokens (bytes):")
print(f"  Mean: {low_token_norms.mean():.6f}")
print(f"  Max: {low_token_norms.max():.6f}")
print(f"  Std: {low_token_norms.std():.6f}")

print(f"Tokens 1000+ (words):")
print(f"  Mean: {high_token_norms.mean():.6f}")
print(f"  Max: {high_token_norms.max():.6f}")
print(f"  Std: {high_token_norms.std():.6f}")

ratio = low_token_norms.mean() / high_token_norms.mean()
print(f"\nBias ratio (low/high): {ratio:.3f}")

if ratio > 2.0:
    print("❌ SEVERE BIAS: Low tokens have much larger embeddings")
elif ratio > 1.5:
    print("⚠️  MODERATE BIAS: Low tokens have larger embeddings")
elif ratio < 0.67:
    print("⚠️  REVERSE BIAS: High tokens have larger embeddings")
else:
    print("✓ BALANCED: Similar embedding magnitudes")

# Check if this correlates with training frequency
print(f"\n" + "="*60)
print("HYPOTHESIS: FREQUENT TOKENS GET LARGER EMBEDDINGS")
print("="*60)

# Load some training data to check correlation
try:
    train_data = np.load('../a1-data/ts-train-encoded-tiktoken.npy')
    
    # Count token frequencies
    unique_tokens, counts = np.unique(train_data[:1000000], return_counts=True)
    
    # Get embedding norms for these tokens
    token_freq_norm_pairs = []
    for token_id, count in zip(unique_tokens, counts):
        if token_id < 10000:  # Valid token
            norm = embedding_norms[token_id]
            frequency = count / 1000000
            token_freq_norm_pairs.append((token_id, frequency, norm))
    
    # Sort by frequency
    token_freq_norm_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 20 most frequent tokens:")
    for i in range(min(20, len(token_freq_norm_pairs))):
        token_id, freq, norm = token_freq_norm_pairs[i]
        print(f"Token {token_id:4d}: freq={freq:.6f}, norm={norm:.6f}")
    
    print("\nBottom 20 least frequent tokens:")
    for i in range(max(0, len(token_freq_norm_pairs)-20), len(token_freq_norm_pairs)):
        token_id, freq, norm = token_freq_norm_pairs[i]
        print(f"Token {token_id:4d}: freq={freq:.6f}, norm={norm:.6f}")
    
    # Calculate correlation
    frequencies = [pair[1] for pair in token_freq_norm_pairs]
    norms = [pair[2] for pair in token_freq_norm_pairs]
    
    correlation = np.corrcoef(frequencies, norms)[0, 1]
    print(f"\nCorrelation between frequency and norm: {correlation:.4f}")
    
    if correlation > 0.3:
        print("✓ POSITIVE CORRELATION: Frequent tokens have larger embeddings (expected)")
    elif correlation < -0.3:
        print("❌ NEGATIVE CORRELATION: Frequent tokens have smaller embeddings (problematic)")
    else:
        print("~ WEAK CORRELATION: No clear pattern")
        
except Exception as e:
    print(f"Could not load training data: {e}")

print(f"\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

print("The embedding bias could be caused by:")
print("1. NORMAL TRAINING DYNAMICS: Frequent tokens naturally get larger embeddings")
print("2. INITIALIZATION ISSUE: Even 'fixed' init might have subtle bias")
print("3. TRAINING BUG: Something in training loop favors low token IDs")
print("4. DATA ISSUE: Low token IDs appear more frequently due to encoding bug")

print(f"\nNext step: Check if a FRESH model with fixed init has balanced embeddings")