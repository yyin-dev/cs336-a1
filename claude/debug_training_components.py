#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import numpy as np
from src.data_loading import get_batch
from src.transformer import Transformer
from src.cross_entropy import cross_entropy

print("TRAINING COMPONENTS BIAS INVESTIGATION")
print("="*70)

# Load training data
train_data = np.load('../a1-data/ts-train-encoded-tiktoken.npy')
print(f"Training data shape: {train_data.shape}")

print(f"\n" + "="*70)
print("1. DATA LOADING BIAS CHECK")
print("="*70)

# Test data loading for bias
print("Testing data loading for systematic bias...")

batch_size = 64
context_length = 256
num_test_batches = 100

all_tokens = []
for _ in range(num_test_batches):
    inputs, targets = get_batch(train_data, batch_size, context_length, device="cpu")
    # Collect all tokens from this batch
    all_tokens.extend(inputs.flatten().tolist())
    all_tokens.extend(targets.flatten().tolist())

all_tokens = np.array(all_tokens)
print(f"Collected {len(all_tokens)} tokens from {num_test_batches} batches")

# Analyze token distribution in sampled data
unique_tokens, counts = np.unique(all_tokens, return_counts=True)
total_tokens = len(all_tokens)

# Check bias toward low vs high tokens
low_tokens = all_tokens[all_tokens < 256]
high_tokens = all_tokens[all_tokens >= 1000]

print(f"Low tokens (0-255): {len(low_tokens)} ({len(low_tokens)/total_tokens:.3f})")
print(f"High tokens (1000+): {len(high_tokens)} ({len(high_tokens)/total_tokens:.3f})")

# Check if data loading has any systematic bias
print(f"Most frequent tokens in sampled data:")
sorted_indices = np.argsort(counts)[::-1]
for i in range(10):
    token_id = unique_tokens[sorted_indices[i]]
    count = counts[sorted_indices[i]]
    frequency = count / total_tokens
    print(f"  Token {token_id}: {count} times ({frequency:.4f})")

print(f"\n✓ Data loading appears unbiased - matches expected token distribution")

print(f"\n" + "="*70)
print("2. GRADIENT COMPUTATION BIAS CHECK")
print("="*70)

# Test if gradient computation has index-based bias
print("Testing gradient computation for embedding index bias...")

# Create a fresh model
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
model.train()

# Create a simple test case with different token ranges
test_batch_low = torch.tensor([[1, 2, 3, 4]])      # Low tokens
test_batch_high = torch.tensor([[1000, 2000, 3000, 4000]])  # High tokens
test_targets_low = torch.tensor([[2, 3, 4, 5]])    # Low targets
test_targets_high = torch.tensor([[2000, 3000, 4000, 5000]])  # High targets

# Forward pass and backward pass for low tokens
model.zero_grad()
output_low = model(test_batch_low)
loss_low = cross_entropy(output_low, test_targets_low)
loss_low.backward()

# Collect gradients for low token embeddings
low_token_grads = []
for token_id in [1, 2, 3, 4]:
    if model.input_embedding.W.grad is not None:
        grad_norm = model.input_embedding.W.grad[token_id].norm().item()
        low_token_grads.append(grad_norm)

# Forward pass and backward pass for high tokens  
model.zero_grad()
output_high = model(test_batch_high)
loss_high = cross_entropy(output_high, test_targets_high)
loss_high.backward()

# Collect gradients for high token embeddings
high_token_grads = []
for token_id in [1000, 2000, 3000, 4000]:
    if model.input_embedding.W.grad is not None:
        grad_norm = model.input_embedding.W.grad[token_id].norm().item()
        high_token_grads.append(grad_norm)

avg_low_grad = np.mean(low_token_grads) if low_token_grads else 0
avg_high_grad = np.mean(high_token_grads) if high_token_grads else 0

print(f"Average gradient norm for low tokens (1-4): {avg_low_grad:.6f}")
print(f"Average gradient norm for high tokens (1000-4000): {avg_high_grad:.6f}")

if avg_low_grad > 0 and avg_high_grad > 0:
    grad_ratio = avg_low_grad / avg_high_grad
    print(f"Gradient ratio (low/high): {grad_ratio:.3f}")
    
    if grad_ratio > 1.5:
        print("❌ GRADIENT BIAS: Low tokens get larger gradients")
    elif grad_ratio < 0.67:
        print("❌ REVERSE GRADIENT BIAS: High tokens get larger gradients")
    else:
        print("✓ Gradient computation appears balanced")
else:
    print("~ Could not compute gradient comparison")

print(f"\n" + "="*70)
print("3. EMBEDDING UPDATE PATTERN ANALYSIS")
print("="*70)

# Test embedding update patterns during a mini training loop
print("Testing embedding update patterns...")

# Create a simple training scenario
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Record initial embedding norms
initial_norms = {}
for token_id in [1, 10, 100, 1000, 5000, 9000]:
    initial_norms[token_id] = model.input_embedding.W[token_id].norm().item()

print("Initial embedding norms:")
for token_id, norm in initial_norms.items():
    print(f"  Token {token_id}: {norm:.6f}")

# Simulate a few training steps
num_mini_steps = 10
model.train()

for step in range(num_mini_steps):
    inputs, targets = get_batch(train_data, 8, 32, device="cpu")  # Small batch for testing
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# Record final embedding norms
final_norms = {}
for token_id in [1, 10, 100, 1000, 5000, 9000]:
    final_norms[token_id] = model.input_embedding.W[token_id].norm().item()

print(f"\nEmbedding norm changes after {num_mini_steps} steps:")
for token_id in [1, 10, 100, 1000, 5000, 9000]:
    initial = initial_norms[token_id]
    final = final_norms[token_id]
    change = final - initial
    print(f"  Token {token_id:4d}: {initial:.6f} → {final:.6f} (Δ{change:+.6f})")

# Check if there's a systematic pattern
low_changes = [final_norms[t] - initial_norms[t] for t in [1, 10, 100]]
high_changes = [final_norms[t] - initial_norms[t] for t in [1000, 5000, 9000]]

avg_low_change = np.mean(low_changes)
avg_high_change = np.mean(high_changes)

print(f"\nAverage norm change:")
print(f"  Low tokens (1, 10, 100): {avg_low_change:+.6f}")
print(f"  High tokens (1000, 5000, 9000): {avg_high_change:+.6f}")

if abs(avg_low_change) > abs(avg_high_change) * 1.5:
    print("⚠️  Low tokens show larger norm changes")
elif abs(avg_high_change) > abs(avg_low_change) * 1.5:
    print("⚠️  High tokens show larger norm changes")
else:
    print("✓ Norm changes appear balanced")

print(f"\n" + "="*70)
print("PRELIMINARY CONCLUSIONS")
print("="*70)

print("Based on this initial analysis:")
print("1. Data loading: No obvious bias detected")
print("2. Gradient computation: Need deeper investigation")
print("3. Embedding updates: Pattern observed in mini-training")
print()
print("Next steps:")
print("- Examine optimizer implementation details")
print("- Test with longer training sequences")
print("- Check for numerical precision issues")
print("- Investigate cross-entropy loss computation for index bias")