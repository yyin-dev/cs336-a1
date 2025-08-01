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
from src.adamw import AdamW
from src.gradient_clipping import clip_gradient

print("TRAINING BIAS SIMULATION")
print("="*60)
print("Simulating extended training to detect systematic bias")

# Load training data
train_data = np.load('../a1-data/ts-train-encoded-tiktoken.npy')

# Create fresh model
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

print(f"Initial model created")
print(f"Training data size: {train_data.shape}")

# Record initial embedding norms for various token ranges
test_tokens = {
    "bytes": [1, 2, 5, 10, 50, 100, 200, 255],
    "low_words": [256, 300, 400, 500, 600, 700, 800, 900],
    "mid_words": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    "high_words": [9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700]
}

def record_embedding_stats(model, description):
    print(f"\n{description}:")
    stats = {}
    for category, tokens in test_tokens.items():
        norms = [model.input_embedding.W[t].norm().item() for t in tokens]
        mean_norm = np.mean(norms)
        max_norm = np.max(norms)
        stats[category] = (mean_norm, max_norm)
        print(f"  {category:12}: mean={mean_norm:.6f}, max={max_norm:.6f}")
    return stats

# Record initial state
initial_stats = record_embedding_stats(model, "INITIAL EMBEDDINGS")

# Training simulation
print(f"\n" + "="*60)
print("TRAINING SIMULATION")
print("="*60)

batch_size = 32  # Smaller for faster testing
context_length = 128
num_steps = 200  # Simulate meaningful training
device = "cpu"

model.train()
losses = []

print(f"Starting training simulation for {num_steps} steps...")

for step in range(num_steps):
    # Get batch
    inputs, targets = get_batch(train_data, batch_size, context_length, device)
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = cross_entropy(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    clip_gradient(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    
    losses.append(loss.item())
    
    # Print progress
    if step % 50 == 0:
        print(f"Step {step:3d}: loss = {loss.item():.4f}")

print(f"Training completed. Final loss: {losses[-1]:.4f}")

# Record final state
final_stats = record_embedding_stats(model, "FINAL EMBEDDINGS")

# Analyze changes
print(f"\n" + "="*60)
print("EMBEDDING CHANGE ANALYSIS")
print("="*60)

print("Change in embedding norms after training:")
for category in test_tokens.keys():
    initial_mean, initial_max = initial_stats[category]
    final_mean, final_max = final_stats[category]
    
    mean_change = final_mean - initial_mean
    max_change = final_max - initial_max
    
    print(f"{category:12}: mean_change={mean_change:+.6f}, max_change={max_change:+.6f}")

# Check for systematic bias
byte_change = final_stats["bytes"][0] - initial_stats["bytes"][0]
word_change = final_stats["mid_words"][0] - initial_stats["mid_words"][0]

print(f"\nSystematic bias analysis:")
print(f"Byte tokens (0-255) mean change: {byte_change:+.6f}")
print(f"Word tokens (1000-8000) mean change: {word_change:+.6f}")

if byte_change > word_change * 1.5:
    print("🚨 BIAS DETECTED: Byte token embeddings growing faster!")
    bias_ratio = byte_change / word_change if word_change != 0 else float('inf')
    print(f"   Bias ratio: {bias_ratio:.2f}x")
elif word_change > byte_change * 1.5:
    print("🚨 REVERSE BIAS: Word token embeddings growing faster!")
    bias_ratio = word_change / byte_change if byte_change != 0 else float('inf')
    print(f"   Reverse bias ratio: {bias_ratio:.2f}x")
else:
    print("✓ No significant systematic bias detected in this simulation")

# Test model predictions
print(f"\n" + "="*60)
print("MODEL PREDICTION ANALYSIS")
print("="*60)

# Test prediction bias
model.eval()
test_input = torch.tensor([[430, 439, 259, 398]])  # "Once upon a time"

with torch.no_grad():
    logits = model(test_input)
    probs = torch.softmax(logits[0, -1], dim=-1)
    
    # Check top predictions
    top_k = torch.topk(probs, k=10)
    print("Top 10 predictions after training simulation:")
    for i, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices)):
        print(f"  {i+1}. Token {token_id.item()}: prob={prob.item():.4f}")
    
    # Check token bias
    low_token_mass = probs[:256].sum().item()
    print(f"\nProbability mass on tokens 0-255: {low_token_mass:.4f}")
    
    if low_token_mass > 0.8:
        print("🚨 SEVERE BIAS: Model strongly prefers byte tokens")
    elif low_token_mass > 0.5:
        print("⚠️  MODERATE BIAS: Model somewhat prefers byte tokens")  
    else:
        print("✓ Reasonable token distribution")

print(f"\n" + "="*60)
print("HYPOTHESIS TESTING")
print("="*60)

print("Possible explanations for the observed bias:")
print()
print("1. EMBEDDING FREQUENCY CORRELATION:")
print("   - Frequent tokens (like '.', ',') naturally get larger gradients")
print("   - But bytes 0-255 are NOT the most frequent tokens")
print("   - So this doesn't fully explain the bias")
print()
print("2. CROSS-ENTROPY NUMERICAL PRECISION:")
print("   - Loss computation might have precision issues with high token IDs")
print("   - Need to investigate loss function implementation")
print()
print("3. OPTIMIZER STATE BIAS:")
print("   - AdamW maintains per-parameter state")
print("   - Might accumulate differently for different token ranges")
print()
print("4. GRADIENT CLIPPING EFFECTS:")
print("   - Clipping might affect different embeddings differently")
print("   - Based on their gradient magnitudes and directions")

if byte_change > word_change * 1.5:
    print(f"\n🔍 BIAS CONFIRMED IN SIMULATION!")
    print(f"Even this short training shows the systematic bias.")
    print(f"Next step: Investigate the specific mechanism causing this.")
else:
    print(f"\n🤔 BIAS NOT REPRODUCED IN SHORT SIMULATION")
    print(f"The bias might require longer training to manifest.")
    print(f"Or there might be a specific trigger we haven't identified.")