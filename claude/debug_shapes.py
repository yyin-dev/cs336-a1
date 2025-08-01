#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
from src.checkpointing import load_checkpoint
from src.transformer import Transformer

# Load the model
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iteration = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model, optimizer=None)

print(f'Input embedding W shape: {model.input_embedding.W.shape}')
print(f'Output embedding W shape: {model.output_embedding.W.shape}')

# Test the initialization scales
from math import sqrt
input_init_std = 1.0  # From embedding.py, it uses std=1 in trunc_normal_
output_init_std = sqrt(2 / (512 + 10000))  # From linear.py

print(f'Expected input embedding init std: {input_init_std}')
print(f'Expected output embedding init std: {output_init_std:.6f}')

# Test forward pass step by step
input_ids = torch.tensor([[430, 439, 259, 398]])  # "Once upon a time" tokens
print(f'\nStep-by-step forward pass:')
print(f'Input tokens: {input_ids[0].tolist()}')

# Input embedding
x = model.input_embedding(input_ids)
print(f'After input embedding: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}')

# Through transformer blocks
for i, block in enumerate(model.transformer_blocks):
    x = block(x)
    print(f'After transformer block {i}: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}')

# Layer norm
x = model.norm(x)
print(f'After layer norm: shape={x.shape}, mean={x.mean().item():.6f}, std={x.std().item():.6f}')

# Output embedding (Linear layer)
print(f'Before output embedding - x[:, -1, :5]: {x[0, -1, :5]}')
output = model.output_embedding(x)
print(f'After output embedding: shape={output.shape}, mean={output.mean().item():.6f}, std={output.std().item():.6f}')
print(f'Output logits for last token [:10]: {output[0, -1, :10]}')

# Check what happens if we scale the output embedding weights
print(f'\nTesting with scaled output embedding:')
original_W = model.output_embedding.W.clone()
model.output_embedding.W.data *= 0.1  # Scale down
output_scaled = model.output_embedding(x)
print(f'With 0.1x scaling: {output_scaled[0, -1, :10]}')

model.output_embedding.W.data = original_W * 10  # Scale up  
output_scaled = model.output_embedding(x)
print(f'With 10x scaling: {output_scaled[0, -1, :10]}')

# Restore original
model.output_embedding.W.data = original_W