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

print(f'Loaded iteration: {iteration}')
print(f'Input embedding weight stats:')
print(f'  mean: {model.input_embedding.W.mean().item():.6f}')
print(f'  std: {model.input_embedding.W.std().item():.6f}')
print(f'  min: {model.input_embedding.W.min().item():.6f}')
print(f'  max: {model.input_embedding.W.max().item():.6f}')

print(f'Output embedding weight stats:')
print(f'  mean: {model.output_embedding.W.mean().item():.6f}')
print(f'  std: {model.output_embedding.W.std().item():.6f}')
print(f'  min: {model.output_embedding.W.min().item():.6f}')
print(f'  max: {model.output_embedding.W.max().item():.6f}')

# Check if weights look reasonable
print(f'Are embeddings identical? {torch.equal(model.input_embedding.W, model.output_embedding.W)}')

# Check the first few rows of embeddings for tokens 0-10
print('\nFirst 10 input embedding vectors (first 5 dims):')
for i in range(10):
    print(f'  Token {i}: {model.input_embedding.W[i, :5]}')

print('\nFirst 10 output embedding vectors (first 5 dims):')  
for i in range(10):
    print(f'  Token {i}: {model.output_embedding.W[i, :5]}')

# Test forward pass with a simple input
input_ids = torch.tensor([[0, 1, 2, 3]])  # batch_size=1, seq_len=4
with torch.no_grad():
    output = model(input_ids)
    print(f'\nForward pass test:')
    print(f'  Input shape: {input_ids.shape}')
    print(f'  Output shape: {output.shape}')
    print(f'  Output logits for last token, first 10: {output[0, -1, :10]}')
    
    # Check if output is reasonable
    logits_last = output[0, -1]  # (vocab_size,)
    probs = torch.softmax(logits_last, dim=-1)
    top_10_probs, top_10_indices = torch.topk(probs, 10)
    print(f'  Top 10 token probabilities:')
    for i in range(10):
        token_id = int(top_10_indices[i])
        prob = top_10_probs[i].item()
        print(f'    Token {token_id}: {prob:.4f}')