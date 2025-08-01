#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
from src.checkpointing import load_checkpoint
from src.transformer import Transformer
import math

print("Loading old checkpoint (buggy SwiGLU)...")
model_old = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iter_old = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model_old, optimizer=None)
print(f"Old checkpoint iteration: {iter_old}")

print("\nLoading new checkpoint (fixed SwiGLU)...")
model_new = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iter_new = load_checkpoint('../a1-checkpoints/fix_swiglu_iter17999.pt', model_new, optimizer=None)
print(f"New checkpoint iteration: {iter_new}")

print("\n" + "="*60)
print("WEIGHT COMPARISON")
print("="*60)

# Compare SwiGLU weights across blocks
for i, (block_old, block_new) in enumerate(zip(model_old.transformer_blocks, model_new.transformer_blocks)):
    print(f"\nTransformer Block {i} SwiGLU:")
    
    # Old SwiGLU weights
    ffn_old = block_old.ffn
    print(f"  OLD - W1: mean={ffn_old.W1.mean().item():.6f}, std={ffn_old.W1.std().item():.6f}")
    print(f"  OLD - W2: mean={ffn_old.W2.mean().item():.6f}, std={ffn_old.W2.std().item():.6f}")
    print(f"  OLD - W3: mean={ffn_old.W3.mean().item():.6f}, std={ffn_old.W3.std().item():.6f}")
    
    # New SwiGLU weights
    ffn_new = block_new.ffn
    print(f"  NEW - W1: mean={ffn_new.W1.mean().item():.6f}, std={ffn_new.W1.std().item():.6f}")
    print(f"  NEW - W2: mean={ffn_new.W2.mean().item():.6f}, std={ffn_new.W2.std().item():.6f}")
    print(f"  NEW - W3: mean={ffn_new.W3.mean().item():.6f}, std={ffn_new.W3.std().item():.6f}")
    
    # Check if weights changed significantly
    w1_change = (ffn_new.W1 - ffn_old.W1).abs().mean().item()
    w2_change = (ffn_new.W2 - ffn_old.W2).abs().mean().item()
    w3_change = (ffn_new.W3 - ffn_old.W3).abs().mean().item()
    print(f"  CHANGE - W1: {w1_change:.6f}, W2: {w2_change:.6f}, W3: {w3_change:.6f}")

# Compare attention weights
print(f"\nAttention Weights:")
for i, (block_old, block_new) in enumerate(zip(model_old.transformer_blocks, model_new.transformer_blocks)):
    attn_old = block_old.mhsa
    attn_new = block_new.mhsa
    
    print(f"\nBlock {i} Attention:")
    print(f"  OLD - W_Q: std={attn_old.W_Q.std().item():.6f}, W_K: std={attn_old.W_K.std().item():.6f}")
    print(f"  NEW - W_Q: std={attn_new.W_Q.std().item():.6f}, W_K: std={attn_new.W_K.std().item():.6f}")
    
    wq_change = (attn_new.W_Q - attn_old.W_Q).abs().mean().item()
    wk_change = (attn_new.W_K - attn_old.W_K).abs().mean().item()
    print(f"  CHANGE - W_Q: {wq_change:.6f}, W_K: {wk_change:.6f}")

# Compare embeddings
print(f"\nEmbeddings:")
input_emb_change = (model_new.input_embedding.W - model_old.input_embedding.W).abs().mean().item()
output_emb_change = (model_new.output_embedding.W - model_old.output_embedding.W).abs().mean().item()

print(f"Input embedding change: {input_emb_change:.6f}")
print(f"Output embedding change: {output_emb_change:.6f}")

print(f"\nInput embedding std - OLD: {model_old.input_embedding.W.std().item():.6f}")
print(f"Input embedding std - NEW: {model_new.input_embedding.W.std().item():.6f}")
print(f"Output embedding std - OLD: {model_old.output_embedding.W.std().item():.6f}")
print(f"Output embedding std - NEW: {model_new.output_embedding.W.std().item():.6f}")

# Test both models with same input
print("\n" + "="*60)
print("FORWARD PASS COMPARISON")
print("="*60)

input_ids = torch.tensor([[430, 439, 259, 398]])  # "Once upon a time" tokens

with torch.no_grad():
    logits_old = model_old(input_ids)
    logits_new = model_new(input_ids)

print(f"Old model logits: mean={logits_old.mean().item():.6f}, std={logits_old.std().item():.6f}")
print(f"New model logits: mean={logits_new.mean().item():.6f}, std={logits_new.std().item():.6f}")

# Check predictions
probs_old = torch.softmax(logits_old[0, -1], dim=-1)
probs_new = torch.softmax(logits_new[0, -1], dim=-1)

top_k_old = torch.topk(probs_old, k=5)
top_k_new = torch.topk(probs_new, k=5)

print(f"\nOld model top 5 predictions:")
for i, (prob, token_id) in enumerate(zip(top_k_old.values, top_k_old.indices)):
    print(f"  {i+1}. Token {token_id.item()}: prob={prob.item():.4f}")

print(f"\nNew model top 5 predictions:")
for i, (prob, token_id) in enumerate(zip(top_k_new.values, top_k_new.indices)):
    print(f"  {i+1}. Token {token_id.item()}: prob={prob.item():.4f}")

# Check low token bias
low_token_mass_old = probs_old[:100].sum().item()
low_token_mass_new = probs_new[:100].sum().item()
print(f"\nOld model - probability mass on tokens 0-99: {low_token_mass_old:.4f}")
print(f"New model - probability mass on tokens 0-99: {low_token_mass_new:.4f}")