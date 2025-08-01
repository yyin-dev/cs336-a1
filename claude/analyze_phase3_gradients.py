#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from src.transformer import Transformer
from src.data_loading import get_batch

def analyze_phase3_gradients():
    """Analyze gradient flow in the Phase 3 model (post both fixes)"""
    
    print("=== PHASE 3 GRADIENT FLOW ANALYSIS ===")
    print("Model: Post SwiGLU + Embedding initialization fixes")
    print("Checkpoint: ../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt")
    
    # Load the Phase 3 model
    checkpoint_path = "../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = "../a1-checkpoints/"
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return
    
    # Model parameters (from CLAUDE.md)
    vocab_size = 10000
    num_heads = 16
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    context_length = 256
    num_layers = 4
    device = "cpu"
    
    # Initialize model
    model = Transformer(
        vocab_size, num_heads, d_model, d_ff, 
        rope_theta, context_length, num_layers
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.train()
    
    # Load training data (with CORRECTED dtype for comparison)
    train_dataset = np.memmap("../a1-data/ts-train-encoded-tiktoken.npy", mode="r", dtype=np.uint16)
    
    # Get a batch
    batch_size = 32
    inputs, targets = get_batch(train_dataset, batch_size, context_length, device)
    
    # Convert to proper types for loss calculation
    inputs = inputs.long()
    targets = targets.long()
    
    # Forward pass
    model.zero_grad()
    outputs = model(inputs)
    
    # Calculate loss (simple cross-entropy for gradient analysis)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Input token range: {inputs.min().item()} to {inputs.max().item()}")
    
    # Analyze gradients by block
    print(f"\n=== GRADIENT FLOW ANALYSIS ===")
    
    for i in range(num_layers):
        block = model.blocks[i]
        
        # Attention gradients
        q_grad_norm = block.attention.W_Q.weight.grad.norm().item() if block.attention.W_Q.weight.grad is not None else 0
        k_grad_norm = block.attention.W_K.weight.grad.norm().item() if block.attention.W_K.weight.grad is not None else 0  
        v_grad_norm = block.attention.W_V.weight.grad.norm().item() if block.attention.W_V.weight.grad is not None else 0
        o_grad_norm = block.attention.W_O.weight.grad.norm().item() if block.attention.W_O.weight.grad is not None else 0
        
        avg_attn_grad = (q_grad_norm + k_grad_norm + v_grad_norm + o_grad_norm) / 4
        
        # FFN gradients  
        gate_grad_norm = block.ffn.W_gate.weight.grad.norm().item() if block.ffn.W_gate.weight.grad is not None else 0
        up_grad_norm = block.ffn.W_up.weight.grad.norm().item() if block.ffn.W_up.weight.grad is not None else 0
        down_grad_norm = block.ffn.W_down.weight.grad.norm().item() if block.ffn.W_down.weight.grad is not None else 0
        
        avg_ffn_grad = (gate_grad_norm + up_grad_norm + down_grad_norm) / 3
        
        print(f"Block {i}:")
        print(f"  Attention grad norm: {avg_attn_grad:.6f}")
        print(f"  FFN grad norm: {avg_ffn_grad:.6f}")
        print(f"  Status: {'✅ LEARNING' if avg_attn_grad > 1e-6 and avg_ffn_grad > 1e-6 else '❌ WEAK/NO GRADIENTS'}")
    
    # Compare with embedding gradients
    if model.embedding.W.grad is not None:
        emb_grad_norm = model.embedding.W.grad.norm().item()
        print(f"\nEmbedding grad norm: {emb_grad_norm:.6f}")
    
    if model.lm_head.weight.grad is not None:
        lm_head_grad_norm = model.lm_head.weight.grad.norm().item()  
        print(f"LM head grad norm: {lm_head_grad_norm:.6f}")

if __name__ == "__main__":
    analyze_phase3_gradients()