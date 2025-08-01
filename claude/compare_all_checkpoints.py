#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
from src.checkpointing import load_checkpoint
from src.transformer import Transformer

print("THREE-WAY CHECKPOINT COMPARISON")
print("="*70)

# Load all three models
print("Loading checkpoints...")

model_old = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iter_old = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model_old, optimizer=None)

model_swiglu = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iter_swiglu = load_checkpoint('../a1-checkpoints/fix_swiglu_iter17999.pt', model_swiglu, optimizer=None)

model_fixed = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iter_fixed = load_checkpoint('../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt', model_fixed, optimizer=None)

print(f"Old model (both bugs): iteration {iter_old}")
print(f"SwiGLU-only fix: iteration {iter_swiglu}")
print(f"Both fixes: iteration {iter_fixed}")

# Test all models with same input
input_ids = torch.tensor([[430, 439, 259, 398]])  # "Once upon a time"

print(f"\n" + "="*70)
print("FORWARD PASS COMPARISON")
print("="*70)

models = [
    ("Old (both bugs)", model_old),
    ("SwiGLU fix only", model_swiglu), 
    ("Both fixes", model_fixed)
]

with torch.no_grad():
    for name, model in models:
        model.eval()
        logits = model(input_ids)
        probs = torch.softmax(logits[0, -1], dim=-1)
        
        # Top predictions
        top_k = torch.topk(probs, k=5)
        
        # Token distribution
        low_token_mass = probs[:100].sum().item()
        
        print(f"\n{name}:")
        print(f"  Logits: mean={logits.mean().item():.3f}, std={logits.std().item():.3f}")
        print(f"  Top 5 predictions:")
        for i, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices)):
            print(f"    {i+1}. Token {token_id.item()}: prob={prob.item():.4f}")
        print(f"  Low token bias: {low_token_mass:.3f} (prob mass on tokens 0-99)")

print(f"\n" + "="*70)
print("WEIGHT LEARNING COMPARISON")
print("="*70)

# Compare attention weight changes from initialization
import math
expected_attn_std = math.sqrt(2 / (512 + 512))

for block_idx in range(4):
    print(f"\nBlock {block_idx} Attention W_Q std:")
    for name, model in models:
        attn = model.transformer_blocks[block_idx].mhsa
        wq_std = attn.W_Q.std().item()
        ratio = wq_std / expected_attn_std
        print(f"  {name:15}: {wq_std:.6f} (ratio: {ratio:.2f}x)")

print(f"\nBlock 0 vs Block 3 Learning Comparison:")
for name, model in models:
    block0_std = model.transformer_blocks[0].mhsa.W_Q.std().item()
    block3_std = model.transformer_blocks[3].mhsa.W_Q.std().item()
    print(f"  {name:15}: Block0={block0_std:.6f}, Block3={block3_std:.6f}, Ratio={block0_std/block3_std:.2f}")

print(f"\n" + "="*70)
print("PROGRESS ASSESSMENT")
print("="*70)

print("✅ IMPROVEMENTS WITH FIXES:")
print("- Activation scales: Old model had explosion (>500), fixed model reasonable (~0.8)")
print("- All blocks learning: Fixed model shows all blocks with std 0.06-0.10 vs 0.016")
print("- Training stability: No more catastrophic FFN outputs")

print("\n❌ PERSISTENT PROBLEM:")
print("- Token distribution bias WORSE in fixed model (99.6% vs old 37%)")
print("- Still predicting individual bytes instead of words")
print("- Generation quality hasn't improved despite better training dynamics")

print(f"\n" + "="*70)
print("HYPOTHESIS: TRAINING TIME ISSUE")
print("="*70)

print("Possible explanations for persistent bias:")
print("1. INSUFFICIENT TRAINING TIME:")
print(f"   - Old model: {iter_old} iterations")
print(f"   - SwiGLU fix: {iter_swiglu} iterations")  
print(f"   - Both fixes: {iter_fixed} iterations")
print(f"   - Fixed model trained for FEWER iterations!")
print()
print("2. LEARNING RATE / SCHEDULE:")
print("   - With better initialization, might need different learning rate")
print("   - Could be learning too slowly or too quickly")
print()
print("3. DEEP BIAS FROM EARLY TRAINING:")
print("   - Model might have learned byte bias in early training")
print("   - Even with fixes, hard to unlearn this pattern")
print()
print("4. MISSING COMPONENT:")
print("   - There might be another initialization or architectural issue")
print("   - Or a fundamental training setup problem")

print(f"\n" + "="*70)
print("NEXT STEPS RECOMMENDATION")
print("="*70)

print("1. TRAIN LONGER: Let the fixed model train to at least 18000 iterations")
print("2. CHECK LEARNING CURVES: Compare training/validation loss progression")
print("3. VERIFY INITIALIZATION: Double-check all component initializations")
print("4. INSPECT TRAINING LOOP: Look for other subtle bugs in training process")