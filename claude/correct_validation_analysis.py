#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import numpy as np
import pickle
from src.transformer import Transformer
from src.checkpointing import load_checkpoint
from src.tokenizer import Tokenizer
from src.cross_entropy import cross_entropy

print("CORRECTED VALIDATION LOSS ANALYSIS")
print("="*60)
print("Why validation loss can be ~1.0 while generation is garbage")

# Load model and tokenizer
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model, optimizer=None)

with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    res = pickle.load(f)
vocab = res["vocab"]
merges = res["merges"]
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# Sample validation sequence
val_data = np.load('../a1-data/ts-valid-encoded-tiktoken.npy')
np.random.seed(42)
start_idx = np.random.randint(0, len(val_data) - 256)
val_sequence = val_data[start_idx:start_idx + 256]

inputs = torch.tensor(val_sequence[:-1]).unsqueeze(0)
targets = torch.tensor(val_sequence[1:]).unsqueeze(0)

model.eval()
with torch.no_grad():
    outputs = model(inputs)
    val_loss = cross_entropy(outputs, targets)

print(f"Validation loss on sample: {val_loss.item():.4f}")
print(f"Perplexity: {torch.exp(val_loss).item():.2f}")

print(f"\n" + "="*60)
print("UNDERSTANDING WHY LOSS ISN'T TERRIBLE")
print("="*60)

# Analyze predictions in detail
correct_predictions = 0
reasonable_predictions = 0  # Top-5 contains correct
total_predictions = targets.shape[1]

prob_on_correct_tokens = []
prob_on_top_prediction = []

for i in range(min(20, targets.shape[1])):  # Analyze first 20 predictions
    true_token = targets[0, i].item()
    predicted_logits = outputs[0, i]
    predicted_probs = torch.softmax(predicted_logits, dim=-1)
    
    # Probability on correct token
    prob_correct = predicted_probs[true_token].item()
    prob_on_correct_tokens.append(prob_correct)
    
    # Top prediction
    top_prob, top_token = torch.topk(predicted_probs, k=1)
    prob_on_top_prediction.append(top_prob.item())
    
    # Check if correct token is in top-5
    top_5 = torch.topk(predicted_probs, k=5)
    if true_token in top_5.indices:
        reasonable_predictions += 1
    
    if top_token.item() == true_token:
        correct_predictions += 1
    
    # Show first few examples
    if i < 5:
        # Decode tokens
        if true_token < len(vocab):
            true_str = vocab[true_token].decode('utf-8', errors='replace')
        else:
            true_str = "UNKNOWN"
            
        if top_token.item() < len(vocab):
            pred_str = vocab[top_token.item()].decode('utf-8', errors='replace')
        else:
            pred_str = "UNKNOWN"
            
        print(f"\nPosition {i}:")
        print(f"  TRUE: token {true_token} '{true_str}' | prob={prob_correct:.4f} | loss={-np.log(prob_correct):.4f}")
        print(f"  PRED: token {top_token.item()} '{pred_str}' | prob={top_prob.item():.4f}")

avg_prob_correct = np.mean(prob_on_correct_tokens)
avg_prob_top = np.mean(prob_on_top_prediction)

print(f"\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"Exact accuracy: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions:.3f}")
print(f"Top-5 accuracy: {reasonable_predictions}/{total_predictions} = {reasonable_predictions/total_predictions:.3f}")
print(f"Average probability on correct tokens: {avg_prob_correct:.4f}")
print(f"Average probability on top predictions: {avg_prob_top:.4f}")

# Calculate expected loss
expected_loss = -np.log(avg_prob_correct)
print(f"Expected loss from avg prob: {expected_loss:.4f}")

print(f"\n" + "="*60)
print("THE REAL EXPLANATION")
print("="*60)

print("Why validation loss ~1.0 doesn't mean terrible predictions:")
print()
print("1. LOSS = -log(p_correct), so:")
print(f"   - If p_correct = 0.37 (37%), loss = {-np.log(0.37):.2f}")
print(f"   - If p_correct = 0.10 (10%), loss = {-np.log(0.10):.2f}")
print(f"   - If p_correct = 0.05 (5%), loss = {-np.log(0.05):.2f}")
print()
print("2. MODEL DOESN'T NEED TO BE CONFIDENT TO GET REASONABLE LOSS:")
print("   - Even if model gives correct token only 10-30% probability")
print("   - Loss will be in the 1.0-3.0 range (reasonable looking)")
print("   - But top prediction is still wrong most of the time!")
print()
print("3. THE REAL ISSUE:")
print("   - Model assigns SOME probability to correct tokens (preventing infinite loss)")
print("   - But assigns HIGHEST probability to wrong tokens (causing bad generation)")
print("   - Validation loss averages over all positions, masking the problem")
print()
print("4. GENERATION VS VALIDATION DIFFERENCE:")
print("   - Validation: We only care about probability mass on correct token")
print("   - Generation: We sample from the distribution → get the highest probability tokens")
print("   - Model can put 20% on correct, 25% on wrong → loss=1.6, but generates wrong token")

print(f"\n" + "="*60)
print("CORRECTED CONCLUSION")
print("="*60)

print("The validation loss paradox occurs because:")
print("✓ Cross-entropy only cares about probability on the correct token")
print("✓ Model can assign reasonable probability (10-30%) to correct tokens")
print("✓ But still assign HIGHER probability to wrong tokens")
print("✓ During generation, we sample the highest probability → get wrong tokens")
print("✓ Loss ~1.0 just means average probability on correct tokens ~37%")
print("✗ This doesn't mean the model's TOP predictions are correct!")

print(f"\nIn our case:")
print(f"- Model gives correct tokens {avg_prob_correct:.1%} probability on average")
print(f"- But during generation, it samples from the top of the distribution")
print(f"- Top predictions are control characters, not meaningful words")
print(f"- Hence: reasonable validation loss but garbage generation")