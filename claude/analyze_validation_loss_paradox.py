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

print("VALIDATION LOSS PARADOX ANALYSIS")
print("="*60)
print("Why can validation loss be good while generation is garbage?")

# Load model and tokenizer
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model, optimizer=None)

with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    res = pickle.load(f)
vocab = res["vocab"]
merges = res["merges"]
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# Load validation data
val_data = np.load('../a1-data/ts-valid-encoded-tiktoken.npy')
print(f"Validation data shape: {val_data.shape}")

print(f"\n" + "="*60)
print("WHAT THE MODEL LEARNED VS WHAT IT SHOULD LEARN")
print("="*60)

# Sample a validation sequence
np.random.seed(42)
start_idx = np.random.randint(0, len(val_data) - 256)
val_sequence = val_data[start_idx:start_idx + 256]

print("Validation sequence analysis:")
print(f"Tokens: {val_sequence[:10]}...{val_sequence[-10:]}")

# Decode a few tokens to see what they represent
decoded_sample = tokenizer.decode(val_sequence[:20])
print(f"Decoded sample: '{decoded_sample}'")

# Create input/target pairs
inputs = torch.tensor(val_sequence[:-1]).unsqueeze(0)  # Shape: (1, 255)
targets = torch.tensor(val_sequence[1:]).unsqueeze(0)   # Shape: (1, 255)

model.eval()
with torch.no_grad():
    outputs = model(inputs)  # Shape: (1, 255, 10000)
    val_loss = cross_entropy(outputs, targets)
    
print(f"\nValidation loss on this sequence: {val_loss.item():.4f}")
print(f"Perplexity: {torch.exp(val_loss).item():.2f}")

# Now let's analyze what the model ACTUALLY predicts vs what it SHOULD predict
print(f"\n" + "="*60)
print("PREDICTION ANALYSIS: Model vs Truth")
print("="*60)

# Take first 5 predictions
for i in range(5):
    true_token = targets[0, i].item()
    predicted_logits = outputs[0, i]  # Logits for position i
    predicted_probs = torch.softmax(predicted_logits, dim=-1)
    
    # Get top 5 predictions
    top_k = torch.topk(predicted_probs, k=5)
    
    # Decode true token
    if true_token < len(vocab):
        true_token_bytes = vocab[true_token]
        true_token_str = true_token_bytes.decode('utf-8', errors='replace')
    else:
        true_token_str = "UNKNOWN"
    
    print(f"\nPosition {i}:")
    print(f"  TRUE token {true_token}: '{true_token_str}' | prob={predicted_probs[true_token].item():.4f}")
    print(f"  MODEL's top 5 predictions:")
    
    for j, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices)):
        token_id = token_id.item()
        prob = prob.item()
        
        if token_id < len(vocab):
            pred_token_bytes = vocab[token_id]
            pred_token_str = pred_token_bytes.decode('utf-8', errors='replace')
        else:
            pred_token_str = "UNKNOWN"
            
        marker = "✓" if token_id == true_token else " "
        print(f"    {marker} {j+1}. Token {token_id:4d}: prob={prob:.4f} | '{pred_token_str}'")

print(f"\n" + "="*60)
print("WHY VALIDATION LOSS CAN BE MISLEADING")
print("="*60)

# Calculate loss contribution from different token types
total_loss = 0
byte_loss = 0
word_loss = 0
num_byte_tokens = 0
num_word_tokens = 0

for i in range(targets.shape[1]):
    true_token = targets[0, i].item()
    predicted_logits = outputs[0, i]
    predicted_probs = torch.softmax(predicted_logits, dim=-1)
    
    # Calculate negative log likelihood for this token
    token_loss = -torch.log(predicted_probs[true_token]).item()
    total_loss += token_loss
    
    # Categorize token
    if true_token < len(vocab):
        token_bytes = vocab[true_token]
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
            if len(token_str) == 1 and ord(token_str) < 256:
                # Single byte token
                byte_loss += token_loss
                num_byte_tokens += 1
            else:
                # Multi-character token
                word_loss += token_loss
                num_word_tokens += 1
        except:
            word_loss += token_loss
            num_word_tokens += 1

avg_loss = total_loss / targets.shape[1]
avg_byte_loss = byte_loss / max(num_byte_tokens, 1)
avg_word_loss = word_loss / max(num_word_tokens, 1)

print(f"Loss breakdown:")
print(f"  Overall average loss: {avg_loss:.4f}")
print(f"  Average loss on byte tokens: {avg_byte_loss:.4f} ({num_byte_tokens} tokens)")
print(f"  Average loss on word tokens: {avg_word_loss:.4f} ({num_word_tokens} tokens)")
print(f"  Byte token fraction: {num_byte_tokens/targets.shape[1]:.3f}")

print(f"\nKey insight:")
if avg_byte_loss < avg_word_loss:
    print(f"✓ Model is BETTER at predicting byte tokens than word tokens!")
    print(f"  This explains why it generates bytes instead of words")
else:
    print(f"✗ Model is actually worse at byte tokens - need deeper analysis")

print(f"\n" + "="*60)
print("THE EMBEDDING INITIALIZATION CONNECTION")
print("="*60)

print("How large embedding initialization might cause byte-token bias:")
print()
print("1. ACTIVATION MAGNITUDE THEORY:")
print("   - Large embeddings → large activations → training instability")
print("   - Model learns to suppress large values → favors low-magnitude weights")
print("   - Low token IDs (0-255) happen to be bytes → accidental bias")
print()
print("2. OPTIMIZATION LANDSCAPE THEORY:")
print("   - Large initial activations make optimization harder")
print("   - Model finds 'easy' patterns first (byte-level regularities)")
print("   - Gets stuck in local minimum before learning word-level patterns")
print()
print("3. GRADIENT FLOW THEORY:")
print("   - Large embeddings → large gradients → gradient clipping kicks in")
print("   - Clipping affects different token embeddings differently")
print("   - Bias emerges toward tokens that 'fit' the clipped gradients")

print(f"\nNOTE: The exact mechanism is complex, but the correlation is clear:")
print(f"Large embedding init + training instability → byte-token bias")