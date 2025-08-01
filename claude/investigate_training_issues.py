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

print("TRAINING ISSUES INVESTIGATION")
print("="*60)
print("Are there fundamental issues beyond training time?")

# Load the fully fixed model
model = Transformer(vocab_size=10000, num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
load_checkpoint('../a1-checkpoints/fix_swiglu_and_embedding_iter15999.pt', model, optimizer=None)

# Load tokenizer and data
with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    res = pickle.load(f)
vocab = res["vocab"]
merges = res["merges"]
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

val_data = np.load('../a1-data/ts-valid-encoded-tiktoken.npy')

print(f"\n" + "="*60)
print("HYPOTHESIS 1: MODEL ARCHITECTURE ISSUE")
print("="*60)

# Test if model CAN learn the right patterns given the right input
print("Testing if model can predict common tokens...")

# Create a simple test: predict " the" after "Once upon a time"
# Token 263 is " the" according to our previous analysis
test_sequence = [430, 439, 259, 398, 263]  # "Once upon a time the"
test_input = torch.tensor(test_sequence[:-1]).unsqueeze(0)
test_target = torch.tensor([test_sequence[-1]])

model.eval()
with torch.no_grad():
    logits = model(test_input)
    probs = torch.softmax(logits[0, -1], dim=-1)
    
    prob_correct = probs[test_target[0]].item()
    
    print(f"Input sequence: {test_sequence[:-1]}")
    print(f"Target token: {test_target[0].item()} (' the')")
    print(f"Model probability on ' the': {prob_correct:.6f}")
    
    if prob_correct < 0.001:
        print("❌ Model assigns virtually no probability to common word ' the'")
    else:
        print(f"✓ Model assigns some probability to ' the'")

print(f"\n" + "="*60)
print("HYPOTHESIS 2: TRAINING DATA LOADING ISSUE")
print("="*60)

# Verify training data actually contains the patterns we expect
print("Checking if training data has the patterns model should learn...")

sample_size = 10000
start_idx = np.random.randint(0, len(val_data) - sample_size)
data_sample = val_data[start_idx:start_idx + sample_size]

# Count occurrences of common tokens
common_tokens = [263, 267, 259, 46, 44]  # " the", " and", " a", ".", ","
token_counts = {}

for token_id in common_tokens:
    count = np.sum(data_sample == token_id)
    frequency = count / sample_size
    token_counts[token_id] = (count, frequency)
    
    # Decode token
    if token_id in vocab:
        token_str = vocab[token_id].decode('utf-8', errors='replace')
    else:
        token_str = "UNKNOWN"
    
    print(f"Token {token_id} ('{token_str}'): {count} times ({frequency:.4f} frequency)")

# Check first few bytes
byte_counts = {}
for token_id in range(10):
    count = np.sum(data_sample == token_id)
    frequency = count / sample_size
    byte_counts[token_id] = (count, frequency)
    
    if token_id in vocab:
        token_str = vocab[token_id].decode('utf-8', errors='replace')
    else:
        token_str = "UNKNOWN"
    
    print(f"Byte token {token_id} ('{repr(token_str)}'): {count} times ({frequency:.4f} frequency)")

word_total = sum(count for count, _ in token_counts.values())
byte_total = sum(count for count, _ in byte_counts.values())

print(f"\nData distribution:")
print(f"Common word tokens: {word_total} ({word_total/sample_size:.3f})")  
print(f"First 10 byte tokens: {byte_total} ({byte_total/sample_size:.3f})")

if byte_total > word_total:
    print("❌ ISSUE: Training data has more byte tokens than word tokens")
else:
    print("✓ Training data has expected token distribution")

print(f"\n" + "="*60)
print("HYPOTHESIS 3: LOSS COMPUTATION ISSUE")
print("="*60)

# Test if loss computation is biased toward certain tokens
print("Testing loss computation on different token types...")

# Create test sequences with common words vs bytes
word_sequence = torch.tensor([[430, 439, 259, 263]])  # "Once upon a the"
byte_sequence = torch.tensor([[1, 2, 3, 4]])  # Random low tokens

word_targets = torch.tensor([[263]])  # Target: " the"
byte_targets = torch.tensor([[4]])    # Target: byte token 4

with torch.no_grad():
    word_logits = model(word_sequence)
    byte_logits = model(byte_sequence)
    
    word_loss = cross_entropy(word_logits, word_targets)
    byte_loss = cross_entropy(byte_logits, byte_targets)
    
    print(f"Loss predicting word token (' the'): {word_loss.item():.4f}")
    print(f"Loss predicting byte token (4): {byte_loss.item():.4f}")
    
    if byte_loss < word_loss:
        print("❌ Model finds byte tokens easier to predict")
    else:
        print("✓ Loss computation seems balanced")

print(f"\n" + "="*60)
print("HYPOTHESIS 4: EMBEDDING LOOKUP ISSUE")
print("="*60)

# Check if embeddings for word tokens vs byte tokens are different
print("Comparing embedding magnitudes...")

word_tokens = [263, 267, 259]  # " the", " and", " a"
byte_tokens = [1, 2, 3]        # Low byte tokens

word_embedding_norms = []
byte_embedding_norms = []

for token_id in word_tokens:
    embedding = model.input_embedding.W[token_id]
    norm = embedding.norm().item()
    word_embedding_norms.append(norm)
    print(f"Word token {token_id} embedding norm: {norm:.6f}")

for token_id in byte_tokens:
    embedding = model.input_embedding.W[token_id]
    norm = embedding.norm().item()
    byte_embedding_norms.append(norm)
    print(f"Byte token {token_id} embedding norm: {norm:.6f}")

avg_word_norm = np.mean(word_embedding_norms)
avg_byte_norm = np.mean(byte_embedding_norms)

print(f"\nAverage word embedding norm: {avg_word_norm:.6f}")
print(f"Average byte embedding norm: {avg_byte_norm:.6f}")
print(f"Ratio (word/byte): {avg_word_norm/avg_byte_norm:.3f}")

if avg_byte_norm > avg_word_norm * 1.5:
    print("❌ Byte embeddings are significantly larger")
elif avg_word_norm > avg_byte_norm * 1.5:  
    print("❌ Word embeddings are significantly larger")
else:
    print("✓ Embedding magnitudes seem balanced")

print(f"\n" + "="*60)
print("CONCLUSIONS")
print("="*60)

print("Based on this analysis:")
print("1. If model assigns very low prob to common words → architectural issue")
print("2. If training data is biased toward bytes → data preprocessing issue")
print("3. If loss computation favors bytes → training loop issue")  
print("4. If embeddings are imbalanced → initialization issue")
print("5. If all above look OK → likely just needs more training time")