#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import pickle
import numpy as np

# Load tokenizer
with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    res = pickle.load(f)

vocab = res["vocab"]
merges = res["merges"]

print(f"BPE tokenizer vocab size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")
print(f"First 10 vocab items: {dict(list(vocab.items())[:10])}")
print(f"Last 10 vocab items: {dict(list(vocab.items())[-10:])}")

# Check training data
try:
    train_data = np.load('../a1-data/ts-train-encoded-tiktoken.npy')
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Training data token range: {train_data.min()} to {train_data.max()}")
    
    # Check if any tokens exceed vocab size
    invalid_tokens = train_data[train_data >= len(vocab)]
    if len(invalid_tokens) > 0:
        print(f"ERROR: Found {len(invalid_tokens)} tokens >= vocab_size ({len(vocab)})")
        print(f"Invalid token range: {invalid_tokens.min()} to {invalid_tokens.max()}")
    else:
        print(f"✓ All training tokens are valid (< {len(vocab)})")
        
except Exception as e:
    print(f"Could not load training data: {e}")

# Check validation data
try:
    val_data = np.load('../a1-data/ts-valid-encoded-tiktoken.npy')
    print(f"\nValidation data shape: {val_data.shape}")
    print(f"Validation data token range: {val_data.min()} to {val_data.max()}")
    
    # Check if any tokens exceed vocab size
    invalid_tokens = val_data[val_data >= len(vocab)]
    if len(invalid_tokens) > 0:
        print(f"ERROR: Found {len(invalid_tokens)} tokens >= vocab_size ({len(vocab)})")
        print(f"Invalid token range: {invalid_tokens.min()} to {invalid_tokens.max()}")
    else:
        print(f"✓ All validation tokens are valid (< {len(vocab)})")
        
except Exception as e:
    print(f"Could not load validation data: {e}")

# Check token distribution in training data
if 'train_data' in locals():
    print(f"\nToken distribution analysis:")
    unique_tokens, counts = np.unique(train_data, return_counts=True)
    print(f"Number of unique tokens used in training: {len(unique_tokens)}")
    print(f"Most frequent tokens:")
    
    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    for i in range(min(20, len(sorted_indices))):
        token_id = unique_tokens[sorted_indices[i]]
        count = counts[sorted_indices[i]]
        frequency = count / len(train_data)
        
        # Decode token
        if token_id < len(vocab):
            token_bytes = vocab[token_id]
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except:
                token_str = "DECODE_ERROR"
        else:
            token_str = "UNKNOWN"
            
        print(f"  Token {token_id:4d}: {count:8d} times ({frequency:.4f}) | '{token_str}' | {repr(token_bytes)}")
    
    # Check how many tokens are rarely used
    rare_threshold = 10  # tokens appearing less than 10 times
    rare_tokens = counts < rare_threshold
    print(f"\nTokens appearing < {rare_threshold} times: {rare_tokens.sum()} / {len(unique_tokens)}")
    unused_tokens = len(vocab) - len(unique_tokens)
    print(f"Completely unused tokens: {unused_tokens} / {len(vocab)}")

# Check model architecture consistency
print(f"\n" + "="*50)
print("MODEL ARCHITECTURE CHECK")
print("="*50)

from src.transformer import Transformer
from src.checkpointing import load_checkpoint

model = Transformer(vocab_size=len(vocab), num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
try:
    iteration = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model, optimizer=None)
    print(f"✓ Model loaded successfully with vocab_size={len(vocab)}")
    print(f"Model input embedding shape: {model.input_embedding.W.shape}")
    print(f"Model output embedding shape: {model.output_embedding.W.shape}")
    
    expected_input_shape = (len(vocab), 512)  # (vocab_size, d_model)
    expected_output_shape = (len(vocab), 512)  # (vocab_size, d_model)  
    
    if model.input_embedding.W.shape == expected_input_shape:
        print(f"✓ Input embedding shape correct: {expected_input_shape}")
    else:
        print(f"❌ Input embedding shape mismatch: expected {expected_input_shape}, got {model.input_embedding.W.shape}")
        
    if model.output_embedding.W.shape == expected_output_shape:
        print(f"✓ Output embedding shape correct: {expected_output_shape}")
    else:
        print(f"❌ Output embedding shape mismatch: expected {expected_output_shape}, got {model.output_embedding.W.shape}")
        
except Exception as e:
    print(f"❌ Model loading failed: {e}")