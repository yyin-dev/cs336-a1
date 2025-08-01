#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import pickle
from src.transformer import Transformer
from src.checkpointing import load_checkpoint
from src.tokenizer import Tokenizer
from src.softmax import softmax
from einops import rearrange

# Load tokenizer
with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    res = pickle.load(f)

vocab = res["vocab"]
merges = res["merges"]
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

print(f"Vocab size: {len(vocab)}")

# Test prompt
prompt = "Once upon a time"
prompt_tokens = tokenizer.encode(prompt)
print(f"Prompt: '{prompt}'")
print(f"Prompt tokens: {prompt_tokens}")

# Decode back to verify
decoded = tokenizer.decode(prompt_tokens)
print(f"Decoded back: '{decoded}'")

# Load model (let's use the old "buggy" one first since it had some learning)
model = Transformer(vocab_size=len(vocab), num_heads=16, d_model=512, d_ff=1344, rope_theta=10_000, context_length=256, num_layers=4)
iteration = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model, optimizer=None)
print(f"\nLoaded OLD checkpoint from iteration: {iteration}")
model.eval()

# Run forward pass
input_tensor = torch.tensor(prompt_tokens).unsqueeze(0)  # Add batch dimension
print(f"Input tensor shape: {input_tensor.shape}")

with torch.no_grad():
    output = model(input_tensor)
    logits = output[0, -1]  # Last token logits
    
    print(f"\nLogits stats: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
    
    # Get probabilities
    probs = softmax(logits, dim=-1)
    
    # Top-k predictions
    top_k = torch.topk(probs, k=20)
    print(f"\nTop 20 predictions:")
    for i, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices)):
        token_id = token_id.item()
        prob = prob.item()
        
        # Try to decode this single token
        try:
            if token_id < len(vocab):
                token_bytes = vocab[token_id]
                token_str = token_bytes.decode('utf-8', errors='replace')
            else:
                token_str = "UNKNOWN"
        except:
            token_str = "DECODE_ERROR"
            
        print(f"  {i+1:2d}. Token {token_id:4d}: prob={prob:.4f} | '{token_str}' | bytes={repr(vocab.get(token_id, b'UNKNOWN'))}")

# Let's also test some manual sampling
print(f"\n" + "="*60)
print("MANUAL SAMPLING TEST")
print("="*60)

temperatures = [0.1, 0.5, 1.0, 2.0]
for temp in temperatures:
    print(f"\nTemperature: {temp}")
    
    # Apply temperature
    temp_logits = logits / temp
    temp_probs = softmax(temp_logits, dim=-1)
    
    # Sample 5 times
    samples = []
    for _ in range(5):
        sample_id = torch.multinomial(temp_probs, 1).item()
        if sample_id < len(vocab):
            token_bytes = vocab[sample_id]
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except:
                token_str = "DECODE_ERROR"
        else:
            token_str = "UNKNOWN"
        samples.append(f"{sample_id}:'{token_str}'")
    
    print(f"  Samples: {', '.join(samples)}")

# Check the distribution of token types
print(f"\n" + "="*60)
print("TOKEN TYPE ANALYSIS")
print("="*60)

# Analyze what types of tokens have high probability
byte_tokens = 0
word_tokens = 0
special_tokens = 0

for token_id in range(min(1000, len(vocab))):  # Check first 1000 tokens
    prob = probs[token_id].item()
    if prob < 0.001:  # Only look at tokens with reasonable probability
        continue
        
    token_bytes = vocab[token_id]
    try:
        token_str = token_bytes.decode('utf-8', errors='replace')
        if len(token_str) == 1 and ord(token_str) < 256:
            byte_tokens += prob
        elif len(token_str) > 1:
            word_tokens += prob
        else:
            special_tokens += prob
    except:
        special_tokens += prob

print(f"Probability mass on byte tokens (single chars): {byte_tokens:.4f}")
print(f"Probability mass on word tokens (multi-char): {word_tokens:.4f}")
print(f"Probability mass on special/other tokens: {special_tokens:.4f}")

# Check if endoftext token has high probability
endoftext_id = None
for token_id, token_bytes in vocab.items():
    try:
        if token_bytes.decode('utf-8') == '<|endoftext|>':
            endoftext_id = token_id
            break
    except:
        pass

if endoftext_id is not None:
    endoftext_prob = probs[endoftext_id].item()
    print(f"<|endoftext|> token {endoftext_id} probability: {endoftext_prob:.6f}")