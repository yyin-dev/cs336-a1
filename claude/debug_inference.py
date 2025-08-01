#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import pickle
import torch
import torch.nn.functional as F
from src.transformer import Transformer
from src.checkpointing import load_checkpoint
from src.tokenizer import Tokenizer
from einops import rearrange
import numpy as np

# Load BPE data
with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    res = pickle.load(f)

vocab = res["vocab"]
merges = res["merges"]

# Initialize model
model = Transformer(
    vocab_size=len(vocab),
    num_heads=16,
    d_model=512,
    d_ff=1344,
    rope_theta=10_000,
    context_length=256,
    num_layers=4,
)

# Load checkpoint
iteration = load_checkpoint('../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt', model, optimizer=None)
print(f"Loaded checkpoint from iteration: {iteration}")

model.eval()

# Test with simple prompt
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
prompt = "Once upon a time"
input_tokens = tokenizer.encode(prompt)
print(f"Input tokens: {input_tokens}")
print(f"Decoded input: '{tokenizer.decode(input_tokens)}'")

# Add batch dimension
input_tensor = torch.tensor(input_tokens).unsqueeze(0)
print(f"Input tensor shape: {input_tensor.shape}")

# Get model output
with torch.no_grad():
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # Get logits for the last token
    logits = output[0, -1]  # (vocab_size,)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits min/max: {logits.min().item():.3f} / {logits.max().item():.3f}")
    
    # Check if logits are reasonable
    probs_temp1 = F.softmax(logits, dim=-1)
    top5_temp1 = torch.topk(probs_temp1, 5)
    print(f"\nTemperature 1.0 - Top 5 tokens and their probabilities:")
    for i in range(5):
        token_id = int(top5_temp1.indices[i])
        prob = top5_temp1.values[i].item()
        try:
            token_text = tokenizer.decode([token_id])
            print(f"  Token {token_id}: '{token_text}' (prob: {prob:.4f})")
        except:
            print(f"  Token {token_id}: [decode error] (prob: {prob:.4f})")
    
    # Test different temperatures
    for temp in [0.1, 0.5, 1.0, 2.0]:
        logits_scaled = logits / temp
        probs = F.softmax(logits_scaled, dim=-1)
        
        # Sample a token
        sampled_id = int(torch.multinomial(probs, 1).item())
        try:
            sampled_text = tokenizer.decode([sampled_id])
            print(f"Temperature {temp}: sampled token {sampled_id} = '{sampled_text}'")
        except:
            print(f"Temperature {temp}: sampled token {sampled_id} = [decode error]")

# Generate a few tokens and show the token IDs
print(f"\nGenerating 10 tokens with temperature 1.0:")
generated_ids = []
current_input = input_tensor

for i in range(10):
    with torch.no_grad():
        output = model(current_input)
        logits = output[0, -1]
        probs = F.softmax(logits, dim=-1)
        
        sampled_id = int(torch.multinomial(probs, 1).item())
        generated_ids.append(sampled_id)
        
        # Append to input for next iteration
        next_token = torch.tensor([[sampled_id]])
        current_input = torch.cat([current_input, next_token], dim=1)
        
        try:
            token_text = tokenizer.decode([sampled_id])
            print(f"  Step {i+1}: token {sampled_id} = '{token_text}'")
        except:
            print(f"  Step {i+1}: token {sampled_id} = [decode error]")

print(f"\nAll generated token IDs: {generated_ids}")
try:
    all_text = tokenizer.decode(generated_ids)
    print(f"Full generated text: '{all_text}'")
except:
    print(f"Full generated text: [decode error]")