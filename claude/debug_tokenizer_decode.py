#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import pickle
from src.tokenizer import Tokenizer

# Load the BPE tokenizer
with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("Tokenizer analysis:")
print(f"Vocab size: {len(tokenizer.get_vocab())}")

# Test the prompt tokenization
prompt = "Once upon a time"
tokens = tokenizer.encode(prompt)
print(f"\nPrompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Decode each token individually
print(f"\nToken breakdown:")
for i, token_id in enumerate(tokens):
    if token_id < len(tokenizer.id_to_token):
        token_str = tokenizer.id_to_token[token_id]
        print(f"  Token {token_id}: '{token_str}'")
    else:
        print(f"  Token {token_id}: UNKNOWN")

# Check what our test tokens decode to
test_tokens = [430, 439, 259, 398]
print(f"\nTest tokens {test_tokens} decode to:")
for token_id in test_tokens:
    if token_id < len(tokenizer.id_to_token):
        token_str = tokenizer.id_to_token[token_id]
        print(f"  Token {token_id}: '{token_str}'")
    else:
        print(f"  Token {token_id}: UNKNOWN")

# Decode the full sequence
decoded = tokenizer.decode(test_tokens)
print(f"\nFull decode of {test_tokens}: '{decoded}'")

# Check some low token IDs that the model likes to predict
print(f"\nLow token IDs (model favorites):")
for token_id in [0, 1, 2, 3, 4, 5, 10, 24, 34, 46]:
    if token_id < len(tokenizer.id_to_token):
        token_str = tokenizer.id_to_token[token_id]
        print(f"  Token {token_id}: '{token_str}' (repr: {repr(token_str)})")
    else:
        print(f"  Token {token_id}: UNKNOWN")

# Check higher token IDs that should represent actual words
print(f"\nHigher token IDs (actual words):")
for token_id in [100, 200, 500, 1000, 2000, 5000]:
    if token_id < len(tokenizer.id_to_token):
        token_str = tokenizer.id_to_token[token_id]
        print(f"  Token {token_id}: '{token_str}' (repr: {repr(token_str)})")
    else:
        print(f"  Token {token_id}: UNKNOWN")

# Test encoding a simple story
story = "Once upon a time there was a little girl."
story_tokens = tokenizer.encode(story)
print(f"\nStory: '{story}'")
print(f"Story tokens: {story_tokens}")
print(f"Story decoded: '{tokenizer.decode(story_tokens)}'")