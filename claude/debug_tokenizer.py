#!/usr/bin/env python3

import pickle
import sys
import os
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from src.tokenizer import Tokenizer
import tiktoken

# Load BPE data
with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    res = pickle.load(f)

vocab = res["vocab"]
merges = res["merges"]

# Test string
test_prompt = "Once upon a time"

print(f"Vocab size: {len(vocab)}")
print(f"Merges count: {len(merges)}")

# Test with custom tokenizer
custom_tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
custom_tokens = custom_tokenizer.encode(test_prompt)
print(f"Custom tokenizer tokens: {custom_tokens}")
print(f"Custom decode: '{custom_tokenizer.decode(custom_tokens)}'")

# Test with tiktoken setup (same as decode.py)
mergeable_ranks = {pair: idx for (idx, pair) in vocab.items()}
num_special_tokens = len(vocab) - 256 - len(merges)
print(f"Number of special tokens: {num_special_tokens}")

special_tokens = []
for i in range(256, 256 + num_special_tokens):
    special_tokens.append(vocab[i])
print(f"Special tokens: {special_tokens}")

special_tokens_dict = {
    token.decode("utf-8"): idx + 256
    for (idx, token) in enumerate(special_tokens)
}

allowed_special = set(list(map(lambda b: b.decode("utf-8"), special_tokens)))

tiktoken_enc = tiktoken.Encoding(
    name="my_encoding",
    pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens_dict,
)

tiktoken_tokens = tiktoken_enc.encode(test_prompt, allowed_special=allowed_special)
print(f"TikToken tokens: {tiktoken_tokens}")
print(f"TikToken decode: '{tiktoken_enc.decode(tiktoken_tokens)}'")

print(f"Tokens match: {custom_tokens == tiktoken_tokens}")

# Test some random token IDs to see decoding behavior
test_token_ids = [100, 200, 300, 400, 500]
print(f"\nTesting decoding of random token IDs: {test_token_ids}")
try:
    print(f"Custom decode: '{custom_tokenizer.decode(test_token_ids)}'")
except Exception as e:
    print(f"Custom decode error: {e}")

try:
    print(f"TikToken decode: '{tiktoken_enc.decode(test_token_ids)}'")
except Exception as e:
    print(f"TikToken decode error: {e}")