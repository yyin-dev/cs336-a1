#!/usr/bin/env python3

import numpy as np
import pickle

def check_encoding_consistency():
    """Check if encoding matches BPE vocab"""
    
    # Load BPE vocab
    with open("../a1-log/ts-bpe.pkl", "rb") as f:
        bpe_data = pickle.load(f)
    
    vocab = bpe_data["vocab"]
    merges = bpe_data["merges"]
    
    print(f"BPE vocab size: {len(vocab)}")
    print(f"BPE merges count: {len(merges)}")
    print(f"Max token ID in vocab: {max(vocab.keys())}")
    print(f"Min token ID in vocab: {min(vocab.keys())}")
    
    # Check some high token IDs
    print(f"\nSample high token IDs from vocab:")
    high_tokens = sorted([k for k in vocab.keys() if k > 9000])[-10:]
    for token_id in high_tokens:
        token_bytes = vocab[token_id]
        try:
            token_str = token_bytes.decode('utf-8')
        except:
            token_str = f"<bytes:{token_bytes}>"
        print(f"Token {token_id}: {repr(token_str)}")
    
    # Load newly encoded data
    print(f"\n=== NEWLY ENCODED DATA ===")
    encoded_data = np.load("../a1-data/ts-train-encoded.npy")
    print(f"Encoded data shape: {encoded_data.shape}")
    print(f"Encoded data dtype: {encoded_data.dtype}")
    print(f"Min token: {encoded_data.min()}")
    print(f"Max token: {encoded_data.max()}")
    
    # Check if max token exceeds vocab
    max_token = encoded_data.max()
    if max_token >= len(vocab):
        print(f"❌ ERROR: Max token {max_token} >= vocab size {len(vocab)}")
        
        # Find problematic tokens
        invalid_tokens = encoded_data[encoded_data >= len(vocab)]
        print(f"Invalid tokens count: {len(invalid_tokens)}")
        print(f"Invalid tokens: {np.unique(invalid_tokens)}")
    else:
        print(f"✅ All tokens fit within vocab size")
    
    # Check token distribution
    unique_tokens = np.unique(encoded_data)
    print(f"Unique tokens in encoded data: {len(unique_tokens)}")
    print(f"Token range: {unique_tokens.min()} to {unique_tokens.max()}")

if __name__ == "__main__":
    check_encoding_consistency()