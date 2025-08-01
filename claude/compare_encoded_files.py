#!/usr/bin/env python3

import numpy as np

def compare_encoded_files():
    """Compare old vs new encoded files"""
    
    files = [
        ("../a1-data/ts-train-encoded-tiktoken.npy", "Old TikToken file"),
        ("../a1-data/ts-train-encoded.npy", "New BPE encoded file")
    ]
    
    for filepath, description in files:
        print(f"\n=== {description} ===")
        
        data = np.load(filepath)
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Min: {data.min()}, Max: {data.max()}")
        print(f"First 20 tokens: {data[:20]}")
        
        # Check token ranges
        over_10k = (data >= 10000).sum()
        print(f"Tokens >= 10000: {over_10k:,} / {data.size:,} ({over_10k/data.size*100:.4f}%)")
        
        if over_10k > 0:
            print(f"❌ Contains tokens outside vocab_size=10000")
            high_tokens = data[data >= 10000][:10]  # First 10 invalid tokens
            print(f"Sample invalid tokens: {high_tokens}")
        else:
            print(f"✅ All tokens valid for vocab_size=10000")

if __name__ == "__main__":
    compare_encoded_files()