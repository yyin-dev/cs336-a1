#!/usr/bin/env python3

import numpy as np

def check_encoded_files():
    """Check different encoded data files"""
    
    files = [
        ("../a1-data/ts-train-encoded-tiktoken.npy", "TikToken encoded"),
        ("../a1-data/ts-train-encoded-new.npy", "Custom BPE encoded")
    ]
    
    for filepath, description in files:
        print(f"\n=== {description} ===")
        print(f"File: {filepath}")
        
        try:
            # Check with correct dtype
            data = np.memmap(filepath, mode="r", dtype=np.uint16)
            print(f"Shape: {data.shape}")
            print(f"Min: {data.min()}, Max: {data.max()}")
            print(f"First 10 tokens: {data[:10]}")
            
            # Check vocab range
            unique_count = len(np.unique(data[:100000]))  # Sample for speed
            print(f"Unique tokens (first 100k): {unique_count}")
            
            # Check if values fit in vocab_size=10000
            over_vocab = np.sum(data > 9999)
            print(f"Tokens > 9999: {over_vocab:,} / {data.size:,}")
            
            if over_vocab == 0:
                print("✅ All tokens fit in vocab_size=10000")
            else:
                print(f"❌ {over_vocab:,} tokens exceed vocab_size=10000")
                
        except Exception as e:
            print(f"Error loading: {e}")

if __name__ == "__main__":
    check_encoded_files()