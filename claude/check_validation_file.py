#!/usr/bin/env python3

import numpy as np

def check_validation_files():
    """Check validation file token ranges"""
    
    files = [
        "../a1-data/ts-valid-encoded-tiktoken.npy",
        "../a1-data/ts-valid-encoded-new.npy"
    ]
    
    for filepath in files:
        print(f"\n=== {filepath} ===")
        
        try:
            data = np.load(filepath)
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Min: {data.min()}, Max: {data.max()}")
            
            over_10k = (data >= 10000).sum()
            print(f"Tokens >= 10000: {over_10k:,}")
            
            if over_10k > 0:
                print(f"❌ Invalid tokens found")
            else:
                print(f"✅ All tokens valid")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_validation_files()