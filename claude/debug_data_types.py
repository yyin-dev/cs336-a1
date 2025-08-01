#!/usr/bin/env python3

import numpy as np

def check_data_types():
    """Check data types and ranges of encoded training data"""
    
    files_to_check = [
        "../a1-data/ts-train-encoded-tiktoken.npy",
        "../a1-data/ts-valid-encoded-tiktoken.npy"
    ]
    
    for filepath in files_to_check:
        print(f"\n=== {filepath} ===")
        
        # Load with numpy.load to get proper dtype
        data = np.load(filepath)
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Min value: {data.min()}")
        print(f"Max value: {data.max()}")
        print(f"Unique values count: {len(np.unique(data))}")
        
        # Check first 20 values
        print(f"First 20 values: {data[:20]}")
        
        # Check if values exceed 255
        above_255 = np.sum(data > 255)
        total = data.size
        print(f"Values > 255: {above_255:,} / {total:,} ({above_255/total*100:.4f}%)")
        
        # Now load as memmap to see what the training loop sees
        memmap_data = np.memmap(filepath, mode="r")
        print(f"\nMemmap shape: {memmap_data.shape}")
        print(f"Memmap dtype: {memmap_data.dtype}")
        print(f"Memmap first 20 values: {memmap_data[:20]}")
        
        if not np.array_equal(data[:20], memmap_data[:20]):
            print("⚠️  WARNING: numpy.load and memmap give different results!")
            print(f"Load first 20: {data[:20]}")
            print(f"Memmap first 20: {memmap_data[:20]}")

if __name__ == "__main__":
    check_data_types()