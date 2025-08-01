#!/usr/bin/env python3

import numpy as np
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from src.data_loading import get_batch

def test_data_loading():
    """Test that data loading now works correctly with uint16"""
    
    # Test with corrected loading
    print("=== TESTING CORRECTED DATA LOADING ===")
    
    train_filename = "../a1-data/ts-train-encoded-tiktoken.npy"
    
    # Load correctly as uint16
    train_dataset = np.memmap(train_filename, mode="r", dtype=np.uint16)
    print(f"Dataset shape: {train_dataset.shape}")
    print(f"Dataset dtype: {train_dataset.dtype}")
    print(f"Min value: {train_dataset.min()}")
    print(f"Max value: {train_dataset.max()}")
    print(f"First 20 tokens: {train_dataset[:20]}")
    
    # Test batch loading
    batch_size = 4
    context_length = 10
    device = "cpu"
    
    inputs, targets = get_batch(train_dataset, batch_size, context_length, device)
    print(f"\nBatch inputs shape: {inputs.shape}")
    print(f"Batch targets shape: {targets.shape}")
    # Convert to int32 for operations
    inputs_int = inputs.int()
    targets_int = targets.int()
    
    print(f"Input tokens range: {inputs_int.min().item()} to {inputs_int.max().item()}")
    print(f"Target tokens range: {targets_int.min().item()} to {targets_int.max().item()}")
    
    # Verify we see actual BPE tokens (> 255)
    high_tokens_input = (inputs_int > 255).sum().item()
    high_tokens_target = (targets_int > 255).sum().item()
    total_tokens = inputs.numel() + targets.numel()
    
    print(f"\nTokens > 255 in batch: {high_tokens_input + high_tokens_target} / {total_tokens}")
    print(f"Percentage of BPE tokens: {(high_tokens_input + high_tokens_target) / total_tokens * 100:.1f}%")
    
    if high_tokens_input + high_tokens_target > 0:
        print("✅ SUCCESS: Found BPE tokens (> 255) in training data!")
    else:
        print("❌ FAILED: Still only seeing byte tokens (0-255)")

if __name__ == "__main__":
    test_data_loading()