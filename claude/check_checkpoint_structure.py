#!/usr/bin/env python3

import torch
import os

def check_checkpoint_structure():
    """Check the structure of available checkpoints"""
    
    checkpoint_dir = "../a1-checkpoints/"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print("=== AVAILABLE CHECKPOINTS ===")
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.endswith('.pt'):
            print(f"\n📁 {f}")
            try:
                checkpoint_path = os.path.join(checkpoint_dir, f)
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                print(f"  Keys: {list(checkpoint.keys())}")
                
                # Show some details about each key
                for key, value in checkpoint.items():
                    if isinstance(value, dict):
                        print(f"    {key}: dict with {len(value)} items")
                    elif isinstance(value, torch.Tensor):
                        print(f"    {key}: tensor {value.shape}")
                    else:
                        print(f"    {key}: {type(value)} = {value}")
                        
            except Exception as e:
                print(f"  ❌ Error loading: {e}")

if __name__ == "__main__":
    check_checkpoint_structure()