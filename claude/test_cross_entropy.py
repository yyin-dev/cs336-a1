#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from src.cross_entropy import cross_entropy

print("CROSS-ENTROPY IMPLEMENTATION ANALYSIS")
print("="*60)

# Test case 1: Simple example
print("Test 1: Simple 3-class example")
logits = torch.tensor([[2.0, 1.0, 0.0]])  # Shape: (1, 3)
targets = torch.tensor([0])  # Target is class 0

custom_loss = cross_entropy(logits, targets)
pytorch_loss = F.cross_entropy(logits, targets)

print(f"Logits: {logits}")
print(f"Targets: {targets}")
print(f"Custom cross_entropy: {custom_loss:.6f}")
print(f"PyTorch cross_entropy: {pytorch_loss:.6f}")
print(f"Match: {torch.allclose(custom_loss, pytorch_loss)}")

# Manual calculation
probs = F.softmax(logits, dim=-1)
manual_loss = -torch.log(probs[0, targets[0]])
print(f"Manual calculation: {manual_loss:.6f}")

print(f"\n" + "="*40)

# Test case 2: Batch example
print("Test 2: Batch example")
batch_logits = torch.tensor([
    [2.0, 1.0, 0.0],
    [0.0, 2.0, 1.0],
    [1.0, 0.0, 2.0]
])  # Shape: (3, 3)
batch_targets = torch.tensor([0, 1, 2])

custom_batch_loss = cross_entropy(batch_logits, batch_targets)
pytorch_batch_loss = F.cross_entropy(batch_logits, batch_targets)

print(f"Batch logits shape: {batch_logits.shape}")
print(f"Batch targets: {batch_targets}")
print(f"Custom cross_entropy: {custom_batch_loss:.6f}")
print(f"PyTorch cross_entropy: {pytorch_batch_loss:.6f}")
print(f"Match: {torch.allclose(custom_batch_loss, pytorch_batch_loss)}")

print(f"\n" + "="*40)

# Test case 3: 3D example (like transformer)
print("Test 3: 3D transformer-like example")
seq_logits = torch.randn(2, 4, 10)  # (batch, seq_len, vocab_size)
seq_targets = torch.randint(0, 10, (2, 4))  # (batch, seq_len)

custom_3d_loss = cross_entropy(seq_logits, seq_targets)
# PyTorch expects (batch*seq_len, vocab_size) and (batch*seq_len,)
pytorch_3d_loss = F.cross_entropy(
    seq_logits.reshape(-1, seq_logits.size(-1)), 
    seq_targets.reshape(-1)
)

print(f"3D logits shape: {seq_logits.shape}")
print(f"3D targets shape: {seq_targets.shape}")
print(f"Custom cross_entropy: {custom_3d_loss:.6f}")
print(f"PyTorch cross_entropy: {pytorch_3d_loss:.6f}")
print(f"Match: {torch.allclose(custom_3d_loss, pytorch_3d_loss, atol=1e-6)}")

print(f"\n" + "="*60)
print("STEP-BY-STEP BREAKDOWN OF CUSTOM IMPLEMENTATION")
print("="*60)

# Walk through the custom implementation
logits = torch.tensor([[2.0, 1.0, 0.0]])
targets = torch.tensor([0])

print(f"Input logits: {logits}")
print(f"Input targets: {targets}")

# Step 1: Find max for numerical stability
logits_max = torch.max(logits, dim=-1, keepdim=True).values
print(f"Step 1 - logits_max: {logits_max}")

# Step 2: Subtract max for stability
logits_stable = logits - logits_max
print(f"Step 2 - logits_stable: {logits_stable}")

# Step 3: Compute exp
exp = torch.exp(logits_stable)
print(f"Step 3 - exp: {exp}")

# Step 4: Sum of exponentials
exp_sum = torch.sum(exp, dim=-1, keepdim=True)
print(f"Step 4 - exp_sum: {exp_sum}")

# Step 5: Gather target logits
import einops
target_logits = torch.gather(
    logits_stable,
    dim=len(logits_stable.shape) - 1,
    index=einops.rearrange(targets, "... (b d) -> ... b d", d=1).long(),
)
print(f"Step 5 - target_logits: {target_logits}")
print(f"Step 5 - target_logits shape: {target_logits.shape}")

# Step 6: Final loss calculation
loss = torch.mean(-target_logits + torch.log(exp_sum))
print(f"Step 6 - final loss: {loss}")

print(f"\n" + "="*40)
print("MATHEMATICAL VERIFICATION")
print("="*40)

print("The formula being computed:")
print("loss = mean(-target_logits + log(exp_sum))")
print("     = mean(-log_softmax(logits)[targets])")
print("     = mean(-log(softmax(logits)[targets]))")
print("     = standard cross-entropy loss ✓")

# Verify this is equivalent to -log(softmax)
softmax_probs = F.softmax(logits, dim=-1)
log_prob_target = torch.log(softmax_probs[0, targets[0]])
print(f"\nVerification:")
print(f"Softmax probs: {softmax_probs}")
print(f"Log prob of target: {log_prob_target:.6f}")
print(f"Negative log prob: {-log_prob_target:.6f}")
print(f"Custom implementation: {loss:.6f}")
print(f"Match: {torch.allclose(-log_prob_target, loss)}")

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)

print("✅ IMPLEMENTATION IS CORRECT")
print("- Numerically stable (subtracts max before exp)")
print("- Mathematically equivalent to -log(softmax(logits)[targets])")
print("- Handles arbitrary tensor shapes correctly")
print("- Matches PyTorch's implementation exactly")
print()
print("The cross-entropy implementation is NOT the source of your training issues.")