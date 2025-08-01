#!/usr/bin/env python3

import numpy as np
from collections import Counter
import pickle

def analyze_training_data():
    """Analyze token distribution in the actual training data"""
    
    # Load training data
    train_data = np.memmap("../a1-data/ts-train-encoded-tiktoken.npy", mode="r")
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data dtype: {train_data.dtype}")
    
    # Get token counts
    token_counts = Counter(train_data[:1000000])  # Sample first 1M tokens for speed
    print(f"\nAnalyzed {len(token_counts)} unique tokens")
    
    # Load vocab to get token strings
    with open("../a1-log/ts-bpe.pkl", "rb") as f:
        bpe_data = pickle.load(f)
    
    vocab = bpe_data["vocab"]  # Maps token_id -> bytes
    print(f"Vocab size: {len(vocab)}")
    
    # Print top tokens by frequency
    print("\n=== TOP 20 MOST FREQUENT TOKENS ===")
    for i, (token_id, count) in enumerate(token_counts.most_common(20)):
        if token_id in vocab:
            token_bytes = vocab[token_id]
            try:
                token_str = token_bytes.decode("utf-8")
            except:
                token_str = f"<bytes:{token_bytes}>"
        else:
            token_str = f"<unknown:{token_id}>"
        
        # Show readable representation
        token_repr = repr(token_str)
        freq_pct = count / 1000000 * 100
        print(f"{i+1:2d}. Token {token_id:4d}: {token_repr:20s} ({count:6d} = {freq_pct:.2f}%)")
    
    # Analyze token ID ranges
    print("\n=== TOKEN ID RANGE ANALYSIS ===")
    ranges = [
        (0, 255, "Byte tokens (0-255)"),
        (256, 999, "Low BPE tokens (256-999)"), 
        (1000, 4999, "Mid BPE tokens (1000-4999)"),
        (5000, 9999, "High BPE tokens (5000-9999)")
    ]
    
    total_tokens = sum(token_counts.values())
    for start, end, label in ranges:
        range_count = sum(count for token_id, count in token_counts.items() 
                         if start <= token_id <= end)
        range_pct = range_count / total_tokens * 100
        print(f"{label}: {range_count:8d} tokens ({range_pct:.2f}%)")
    
    # Check for problematic patterns
    print("\n=== CHECKING FOR BIAS PATTERNS ===")
    
    # Are low token IDs over-represented?
    low_tokens = sum(count for token_id, count in token_counts.items() if token_id < 256)
    low_pct = low_tokens / total_tokens * 100
    print(f"Byte tokens (0-255): {low_pct:.1f}%")
    
    high_tokens = sum(count for token_id, count in token_counts.items() if token_id >= 256)
    high_pct = high_tokens / total_tokens * 100
    print(f"BPE tokens (256+): {high_pct:.1f}%")
    
    # Check specific problematic tokens from debugging
    problem_tokens = [0, 24, 34, 181]  # From previous analysis
    print(f"\nProblem tokens from model predictions:")
    for token_id in problem_tokens:
        count = token_counts.get(token_id, 0)
        freq_pct = count / total_tokens * 100 if count > 0 else 0
        if token_id in vocab:
            token_bytes = vocab[token_id]
            try:
                token_str = token_bytes.decode("utf-8")
            except:
                token_str = f"<bytes:{token_bytes}>"
        else:
            token_str = f"<unknown:{token_id}>"
        print(f"Token {token_id}: {repr(token_str)} = {count} occurrences ({freq_pct:.4f}%)")
    
    # Check common English tokens by searching vocab
    common_words = [" the", " and", " a", " to", " of", " in", " that", " was"]
    print(f"\nCommon English tokens:")
    # Create reverse vocab mapping
    reverse_vocab = {v: k for k, v in vocab.items()}
    for word in common_words:
        word_bytes = word.encode("utf-8")
        if word_bytes in reverse_vocab:
            token_id = reverse_vocab[word_bytes]
            count = token_counts.get(token_id, 0)
            freq_pct = count / total_tokens * 100 if count > 0 else 0
            print(f"Token {token_id}: '{word}' = {count} occurrences ({freq_pct:.4f}%)")
        else:
            print(f"'{word}' not found in vocab")

if __name__ == "__main__":
    analyze_training_data()