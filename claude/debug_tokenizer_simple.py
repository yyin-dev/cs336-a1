#!/usr/bin/env python3

import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import pickle

# Load the BPE tokenizer
with open('../a1-log/ts-bpe.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)

print("Tokenizer pickle contents:")
print(f"Type: {type(tokenizer_data)}")
if isinstance(tokenizer_data, dict):
    print(f"Keys: {list(tokenizer_data.keys())}")
    
    if 'id_to_token' in tokenizer_data:
        id_to_token = tokenizer_data['id_to_token']
        print(f"Vocab size: {len(id_to_token)}")
        
        # Check our test tokens
        test_tokens = [430, 439, 259, 398]
        print(f"\nTest tokens {test_tokens} decode to:")
        for token_id in test_tokens:
            if token_id < len(id_to_token):
                token_str = id_to_token[token_id]
                print(f"  Token {token_id}: '{token_str}' (repr: {repr(token_str)})")
            else:
                print(f"  Token {token_id}: UNKNOWN (vocab size: {len(id_to_token)})")
        
        # Check low token IDs
        print(f"\nLow token IDs (model favorites):")
        for token_id in [0, 1, 2, 3, 4, 5, 10, 24, 34, 46]:
            if token_id < len(id_to_token):
                token_str = id_to_token[token_id]
                print(f"  Token {token_id}: '{token_str}' (repr: {repr(token_str)})")
            else:
                print(f"  Token {token_id}: UNKNOWN")
        
        # Check higher token IDs
        print(f"\nHigher token IDs (actual words):")
        for token_id in [100, 200, 500, 1000, 2000, 5000]:
            if token_id < len(id_to_token):
                token_str = id_to_token[token_id]
                print(f"  Token {token_id}: '{token_str}' (repr: {repr(token_str)})")
            else:
                print(f"  Token {token_id}: UNKNOWN")
    
    if 'token_to_id' in tokenizer_data:
        token_to_id = tokenizer_data['token_to_id']
        
        # Check for common words
        common_words = ['once', 'upon', 'time', 'the', 'a', 'was', 'there', 'little', 'girl']
        print(f"\nCommon words in vocab:")
        for word in common_words:
            if word in token_to_id:
                print(f"  '{word}': token {token_to_id[word]}")
            else:
                print(f"  '{word}': NOT FOUND")

else:
    print("Tokenizer is not a dict, checking methods...")
    print(dir(tokenizer_data))