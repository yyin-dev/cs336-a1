"""
uv run experiments/generate_text.py \
--bpe ../a1-log/ts-train-bpe.pkl \
--model-checkpoint ../a1-checkpoints/lr_max_1e-2_checkpoint_iter18999.pt \
--prompt "Once upon a time, there was a pretty girl named Lily."
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import pickle

from src.decode import decode
from src.transformer import Transformer
from src.checkpointing import load_checkpoint
from src.logger import logging
import torch
import numpy as np
import random


# Seed for deterministic training
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bpe", type=str, required=True)
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-generated-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p-sampling-threshold", type=float, default=0.9)

    args = parser.parse_args()

    with open(args.bpe, "rb") as bpe_file:
        res = pickle.load(bpe_file)

    vocab: dict[int, bytes] = res["vocab"]
    merges: list[tuple[bytes, bytes]] = res["merges"]

    device = "cpu"
    if torch.cuda.is_available():
        logging.info("cuda available")
        device = "cuda"
    elif torch.backends.mps.is_available():
        logging.info("mps available")
        device = "mps"

    model = Transformer(
        vocab_size=len(vocab),
        num_heads=16,
        d_model=512,
        d_ff=1344,
        rope_theta=10_000,
        context_length=256,
        num_layers=4,
    )
    model.to(device)
    model.eval()

    iteration = load_checkpoint(args.model_checkpoint, model, optimizer=None)
    print(f"Loaded checkpoint from iteration: {iteration}")

    res = decode(
        model=model,
        prompt=args.prompt,
        vocab=vocab,
        merges=merges,
        max_generated_tokens=args.max_generated_tokens,
        temperature=args.temperature,
        top_p_sampling_threshold=args.top_p_sampling_threshold,
    )

    print(f"Generated:<{res}>")


if __name__ == "__main__":
    main()
