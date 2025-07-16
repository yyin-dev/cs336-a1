import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from src.bpe import train_bpe

import argparse
import pathlib
import pickle
import cProfile

# Run this with
# $ uv run -m experiments.train_bpe


def command_read(args):
    """Read and display BPE data from pickle file."""
    path = pathlib.Path(args.file)
    print(f"Reading from {path}")

    with open(path, "rb") as f:
        res = pickle.load(f)

    print("vocab: ")
    print(res["vocab"])
    print("merges")
    print(res["merges"])

    vocab = res["vocab"]
    longest_vocab = vocab[0] if vocab else b""
    for v in vocab.values():
        if len(v) > len(longest_vocab):
            longest_vocab = v

    print(f"Longest vocab: {longest_vocab}. Length: {len(longest_vocab)}")


def command_train(args):
    """Train BPE and save to pickle file."""
    path = pathlib.Path(args.input_file)
    output = pathlib.Path(args.output_file)
    vocab_size = args.vocab_size
    special_tokens = args.special_tokens or ["<|endoftext|>"]

    print(f"Training BPE using {path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    vocab, merges = train_bpe(path, vocab_size, special_tokens=special_tokens)

    with open(output, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    print(f"Written to {output} successfully")


def main():
    parser = argparse.ArgumentParser(
        description="BPE Training and Reading Tool",
        prog="python -m experiments.train_bpe",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Read command
    read_parser = subparsers.add_parser(
        "read", help="Read and display BPE data from pickle file"
    )
    read_parser.add_argument("file", type=str, help="Path to pickle file to read")
    read_parser.set_defaults(func=command_read)

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train BPE and save to pickle file"
    )
    train_parser.add_argument("--input-file", type=str, help="Input corpus file path")
    train_parser.add_argument("--output-file", type=str, help="Output pickle file path")
    train_parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    train_parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help='Special tokens (default: ["<|endoftext|>"])',
    )
    train_parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with cProfile for performance analysis",
    )
    train_parser.set_defaults(func=command_train)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if hasattr(args, "profile") and args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        args.func(args)
        profiler.disable()
        profiler.print_stats(sort="tottime")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
