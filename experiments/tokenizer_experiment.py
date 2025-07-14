import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import pickle
import pathlib
import numpy as np
from src.tokenizer import Tokenizer


def encode_command(args):
    bpe_file = pathlib.Path(args.bpe_file)
    input_file = pathlib.Path(args.input_file)

    with open(bpe_file, "rb") as f:
        res = pickle.load(f)

    vocab = res["vocab"]
    merges = res["merges"]

    tokenizer = Tokenizer(vocab, merges, special_tokens=[])
    with open(input_file) as f:
        ids = list(tokenizer.encode_iterable(f))
        print(f"Encoding output length: {len(ids)}")
        input_length = len(f.read())
        print(f"Input length: {input_length}")
        print(f"Byte per token: {input_length / len(ids)}")

    # Save token IDs to file if output_file is provided
    if args.output_file:
        output_file = pathlib.Path(args.output_file)
        ids_array = np.array(ids, dtype=np.uint16)
        np.save(output_file, ids_array)
        print(f"Token IDs saved to {output_file}.npy")


def main():
    parser = argparse.ArgumentParser(description="BPE tokenizer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    encode_parser = subparsers.add_parser("encode", help="Encode using trained BPE")
    encode_parser.add_argument(
        "--bpe-file",
        type=str,
        help="Trained bpe file with vocab and merges in pickle format.",
    )
    encode_parser.add_argument("--input-file", type=str, help="File to encode")
    encode_parser.add_argument(
        "--output-file",
        type=str,
        help="If provided, will write encode result (token IDs) into the file as numpy array as uint16",
    )
    encode_parser.set_defaults(func=encode_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()