# Experiments with tiktoken
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import tiktoken

import argparse
import pathlib
import pickle
import sys
import numpy as np
from src.tokenizer import Tokenizer
from src.pretokenization_example import find_chunk_boundaries
import src.logger as logger


def command_encode(args):
    """Read and display BPE data from pickle file."""
    path = pathlib.Path(args.bpe_file)
    print(f"Reading from {path}")

    with open(path, "rb") as f:
        res = pickle.load(f)

    vocab: dict[int, bytes] = res["vocab"]
    merges: list[tuple[bytes, bytes]] = res["merges"]

    # Find [mergeable_ranks] and [special_tokens]
    # vocab = [ 256 individual bytes; speical tokens; merges ]
    mergeable_ranks: dict[bytes, int] = {pair: idx for (idx, pair) in vocab.items()}
    num_special_tokens = len(vocab) - 256 - len(merges)
    print(f"Number of special tokens: {num_special_tokens}. ")
    special_tokens = []
    for i in range(256, 256 + num_special_tokens):
        special_tokens.append(vocab[i])
    print(f"Special tokens: {special_tokens}")

    special_tokens_dict = {
        token.decode("utf-8"): idx + 256 for (idx, token) in enumerate(special_tokens)
    }

    tiktoken_enc = tiktoken.Encoding(
        name="my_encoding",
        pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_dict,
    )

    my_tokenizer = Tokenizer(vocab, merges, special_tokens)

    allowed_special = set(list(map(lambda b: b.decode("utf-8"), special_tokens)))
    run_test = True
    if run_test:
        tests = [
            "hi",
            "hi<|endoftext|>",
            "<|endoftext|>",
            "hello <|endoftext|> fdfd ab ab",
            "hello <|endoftext|> fdfd ab ab <|endoftext|>",
            "hello <|endoftext|> fdfd ab ab <|endoftext|>\ndfd<|endoftext|>\n\ndbadfd",
        ]

        for test in tests:
            res1 = tiktoken_enc.encode(test, allowed_special=allowed_special)
            res2 = my_tokenizer.encode(test)
            assert res1 == res2

        print("All tests passed")

    if os.path.exists(args.output_file):
        os.remove(args.output_file)
        print(f"File {args.output_file} deleted")

    num_chunks = 64
    with open(args.input_file, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, "<|endoftext|>".encode("utf-8")
        )

    # Instead of keeping np.append after every chunk, keep np arrays in a list 
    # and do np.concat at the end.
    # np.append is O(m+n) and copies both arrays into a new memory location.
    np_arrays: list[np.ndarray] = []
    with open(args.input_file, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            logger.info(f"{start, end}")
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")

            tiktoken_enc = tiktoken.Encoding(
                name="my_encoding",
                pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens_dict,
            )

            ids = tiktoken_enc.encode(chunk, allowed_special=allowed_special)
            np_arrays.append(np.array(ids, dtype=np.uint16))

    res = np.concat(np_arrays)
    input_length = os.path.getsize(args.input_file)
    print(f"Encoding output length: {len(res)}")
    print(f"Input length: {input_length}")
    print(f"Byte per token: {input_length / len(res)}")

    # Save token IDs to file if output_file is provided
    if args.output_file:
        np.save(args.output_file, res)
        logger.info(f"Token IDs saved to {args.output_file}")


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
    encode_parser.add_argument(
        "--parallel", action="store_true", help="Run in parallel"
    )

    encode_parser.set_defaults(func=command_encode)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
