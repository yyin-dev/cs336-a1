import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import pickle
import pathlib
import numpy as np
from multiprocessing import Pool
import src.logger as logger
from src.tokenizer import Tokenizer
from src.pretokenization_example import find_chunk_boundaries


def encode_chunk(args):
    (input_file, start, end, vocab, merges) = args
    logger.info(f"Start encoding: {start, end}")
    tokenizer = Tokenizer(vocab, merges, special_tokens=[])

    with open(input_file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        ids = tokenizer.encode(chunk.decode("utf-8"))

    logger.info(f"Done encoding: {start, end}")
    return ids


def encode_command(args):
    bpe_file = pathlib.Path(args.bpe_file)
    input_file = pathlib.Path(args.input_file)

    with open(bpe_file, "rb") as f:
        res = pickle.load(f)

    vocab = res["vocab"]
    merges = res["merges"]

    if args.parallel:
        num_chunks = 128
        num_processes = 8

        with open(input_file, "rb") as input_f:
            boundaries = find_chunk_boundaries(
                input_f, num_chunks, "<|endoftext|>".encode("utf-8")
            )

        pool = Pool(num_processes)
        args_list = [
            (input_file, start, end, vocab, merges)
            for (start, end) in zip(boundaries[:-1], boundaries[1:])
        ]
        chunk_results = pool.map(encode_chunk, args_list)

        ids = []
        for vs in chunk_results:
            ids.extend(vs)
    else:
        tokenizer = Tokenizer(vocab, merges, special_tokens=[])
        with open(input_file) as f:
            # ids = list(tokenizer.encode_iterable(f))
            ids = tokenizer.encode(f.read())


    input_length = os.path.getsize(input_file)
    print(f"Encoding output length: {len(ids)}")
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
    encode_parser.add_argument(
        "--parallel", action="store_true", help="Run in parallel"
    )

    encode_parser.set_defaults(func=encode_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
