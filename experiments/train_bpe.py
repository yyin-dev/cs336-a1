import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from src.bpe import train_bpe

import sys
import pathlib
import pickle
import cProfile

# Run this with
# $ uv run -m expriments.train_bpe


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("read FILE")
        print("train INPUT_FILE OUTPUT_FILE VOCAB_SIZE")
        return

    operation = str.lower(sys.argv[1])

    if operation == "read":
        path = pathlib.Path(sys.argv[2])
        print(f"Reading from {path}")
        with open(path, "rb") as f:
            res = pickle.load(f)

        print("vocab: ")
        print(res["vocab"])
        print("merges")
        print(res["merges"])

        vocab = res["vocab"]
        longest_vocab = vocab[0]
        for v in vocab.values():
            if len(v) > len(longest_vocab):
                longest_vocab = v

        print(f"Longest vocab: {longest_vocab}. Length: {len(longest_vocab)}")
    elif operation == "train":
        path = pathlib.Path(sys.argv[2])
        output = pathlib.Path(sys.argv[3])
        vocab_size = int(sys.argv[4])
        print(f"Training BPE using {path}")
        vocab, merges = train_bpe(path, vocab_size, special_tokens=["<|endoftext|>"])

        with open(output, "wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges}, f)

        print("Written to file successfully")
    else:
        print("Unknown operation")


if __name__ == "__main__":
    main()
    # cProfile.run("main()", sort="tottime")
