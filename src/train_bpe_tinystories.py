from .bpe import train_bpe
import sys
import pathlib
import json

# Run this with
# $ uv run -m src.train_bpe_tinystories


def main():
    if len(sys.argv) < 2:
        print("No path provided")
        return

    path = pathlib.Path(sys.argv[1])
    print(f"Training BPE using {path}")
    vocab, merges = train_bpe(path, 10000, special_tokens=["<|endoftext|>"])
    merges = list(map(lambda s: (s[0].decode("utf-8"), s[1].decode("utf-8")), merges))

    with open("tinystories_bpe_result.json", "w") as f:
        json.dump({"vocab": vocab, "merges": merges}, f)


if __name__ == "__main__":
    main()
