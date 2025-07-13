import pathlib
import os
import sys
import json

# Run this with so that relative import works without path hack
# $ uv run -m profiling.tokenizer

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent).parent / "tests" / "fixtures"

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
from src.tokenizer import Tokenizer


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)


def main():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        contents = f.read()
        _ = tokenizer.encode(contents)


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()", sort="tottime")
