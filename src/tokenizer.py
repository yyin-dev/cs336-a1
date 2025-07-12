from typing import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Given a vocabulary, a list of merges, and a list of special tokens,
        return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.
        """
        pass

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merge_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        pass

    def encode(self, text: str) -> list[int]:
        return []

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return iter([1])

    def decode(self, ids: list[int]) -> str:
        return ""
