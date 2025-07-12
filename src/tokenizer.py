import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(s, special_tokens: list[str]) -> list[list[bytes]]:
    # Different from training, at encoding we should include special tokens
    # in the pretokenization result.
    #
    # Special tokens are not broken down into individual bytes. The vocabulary
    # includes special tokens, so will be encoded as is.
    speical_tokens_set = set(special_tokens)

    # re.split is greedy when finding matches in patterns. For example,
    # re.split("abbc", "(b)|(bb)") returns ["a", "b", "b", "c"].
    # To prioritize longest match, put "bb" before "b" in the regex pattern.
    special_tokens.sort(key=len, reverse=True)
    escaped_special_tokens = list(map(re.escape, special_tokens))

    # Wrap special token patterns in parenthesis so that they are returned
    # by `re.split`
    escaped_and_wrapped_in_parens = list(
        map(lambda s: f"({s})", escaped_special_tokens)
    )
    special_tokens_regex = "|".join(escaped_and_wrapped_in_parens)

    res: list[list[bytes]] = []

    # Edge case: there's no special tokens, so regex becomes "".
    if special_tokens_regex == "":
        parts = [s]
    else:
        parts = re.split(special_tokens_regex, s)

    for part in parts:
        if part is None:
            continue

        if part in speical_tokens_set:
            res.append([part.encode("utf-8")])
            continue

        for pretoken in re.finditer(PAT, part):
            b = pretoken.group().encode("utf-8")
            bs: list[bytes] = [b[i : i + 1] for i in range(len(b))]
            res.append(bs)

    return res


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.
        """
        self.vocab: dict[int, bytes] = vocab

        reversed_vocab = {}
        for id, w in vocab.items():
            reversed_vocab[w] = id
        self.reversed_vocab: dict[bytes, int] = reversed_vocab

        self.merges: list[tuple[bytes, bytes]] = merges
        if special_tokens is None:
            special_tokens = []

        self.special_tokens: list[str] = special_tokens

    def encode(self, text: str) -> list[int]:
        pretokens = pretokenize(text, self.special_tokens)
        res = []
        for pretoken in pretokens:
            current: list[bytes] = pretoken.copy()

            for merge in self.merges:
                next: list[bytes] = []

                idx = 0
                while idx < len(current):
                    if idx + 1 < len(current) and merge == (
                        current[idx],
                        current[idx + 1],
                    ):
                        # print(f"Merge pair: {merge}")
                        next.append(current[idx] + current[idx + 1])
                        idx += 2
                    else:
                        next.append(current[idx])
                        idx += 1

                current = next

            # Turn vocab into int
            encoded = [self.reversed_vocab[w] for w in current]
            res.extend(encoded)

        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode(
            "utf-8", errors="replace"
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merge_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        raise NotImplementedError
