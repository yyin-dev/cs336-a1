import regex as re
import logger
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_as_str(s: str, special_tokens: set[str]) -> list[str]:
    """
    Return pretokens as str.
    """

    # Different from training, at encoding we should include special tokens
    # in the pretokenization result.

    # re.split is greedy when finding matches in patterns. For example,
    # re.split("abbc", "(b)|(bb)") returns ["a", "b", "b", "c"].
    # To prioritize longest match, put "bb" before "b" in the regex pattern.
    escaped_special_tokens = list(
        map(re.escape, sorted(list(special_tokens), key=len, reverse=True))
    )

    # Wrap special token patterns in parenthesis so that they are returned
    # by `re.split`
    escaped_and_wrapped_in_parens = list(
        map(lambda s: f"({s})", escaped_special_tokens)
    )
    special_tokens_regex = "|".join(escaped_and_wrapped_in_parens)

    # Edge case: there's no special tokens, so regex becomes "".
    if special_tokens_regex == "":
        parts = [s]
    else:
        parts = re.split(special_tokens_regex, s)

    res: list[str] = []
    for part in parts:
        if part is None:
            continue

        if part in special_tokens:
            res.append(part)
        else:
            for pretoken in re.finditer(PAT, part):
                res.append(pretoken.group())

    return res


def pretokenize(s: str, special_tokens: set[str]) -> list[list[bytes]]:
    """
    Return pretokens as list[bytes]. Special tokens are not broken down into
    individual bytes. The vocabulary includes special tokens, so will be encoded
    as is.
    """

    pretokens = pretokenize_as_str(s, special_tokens)

    res: list[list[bytes]] = []
    for pretoken in pretokens:
        if pretoken in special_tokens:
            res.append([pretoken.encode("utf-8")])
        else:
            b = pretoken.encode("utf-8")
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
        self.merges_dict: dict[tuple[bytes, bytes], int] = {}
        for index, merge in enumerate(merges):
            priority = index
            self.merges_dict[merge] = priority

        if special_tokens is None:
            special_tokens = []
        self.special_tokens: set[str] = set(special_tokens)
        self.special_tokens.add("<|endoftext|>")
        logger.info(f"Special tokens: {self.special_tokens}")

    def apply_merges_naive(self, current: list[bytes]):
        """
        This function has no return value. Update [pretoken] in place.
        """
        print(f"pretoken: {current}")

        # Naive approach:
        # For each merge, iterate through all pairs in the pretoken.
        # Time complexity: O(merges x avg pretoken length)
        for merge in self.merges:
            n = len(current)
            if n == 1:
                break

            # Find merges from left to right
            merges = []
            idx = 0
            while idx < n:
                if idx + 1 < n and merge == (
                    current[idx],
                    current[idx + 1],
                ):
                    merges.append(idx)
                    idx += 2
                else:
                    idx += 1

            # Merge from right to left, to avoid index shifting
            for merge_idx in reversed(merges):
                current[merge_idx] = current[merge_idx] + current[merge_idx + 1]
                current.pop(merge_idx + 1)

    def apply_merges_optimized(self, current: list[bytes]):
        if len(current) == 1:
            return

        while True:
            highest_priority_pair = None
            highest_priority = len(self.merges)
            for pair in zip(current[:-1], current[1:]):
                if pair not in self.merges_dict:
                    continue

                priority = self.merges_dict[pair]
                if highest_priority_pair is None or priority < highest_priority:
                    highest_priority = priority
                    highest_priority_pair = pair
                elif priority == highest_priority:
                    highest_priority_pair = pair

            if highest_priority_pair is None:
                break

            idx = 0
            merge_indices = []
            while idx < len(current) - 1:
                if highest_priority_pair == (current[idx], current[idx + 1]):
                    merge_indices.append(idx)
                    idx += 2
                else:
                    idx += 1

            for merge_idx in reversed(merge_indices):
                current[merge_idx] = current[merge_idx] + current[merge_idx + 1]
                current.pop(merge_idx + 1)

    def encode(self, text: str) -> list[int]:
        pretokens = pretokenize(text, self.special_tokens)
        res = []
        for pretoken in pretokens:
            entire_pretoken = b"".join(pretoken)
            if entire_pretoken in self.reversed_vocab:
                res.append(self.reversed_vocab[entire_pretoken])
            else:
                current: list[bytes] = pretoken
                # self.apply_merges_naive(current)
                self.apply_merges_optimized(current)

                # Turn vocab into int
                encoded = [self.reversed_vocab[w] for w in current]
                res.extend(encoded)

        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields
        token IDs. This is required for memory-efficient tokenization of large
        files that we cannot load directly into memory.
        """
        # We need to break the file into chunks, and process each chunk in-turn.
        # Need to make sure that a token doesn't cross chunk boundaries. Let's
        # use the same trick in pretokenization: split by speical tokens.
        #
        # Algorithm: after seeing the n-th special token, we know the string
        # between n-1-th and n-th speical tokens is a valid chunk. A chunk could
        # cross the results from [iterable]!
        current_chunk = ""
        for s in iterable:
            # Don't forget leftover from previous iteration!
            s = current_chunk + s
            current_chunk = ""

            pretokens = pretokenize_as_str(s, self.special_tokens)

            for pretoken in pretokens:
                current_chunk += pretoken

                if pretoken in self.special_tokens:
                    # This is a valid chunk, yield all token ids in the chunk
                    for id in self.encode(current_chunk):
                        yield id

                    current_chunk = ""

        # yield any remaining ids
        for id in self.encode(current_chunk):
            yield id

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
