import os
import regex as re
import collections
from .pretokenization_example import find_chunk_boundaries


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_subword_pair_freq(
    pretoken_freq: dict[tuple, int],
) -> dict[tuple[bytes, bytes], int]:
    """
    Args:
        pretoken_freq (dict[tuple, int]). The key is a tuple of bytes
    """
    subword_pair_freq: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)

    for pretoken_subwords, freq in pretoken_freq.items():
        for first, second in zip(pretoken_subwords, pretoken_subwords[1:]):
            subword_pair_freq[(first, second)] += freq

    return subword_pair_freq


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # TODO: Parallize pretokenization
    escaped_special_tokens = list(map(re.escape, special_tokens))
    special_tokens_regex = "|".join(escaped_special_tokens)
    print(escaped_special_tokens)

    pretoken_freq: dict[tuple, int] = collections.defaultdict(int)
    with open(input_path, "r") as f:
        # Split on special tokens, and pretokensize results separately.
        # For example, [Doc 1]<ike [Doc 1]<|endoftext|>[Doc2], you should split
        # on the special token <|endoftext|>, and pre-tokenize [Doc 1] and
        # [Doc 2] separately, so that no merging can occur across the document
        # boundary.
        for part in re.split(special_tokens_regex, f.read()):
            for pretoken in re.finditer(PAT, part):
                pretoken = pretoken.group()
                pretoken_bytes_singleton: bytes = pretoken.encode("utf-8")
                individual_bytes: list[bytes] = [
                    pretoken_bytes_singleton[i : i + 1]
                    for i in range(len(pretoken_bytes_singleton))
                ]

                pretoken_freq[tuple(individual_bytes)] += 1

    vocab: list[bytes] = list(map(int.to_bytes, list(range(0, 256))))
    vocab.extend(list(map(lambda s: s.encode("utf-8"), special_tokens)))
    print(f"Init vocab size (including special tokens): {len(vocab)}")

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        subword_pair_freq = get_subword_pair_freq(pretoken_freq)

        # Stop if all pretokens have been merged
        if len(subword_pair_freq) == 0:
            break

        def comp_freq_then_pair(pair):
            pair_freq = subword_pair_freq[pair]
            return (pair_freq, pair)

        selected_pair = max(subword_pair_freq, key=comp_freq_then_pair)
        new_vocab = selected_pair[0] + selected_pair[1]

        merged_pretoken_freq: dict[tuple, int] = collections.defaultdict(int)
        for pretoken_subwords, freq in pretoken_freq.items():
            merged_pretoken_subwords: list[bytes] = [pretoken_subwords[0]]
            for curr in pretoken_subwords[1:]:
                prev = merged_pretoken_subwords[-1]
                if (prev, curr) == selected_pair:
                    merged_pretoken_subwords[-1] = new_vocab
                else:
                    merged_pretoken_subwords.append(curr)

            merged_pretoken_freq[tuple(merged_pretoken_subwords)] = freq

        print(f"Merge: {selected_pair} -> {new_vocab}")
        pretoken_freq = merged_pretoken_freq

        vocab.append(new_vocab)
        merges.append(selected_pair)

    vocab_dict = dict(enumerate(vocab))
    return (vocab_dict, merges)
