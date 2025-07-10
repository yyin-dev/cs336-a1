import os
import regex as re
import collections
from multiprocessing import Process, Queue
from .pretokenization_example import find_chunk_boundaries

NUM_PROCESS = 8
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(input_path, start, end, special_tokens, queue):
    escaped_special_tokens = list(map(re.escape, special_tokens))
    special_tokens_regex = "|".join(escaped_special_tokens)

    pretoken_freq: dict[bytes, int] = collections.defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Split on special tokens, and pretokensize results separately.
    # For example, [Doc 1]<ike [Doc 1]<|endoftext|>[Doc2], you should split
    # on the special token <|endoftext|>, and pre-tokenize [Doc 1] and
    # [Doc 2] separately, so that no merging can occur across the document
    # boundary.
    for part in re.split(special_tokens_regex, chunk):
        for pretoken in re.finditer(PAT, part):
            pretoken = pretoken.group()
            pretoken_bytes: bytes = pretoken.encode("utf-8")
            pretoken_freq[pretoken_bytes] += 1

    queue.put(pretoken_freq)


def parallel_pretokenize(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[bytes, int]:
    pretoken_freq: dict[bytes, int] = collections.defaultdict(int)
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, NUM_PROCESS, "<|endoftext|>".encode("utf-8")
        )

    queue = Queue()
    processes = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        process = Process(
            target=pretokenize_chunk,
            args=(input_path, start, end, special_tokens, queue),
        )
        processes.append(process)
        process.start()

    for process in processes:
        # Must get() from queue before join(). O/w the queue fills up and blocks
        # process from put(), and blocks everything
        chunk_pretoken_freq = queue.get()

        for pretoken, freq in chunk_pretoken_freq.items():
            pretoken_freq[pretoken] += freq

    for process in processes:
        process.join()

    return pretoken_freq


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
    pretoken_counts: dict[bytes, int] = parallel_pretokenize(input_path, special_tokens)

    vocab: list[bytes] = list(map(int.to_bytes, list(range(0, 256))))
    vocab.extend(list(map(lambda s: s.encode("utf-8"), special_tokens)))
    print(f"Init vocab size (including special tokens): {len(vocab)}")

    pretoken_frequencies: dict[int, int] = {}
    pretoken_subwords: dict[int, list[bytes]] = {}
    pair_frequencies: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    pair_positions: dict[tuple[bytes, bytes], dict[int, set[int]]] = (
        collections.defaultdict(lambda: collections.defaultdict(set))
    )

    pretoken_id = 0
    for pretoken, count in pretoken_counts.items():
        pretoken_frequencies[pretoken_id] = count
        pretoken_subwords[pretoken_id] = list(map(int.to_bytes, list(pretoken)))
        pretoken_id += 1

    for pretoken_id, frequency in pretoken_frequencies.items():
        subwords = pretoken_subwords[pretoken_id]

        for position, pair in enumerate(zip(subwords[:-1], subwords[1:])):
            pair_frequencies[pair] += frequency
            pair_positions[pair][pretoken_id].add(position)

    iter = 0
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        if len(pair_frequencies) == 0 or max(pair_frequencies.values()) == 0:
            break

        # print(f"=== Iteration {iter} ===")

        def comp(pair):
            pair_freq = pair_frequencies.get(pair)
            return (pair_freq, pair)

        # print(f"Before merge")
        # print(f"Pair frequencies: {pair_frequencies}")
        # print(f"Pair positions: {pair_positions}")

        selected_pair = max(pair_frequencies, key=comp)
        new_vocab: bytes = selected_pair[0] + selected_pair[1]
        # print(f"!!! Selected: {selected_pair}")
        # print(f"Pair positions: {pair_positions[selected_pair]}")

        for pretoken_id in pair_positions[selected_pair]:
            positions = list(pair_positions[selected_pair][pretoken_id])

            # print(f"pretoken: {pretoken_id}, {pretoken_subwords[pretoken_id]}")
            # print(f"positions: {positions}")

            frequency = pretoken_frequencies[pretoken_id]

            for i in reversed(range(len(positions))):
                position = positions[i]

                old_subwords = pretoken_subwords[pretoken_id]
                new_subwords = old_subwords.copy()

                # print(f"position: {position}")

                # prev pair
                if position - 1 >= 0:
                    old_prev_pair = (old_subwords[position - 1], old_subwords[position])
                    # print(f"old prev pair: {old_prev_pair}")
                    pair_frequencies[old_prev_pair] -= frequency

                    pair_positions[old_prev_pair][pretoken_id].remove(position - 1)

                    new_prev_pair = (old_subwords[position - 1], new_vocab)
                    # print(f"new prev pair: {new_prev_pair}")
                    pair_frequencies[new_prev_pair] += frequency
                    pair_positions[new_prev_pair][pretoken_id].add(position - 1)

                # next pair
                if position + 2 < len(old_subwords):
                    old_next_pair = (
                        old_subwords[position + 1],
                        old_subwords[position + 2],
                    )
                    # print(f"old next pair: {old_next_pair}")
                    pair_frequencies[old_next_pair] -= frequency
                    pair_positions[old_next_pair][pretoken_id].remove(position + 1)

                    new_next_pair = (new_vocab, old_subwords[position + 2])
                    # print(f"new next pair: {new_next_pair}")
                    pair_frequencies[new_next_pair] += frequency
                    pair_positions[new_next_pair][pretoken_id].add(position)

                for pair in set(
                    zip(old_subwords[position + 2 : -1], old_subwords[position + 3 :])
                ):
                    ps = list(pair_positions[pair][pretoken_id])
                    for idx, p in enumerate(ps):
                        if p >= position + 2:
                            ps[idx] -= 1

                    pair_positions[pair][pretoken_id] = set(ps)
                    # print(f"adjusting: {pair}: {pair_positions[pair][pretoken_id]}")

                new_subwords[position] = new_vocab
                new_subwords.pop(position + 1)

                for i, pos in enumerate(positions):
                    if pos > position:
                        positions[i] -= 1

                pretoken_subwords[pretoken_id] = new_subwords

        del pair_frequencies[selected_pair]
        del pair_positions[selected_pair]

        iter += 1
        merges.append(selected_pair)
        vocab.append(new_vocab)

    return (dict(enumerate(vocab)), merges)
