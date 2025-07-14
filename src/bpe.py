"""
BPE (Byte Pair Encoding) training.

Time complexity analysis
------------------------

Let V be the vocabulary size. Let L be the corpus size as number of pretokens.
The number of merges is O(V - 256 - speical tokens) = O(V).

The pretokenization phase takes O(L) time.

- For naive implementation.
  - For each merging:
    - Find the pair to merge takes O(L) because it iterates over all pretokens.
    - Merging takes O(L) because we need to iterate over all pretokens too.
  - So the total complexity is O(VL)

- For the implementation that tracks pair positions
  - For each merging:
    - Assuming we store pair counts in a heap, finding the pair takes O(1) time,
      and maintaining it takes O(logP) time, where P is the number of pairs.
    - Merging one occurence of a pair takes O(1) time.
  - Total time depends on the total number of occurences processed across all
    merge iterations. So the total time is:
    - pretokenization: O(L)
    - finding pair to merge: O(V logP).
    - merging: O(C), where C is the total number of occurence of all merged pairs.
    - It seems right to say that both P and C are O(L). So total time is
      O(L) + O(V logL) + O(L) = O(L + VlogL).
"""

import os
import regex as re
import collections
from multiprocessing import Process, Queue
from .pretokenization_example import find_chunk_boundaries
import logger
import heapq

NUM_CHUNKS = 6
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(input_path, start, end, special_tokens, queue):
    logger.info(f"Start pretokenizing chunk, {start}, {end}")
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

    logger.info(f"Done pretokenizing chunk, {start}, {end}")
    queue.put(pretoken_freq)

    return pretoken_freq


def parallel_pretokenize(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[bytes, int]:
    pretoken_freq: dict[bytes, int] = collections.defaultdict(int)
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, NUM_CHUNKS, "<|endoftext|>".encode("utf-8")
        )

    queue = Queue()
    processes = []

    # Set to False to save memory with serial pretokenization to when deal with
    # very large corpus.
    parallel = True

    if parallel:
        logger.info("Parallel tokenization!")
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            process = Process(
                target=pretokenize_chunk,
                args=(input_path, start, end, special_tokens, queue),
            )
            processes.append(process)
            process.start()

        for _ in processes:
            # Must get() from queue before join(). O/w the queue fills up and blocks
            # process from put(), and blocks everything
            chunk_pretoken_freq = queue.get()

            for pretoken, freq in chunk_pretoken_freq.items():
                pretoken_freq[pretoken] += freq

        for process in processes:
            process.join()
    else:
        logger.info("Serial tokenization!")
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            chunk_pretoken_freq = pretokenize_chunk(
                input_path, start, end, special_tokens, queue
            )

            for pretoken, freq in chunk_pretoken_freq.items():
                pretoken_freq[pretoken] += freq

    return pretoken_freq


class BytesPairReversedComp:
    """
    To be used in `PairFrequencyHeap` for tie-breaking when two pairs
    have the same frequency.
    """

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        # Reverse the comparison to get max lexicographical order in min-heap
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair

    def __hash__(self):
        return hash(self.pair)


class PairFrequencyHeap:
    """
    Max heap for pair frequencies

    Use negative frequencies because `heapq` is a min-heap but we want a max heap.
    Use `BytesPairReversedComp` to select max-lexicographical pair for tie breaking.
    """

    def __init__(self):
        # There might be stale entries in self.heap. Treat [self.pair_to_freq] as
        # source of truth. See pop_max.
        self.heap: list[tuple[int, BytesPairReversedComp]] = []
        self.pair_to_freq: dict[tuple[bytes, bytes], int] = {}

    def push(self, pair, frequency):
        """Add or update a pair with its frequency"""
        self.pair_to_freq[pair] = frequency
        heapq.heappush(self.heap, (-frequency, BytesPairReversedComp(pair)))

    def pop_max(self) -> tuple[tuple[bytes, bytes] | None, int]:
        """Remove and return the pair with highest frequency"""
        while self.heap:
            neg_freq, pair = heapq.heappop(self.heap)
            freq = -neg_freq
            pair = pair.pair

            if pair not in self.pair_to_freq or self.pair_to_freq[pair] != freq:
                continue

            # Skip if frequency is 0 (pair was decremented to 0)
            if freq == 0:
                del self.pair_to_freq[pair]
                continue

            del self.pair_to_freq[pair]
            return pair, freq

        return None, 0

    def update_frequency(self, pair: tuple[bytes, bytes], delta):
        """Update frequency of a pair by delta amount"""
        if pair not in self.pair_to_freq:
            if delta > 0:
                self.push(pair, delta)
            return

        old_freq = self.pair_to_freq[pair]
        new_freq = old_freq + delta

        if new_freq <= 0:
            if pair in self.pair_to_freq:
                del self.pair_to_freq[pair]
        else:
            self.push(pair, new_freq)

    def is_empty(self):
        """Check if heap has any valid pairs"""
        return len(self.pair_to_freq) == 0


def select_pair_to_merge(
    pair_frequencies: PairFrequencyHeap,
) -> tuple[bytes, bytes] | None:
    return pair_frequencies.pop_max()[0]


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
    logger.info("Finished pretokenization!")

    vocab: list[bytes] = list(map(int.to_bytes, list(range(0, 256))))
    vocab.extend(list(map(lambda s: s.encode("utf-8"), special_tokens)))

    # pretoken_id -> frequency
    pretoken_frequencies: dict[int, int] = {}

    # pretoken_id -> subwords
    pretoken_subwords: dict[int, list[bytes]] = {}

    # pair -> frequency
    pair_frequencies: PairFrequencyHeap = PairFrequencyHeap()

    # pair -> pretoken_id -> positions. Store positions in a set so that we
    # can remove efficiently
    pair_positions: dict[tuple[bytes, bytes], dict[int, set[int]]] = (
        collections.defaultdict(lambda: collections.defaultdict(set))
    )

    # Populate initial values
    for pretoken_id, (pretoken, frequency) in enumerate(pretoken_counts.items()):
        pretoken_frequencies[pretoken_id] = frequency

        subwords = list(map(int.to_bytes, list(pretoken)))
        pretoken_subwords[pretoken_id] = subwords

        for position, pair in enumerate(zip(subwords[:-1], subwords[1:])):
            pair_frequencies.update_frequency(pair, frequency)
            pair_positions[pair][pretoken_id].add(position)

    iter = 0
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        selected_pair = select_pair_to_merge(pair_frequencies)
        if selected_pair is None:
            break

        new_vocab: bytes = selected_pair[0] + selected_pair[1]

        # Core logic of merging & incrementally updating counts.
        for pretoken_id in pair_positions[selected_pair]:
            frequency = pretoken_frequencies[pretoken_id]

            # Iterate from left to right. Iterating from right to left can
            # slightly simplify index-shifting handling, but produces
            # different results. Example: for a single-world corpus 'ooo',
            # after the 1st merge of ('o', 'o'), left-to-right produces
            # 'oo o' while right-to-left produces 'o oo'.
            while len(pair_positions[selected_pair][pretoken_id]) > 0:
                subwords = pretoken_subwords[pretoken_id]
                position = min((pair_positions[selected_pair][pretoken_id]))

                # When merging a pair in a token, we must do two things:
                # 1. Update pair_frequencies for all affected pairs. This includes
                # - pairs *overlapping* with the selected pair.
                # - selected pair
                #
                # 2. Update pair_positions for all affected pairs. This includes
                # - Pairs overlapping with the selected pair
                # - Pairs after the merged pair in this pretoken. Shift
                #   to the left by 1.
                #
                #   Note that when removing positions, remove positions in
                #   the subword *before* merge; when adding positions, add
                #   positions in the subword *after* merge.

                # Pair overlapping with the first word
                # E.g. For pretoken ['a', 'b', 'c', 'd'], merge 'b' and 'c'.
                # The pair ('a', 'b') becomes ('a', 'bc')
                if position - 1 >= 0:
                    pair_before_merge = (subwords[position - 1], subwords[position])
                    pair_frequencies.update_frequency(pair_before_merge, -frequency)
                    pair_positions[pair_before_merge][pretoken_id].discard(position - 1)

                    pair_after_merge = (subwords[position - 1], new_vocab)
                    pair_frequencies.update_frequency(pair_after_merge, frequency)
                    pair_positions[pair_after_merge][pretoken_id].add(position - 1)

                # Pair overlapping with the second word
                # E.g. For pretoken ['a', 'b', 'c', 'd'], merge 'b' and 'c'.
                # The pair ('c', 'd') becomes ('bc', 'd')
                if position + 2 < len(subwords):
                    pair_before_merge = (subwords[position + 1], subwords[position + 2])
                    pair_frequencies.update_frequency(pair_before_merge, -frequency)
                    pair_positions[pair_before_merge][pretoken_id].discard(position + 1)

                    pair_after_merge = (new_vocab, subwords[position + 2])
                    pair_frequencies.update_frequency(pair_after_merge, frequency)
                    pair_positions[pair_after_merge][pretoken_id].add(position)

                # Update for selected pair
                # Don't need to update [pair_frequencies] because we called
                # `pop_max` which also removed the entry.
                pair_positions[selected_pair][pretoken_id].discard(position)

                # Two edge cases:
                # 1. Even if a pair appears multiple times in the pretoken after
                # the merged pair, we just need to shift the index by one!
                # 2. The pair could be the same as the selected pair!
                pairs_after_selected_pair = set(
                    zip(subwords[position + 2 :], subwords[position + 3 :])
                )

                for pair in pairs_after_selected_pair:
                    new_positions = set()
                    for pos in pair_positions[pair][pretoken_id]:
                        if pos >= position + 2:
                            new_positions.add(pos - 1)
                        else:
                            new_positions.add(pos)

                    pair_positions[pair][pretoken_id] = new_positions

                # Update subwords
                subwords[position] = new_vocab
                subwords.pop(position + 1)

        del pair_positions[selected_pair]

        iter += 1
        merges.append(selected_pair)
        vocab.append(new_vocab)

    logger.info("Finished training BPE tokenizer!")
    return (dict(enumerate(vocab)), merges)
