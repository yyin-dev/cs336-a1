import collections

corpus = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""


pretoken_frequencies: dict[int, int] = {}
pretoken_subwords: dict[int, list[bytes]] = {}
pair_frequencies: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
pair_positions: dict[tuple[bytes, bytes], dict[int, set[int]]] = (
    collections.defaultdict(lambda: collections.defaultdict(set))
)

pretokens = corpus.split()

pretoken_counts = collections.defaultdict(int)
for pretoken in pretokens:
    pretoken_counts[pretoken] += 1

pretoken_id = 0
for pretoken, count in pretoken_counts.items():
    pretoken_frequencies[pretoken_id] = count
    pretoken_subwords[pretoken_id] = list(
        map(int.to_bytes, list(pretoken.encode("utf-8")))
    )
    pretoken_id += 1


for pretoken_id, frequency in pretoken_frequencies.items():
    subwords = pretoken_subwords[pretoken_id]

    for position, pair in enumerate(zip(subwords[:-1], subwords[1:])):
        pair_frequencies[pair] += frequency
        pair_positions[pair][pretoken_id].add(position)

print(pretoken_frequencies)
print(pretoken_subwords)
print("pair frequencies: ", pair_frequencies)
print("pair positions: ", pair_positions)

for i in range(100):
    if len(pair_frequencies) == 0 or max(pair_frequencies.values()) == 0:
        print("Done!")
        break

    print(f"=== Iteration {i} ===")

    def comp(pair):
        pair_freq = pair_frequencies.get(pair)
        return (pair_freq, pair)

    print(f"Before merge")
    print(f"Pair frequencies: {pair_frequencies}")
    print(f"Pair positions: {pair_positions}")
    print(f"Subwords: {pretoken_subwords}")

    selected_pair = max(pair_frequencies, key=comp)
    new_vocab: bytes = selected_pair[0] + selected_pair[1]
    print(f"!!! Selected: {selected_pair}")
    print(f"Pair positions: {pair_positions[selected_pair]}")

    for pretoken_id in pair_positions[selected_pair]:
        positions = list(pair_positions[selected_pair][pretoken_id])

        print(
            f"pretoken id: {pretoken_id}, subwords: {pretoken_subwords[pretoken_id]}, positions: {positions}"
        )

        frequency = pretoken_frequencies[pretoken_id]

        for i in reversed(range(len(positions))):
            position = positions[i]

            old_subwords = pretoken_subwords[pretoken_id]
            new_subwords = old_subwords.copy()

            print(f"position: {position}")

            # prev pair
            print("prev pair")
            if position - 1 >= 0:
                old_prev_pair = (old_subwords[position - 1], old_subwords[position])
                print(f"old prev pair: {old_prev_pair}")
                pair_frequencies[old_prev_pair] -= frequency
                pair_positions[old_prev_pair][pretoken_id].remove(position - 1)

                new_prev_pair = (old_subwords[position - 1], new_vocab)
                print(f"new prev pair: {new_prev_pair}")
                pair_frequencies[new_prev_pair] += frequency
                pair_positions[new_prev_pair][pretoken_id].add(position - 1)

            # next pair
            print("next pair")
            if position + 2 < len(old_subwords):
                old_next_pair = (old_subwords[position + 1], old_subwords[position + 2])
                print(f"old next pair: {old_next_pair}")
                pair_frequencies[old_next_pair] -= frequency
                print(pair_positions[old_next_pair][pretoken_id])
                pair_positions[old_next_pair][pretoken_id].remove(position + 1)

                new_next_pair = (new_vocab, old_subwords[position + 2])
                print(f"new next pair: {new_next_pair}")
                pair_frequencies[new_next_pair] += frequency
                pair_positions[new_next_pair][pretoken_id].add(position)

            for pair in set(
                zip(old_subwords[position + 2 : -1], old_subwords[position + 3 :])
            ):
                ps = list(pair_positions[pair][pretoken_id])
                for i, p in enumerate(ps):
                    if p >= position + 2:
                        ps[i] -= 1

                pair_positions[pair][pretoken_id] = set(ps)

            new_subwords[position] = new_vocab
            new_subwords.pop(position + 1)

            print(f"--new subwords: {new_subwords}--")

            for i, pos in enumerate(positions):
                if pos > position:
                    positions[i] -= 1

            pretoken_subwords[pretoken_id] = new_subwords

    del pair_frequencies[selected_pair]
    del pair_positions[selected_pair]

    print("After merge")
    print(f"Pair frequencies: {pair_frequencies}")
    print(f"Pair positions: {pair_positions}")
    print(f"Subwords: {pretoken_subwords}")

print(pretoken_subwords)
