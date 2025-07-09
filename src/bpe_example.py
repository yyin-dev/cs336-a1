import collections

corpus = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

PAT = " "

pretokens = corpus.split()

freq_table = collections.defaultdict(int)
for pretoken in pretokens:
    pretoken_bytes = tuple(list(map(int.to_bytes, list(pretoken.encode("utf-8")))))
    freq_table[pretoken_bytes] += 1

print(f"Initial freq_table: {freq_table}")

vocabs = list(map(lambda i: int.to_bytes(i), list(range(0, 256))))
vocabs = ["<|endoftext|>"] + vocabs
print(f"vocabs: {vocabs}")


def merge(freq_table):
    pair_freqs = collections.defaultdict(int)

    for pretoken, freq in freq_table.items():
        for first, second in zip(pretoken, pretoken[1:]):
            pair_freqs[(first, second)] += freq

    print(f"Pair counts: {pair_freqs}")

    def comp(pair):
        pair_freq = pair_freqs.get(pair)
        return (pair_freq, pair)

    selected_pair = max(pair_freqs, key=comp)
    new_vocab = selected_pair[0] + selected_pair[1]
    print(f"Selected pair: {selected_pair}")

    updated_freq_table = {}
    for pretoken, freq in freq_table.items():
        # Edge case: when pretoken has only one element
        updated_pretoken = list(pretoken[:1])
        for subword in pretoken[1:]:
            prev = updated_pretoken[-1]
            if (prev, subword) == selected_pair:
                updated_pretoken[-1] = prev + subword
            else:
                updated_pretoken += [subword]

        updated_freq_table[tuple(updated_pretoken)] = freq

    return new_vocab, updated_freq_table


for i in range(6):
    print(f"=== Iteration {i} ===")
    print(f"Before merge")
    print(f"freq_table: {freq_table}")

    new_vocab, freq_table = merge(freq_table)
    vocabs.append(new_vocab)

    print("After merge")
    print(f"freq_table: {freq_table}")
    print(f"New cab: {new_vocab}")

print(f"Vocabs after merging: {vocabs}")


