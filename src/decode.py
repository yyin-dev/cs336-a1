"""
Decode LM to generate text.
"""

import torch
from softmax import softmax
from transformer import Transformer
from tokenizer import Tokenizer
from einops import rearrange
import tiktoken


def decode(
    model: Transformer,
    prompt: str,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    max_generated_tokens: int = 1,
    temperature: float = 1.0,
    top_p_sampling_threshold: float = 1.0,
    use_tiktoken: bool = True,
):
    """
    Stops at <|endoftext|> or max_generated_tokens.

    Args:
        prompt: (seq_len,) TODO: support batching
    """

    generated_token_ids: list[int] = []

    if use_tiktoken:
        # Find [mergeable_ranks] and [special_tokens]
        # vocab = [ 256 individual bytes; speical tokens; merges ]
        mergeable_ranks: dict[bytes, int] = {pair: idx for (idx, pair) in vocab.items()}
        num_special_tokens = len(vocab) - 256 - len(merges)
        print(f"Number of special tokens: {num_special_tokens}. ")
        special_tokens = []
        for i in range(256, 256 + num_special_tokens):
            special_tokens.append(vocab[i])
        print(f"Special tokens: {special_tokens}")

        special_tokens_dict = {
            token.decode("utf-8"): idx + 256
            for (idx, token) in enumerate(special_tokens)
        }

        allowed_special = set(list(map(lambda b: b.decode("utf-8"), special_tokens)))
        tiktoken_enc = tiktoken.Encoding(
            name="my_encoding",
            pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens_dict,
        )

        prompt_tokens = tiktoken_enc.encode(prompt, allowed_special=allowed_special)
        input = torch.tensor(prompt_tokens)
    else:
        tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
        input = torch.tensor(tokenizer.encode(prompt))

    # Add a dummy batch dimension
    input = rearrange(input, "(b s) -> b s", b=1)

    while len(generated_token_ids) < max_generated_tokens:
        # TODO: break on endoftext

        output = model(input)
        logits = output[0, -1]

        # Temperature scaling (applied before softmax)
        logits /= temperature

        probs = softmax(logits, dim=-1)

        # Top-p sampling
        probs_sorted, indices = torch.sort(probs, descending=True)
        assert len(vocab) == probs.shape[-1]

        p = len(vocab)  # Set to vocab size in case we never reach threshold
        accumulated_probability = 0
        for i in range(len(vocab)):
            accumulated_probability += probs_sorted[i]
            if accumulated_probability >= top_p_sampling_threshold:
                p = i
                break

        mask = torch.zeros((len(vocab),), device=probs.device).bool()
        top_p_indices = indices[: p + 1]
        for idx in top_p_indices:
            mask[int(idx)] = True

        probs = torch.masked_fill(probs, ~mask, 0)
        probs /= probs.sum()

        # Sample from probabilities
        selected_id = int(torch.multinomial(probs, 1).item())

        generated_token_ids.append(selected_id)

        # input: (b, s)
        next_token_tensor = rearrange(
            torch.tensor([selected_id], device=input.device), "(b s) -> b s", b=1, s=1
        )
        input = torch.cat([input, next_token_tensor], dim=1)

    if use_tiktoken:
        res = tiktoken_enc.decode(  # pyright: ignore[reportPossiblyUnboundVariable]
            generated_token_ids
        )
    else:
        res = tokenizer.decode(  # pyright: ignore[reportPossiblyUnboundVariable]
            generated_token_ids
        )

    return res
