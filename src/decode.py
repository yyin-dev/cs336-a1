"""
Decode LM to generate text.
"""

import torch
from softmax import softmax
from transformer import Transformer
from tokenizer import Tokenizer


def decode(
    model: Transformer,
    prompt: torch.Tensor,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] = ["<|endoftext|>"],
    max_generated_tokens: int = 1,
    temperature: float = 1.0,
    top_p_sampling_threshold: float = 1.0,
):
    """
    Stops at <|endoftext|> or max_generated_tokens.

    Args:
        prompt: (seq_len,) TODO: support batching
    """

    generated_tokens: list[str] = []

    input = prompt
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    while len(generated_tokens) < max_generated_tokens:
        if generated_tokens and generated_tokens[-1] == "<|endoftext|>":
            break

        output = model(input)

        # Temperature scaling (applied before softmax)
        output /= temperature

        # Extract the last logits only
        logits = softmax(output, dim=-1)[-1]

        # Top-p sampling
        logits_sorted, indices = torch.sort(logits, descending=True)
        assert len(vocab) == logits.shape[-1]

        p = len(vocab)  # Set to vocab size in case we never reach threshold
        accumulated_probability = 0
        for i in range(len(vocab)):
            accumulated_probability += logits_sorted[i]
            if accumulated_probability >= top_p_sampling_threshold:
                p = i
                break

        mask = torch.zeros((len(vocab),)).bool()
        top_p_indices = indices[: p + 1]
        for idx in top_p_indices:
            mask[idx] = True

        logits = torch.masked_fill(logits, ~mask, 0)

        # Sample from probabilities
        # Re-normalize with softmax
        probs = softmax(logits, dim=-1)
        selected_id = torch.multinomial(probs, 1).numpy()[0]
        next_token = tokenizer.decode([selected_id])

        generated_tokens.append(next_token)
        input = torch.cat([input, torch.tensor([selected_id])])
