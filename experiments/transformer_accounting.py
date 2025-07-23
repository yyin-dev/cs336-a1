def param_count(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    # context_length doesn't matter for param count!

    token_embedding = vocab_size * d_model

    transformer_block = 2 * d_model + 3 * d_ff * d_model + 3 * d_model * d_model
    transofmer_block_total = num_layers * transformer_block

    rms_norm = d_model
    output_embedding = d_model * vocab_size

    return token_embedding + transofmer_block_total + rms_norm + output_embedding


def transformer_flop_count(
    n,  # sequence length
    vocab_size,
    num_layers,
    d_model,
    d_ff,
):
    """
    Count forward-pass FLOPs for a decoder-only transformer.
    Only matrix multiplications are included.
    """

    # Attention FLOPs per layer
    attn_proj = 6 * n * d_model * d_model  # Q, K, V projections
    qk_dot = 2 * n * n * d_model  # Q x K^T
    av_dot = 2 * n * n * d_model  # Attention weights x V
    out_proj = 2 * n * d_model * d_model  # Output projection
    attn_total = attn_proj + qk_dot + av_dot + out_proj

    # Feedforward FLOPs per layer (GLU style)
    ffn_total = 6 * n * d_model * d_ff

    # Total FLOPs per layer
    per_layer_flops = attn_total + ffn_total

    # All layers
    all_layers_flops = num_layers * per_layer_flops

    # Final output embedding projection (logits)
    output_proj = 2 * n * d_model * vocab_size

    total_flops = all_layers_flops + output_proj
    return total_flops


# Around 2B. GPT2-XL's param is roughly 1.5B.
# Reason: GPT2-XL uses only W1 and W2 in FFN (it's not gated)
#
# Updating this, the param is roughly
gpt2_xl = param_count(
    vocab_size=50257,
    context_length=1024,
    num_layers=48,
    d_model=1600,
    num_heads=25,
    d_ff=6400,
)
print(f"GPT2-XL param count: {gpt2_xl:,}")

# Around 4T.
gpt2_xl_flops = transformer_flop_count(
    n=1024,
    vocab_size=50257,
    num_layers=48,
    d_model=1600,
    d_ff=6400,
)
print(f"FLOPs (GPT2-XL, n=1024): {gpt2_xl_flops:,}")

# Around 0.5T.
gpt2_small_flops = transformer_flop_count(
    n=1024,
    vocab_size=50257,
    num_layers=12,
    d_model=768,
    d_ff=6400,
)
print(f"FLOPs (GPT2-small, n=1024): {gpt2_small_flops:,}")

# Around 1.4T
gpt2_medium_flops = transformer_flop_count(
    n=1024,
    vocab_size=50257,
    num_layers=24,
    d_model=1024,
    d_ff=6400,
)
print(f"FLOPs (GPT2-medium, n=1024): {gpt2_medium_flops:,}")

# Around 2.6T
gpt2_large_flops = transformer_flop_count(
    n=1024,
    vocab_size=50257,
    num_layers=36,
    d_model=1280,
    d_ff=6400,
)
print(f"FLOPs (GPT2-large, n=1024): {gpt2_large_flops:,}")

gpt2_xl_flops_larger_context = transformer_flop_count(
    n=16384,
    vocab_size=50257,
    num_layers=48,
    d_model=1600,
    d_ff=6400,
)
print(f"FLOPs (GPT2-XL, n=16384): {gpt2_xl_flops_larger_context:,}")
