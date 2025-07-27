import math


def max_batch_size(
    gpu_mem_gb: float,
    n: int,  # sequence length
    d: int,  # model hidden size (d_model)
    L: int,  # number of transformer layers
    V: int,  # vocabulary size
    dtype_bytes: int = 4,  # bytes per float (4 for float32, 2 for float16)
) -> int:
    """
    Estimate the maximum batch size that fits in GPU memory when training a Transformer with AdamW.

    Parameters:
    - gpu_mem_gb: total GPU memory in GB
    - n: sequence length
    - d: hidden size (d_model)
    - L: number of transformer layers
    - V: vocabulary size
    - dtype_bytes: number of bytes per float (default=4 for float32, use 2 for float16)

    Returns:
    - max_batch_size: estimated maximum batch size
    """

    # Convert GB to bytes
    M = gpu_mem_gb * (2**30)

    # -------- Parameter size --------
    # P ≈ Vd (token embedding) + dV (output projection) + L * (4d^2 + 12d^2) + negligible terms
    param_count = V * d + d * V + L * (16 * d**2)  # ignoring RMSNorm params
    param_bytes = dtype_bytes * (param_count)

    # Total state: param + grad + AdamW (2 moments)
    model_state_bytes = param_bytes * 4  # 1 param + 1 grad + 2 optimizer states

    # -------- Activation size --------
    # Activation size = 4 × [L(20 b n d + 2 b n^2) + 2 b n d + b n]
    def activation_bytes_per_batch(b):
        act = L * (20 * b * n * d + 2 * b * n**2) + 2 * b * n * d + b * n
        return dtype_bytes * 4 * act

    # -------- Solve for max b --------
    # M ≥ model_state_bytes + activation_bytes_per_batch(b)
    # → b ≤ (M - model_state_bytes) / activation_bytes_per_batch(1)

    available_bytes = M - model_state_bytes
    if available_bytes <= 0:
        return 0  # not enough memory for even the parameters

    activation_per_batch = activation_bytes_per_batch(1)
    b_max = math.floor(available_bytes / activation_per_batch)

    return b_max


# 80GB GPU, 1024 seq len, 1600 hidden dim, 48 layers, 50k vocab
gpt2_xl_max_batch_size = max_batch_size(
    gpu_mem_gb=80, n=1024, d=1600, L=48, V=50527, dtype_bytes=4
)
print(f"Max batch size for GPT2-XL:{gpt2_xl_max_batch_size}")

# 16GB GPU, 1024 seq len, 1600 hidden dim, 48 layers, 50k vocab
gpt2_xl_max_batch_size = max_batch_size(
    gpu_mem_gb=16, n=256, d=512, L=4, V=10000, dtype_bytes=4
)
print(f"Max batch size for A1 small run:{gpt2_xl_max_batch_size}")
