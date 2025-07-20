import torch
from einops import einsum
from softmax import softmax
import math


def scaled_dot_product_attention(
    Q, K, V, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args
        Q: (... n d_k)
        K: (... m d_k)
        V: (... m d_v)
        mask: (n, m), bool tensor.
    Returns:
        (... n d_v)
    """

    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[-2] == V.shape[-2]

    d_k = K.shape[-1]

    QK = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
    QK_scaled: torch.Tensor = QK / math.sqrt(d_k)

    if mask is not None:
        # Don't do:
        #   QK_scaled[~mask] += -torch.inf
        # Because in-place matrix modification can break torch's autograd.
        QK_scaled = QK_scaled.masked_fill(~mask, -torch.inf)

    attention_weights = softmax(QK_scaled, dim=-1)
    result = einsum(attention_weights, V, "... n m, ... m d_v -> ... n d_v")

    return result
