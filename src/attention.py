import torch
import torch.nn as nn
from einops import einsum, rearrange
from softmax import softmax
from rope import RotaryPositionalEmbedding
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


def init_linear_weights(d_in, d_out):
    std = math.sqrt(2 / (d_in + d_out))
    return nn.init.trunc_normal_(
        torch.zeros((d_in, d_out)), mean=0, std=std, a=-3 * std, b=3 * std
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        assert d_model % num_heads == 0

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        d = d_model // num_heads

        self.apply_rope = False
        if theta and max_seq_len:
            self.apply_rope = True
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=d, max_seq_len=max_seq_len
            )

        self.W_Q = nn.Parameter(init_linear_weights(num_heads * d, d_model))
        self.W_K = nn.Parameter(init_linear_weights(num_heads * d, d_model))
        self.W_V = nn.Parameter(init_linear_weights(num_heads * d, d_model))
        self.W_O = nn.Parameter(init_linear_weights(d_model, num_heads * d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (... seq_len d_in)
        """
        seq_len = x.shape[-2]
        d = self.d_model // self.num_heads

        # hd = num_heads * d
        Q = einsum(self.W_Q, x, "hd d_model, ... seq_len d_model -> ... seq_len hd")
        K = einsum(self.W_K, x, "hd d_model, ... seq_len d_model -> ... seq_len hd")
        V = einsum(self.W_V, x, "hd d_model, ... seq_len d_model -> ... seq_len hd")

        Q_multi = rearrange(
            Q,
            "... seq_len (num_heads d) -> ... num_heads seq_len d",
            num_heads=self.num_heads,
            d=d,
        )
        K_multi = rearrange(
            K,
            "... seq_len (num_heads d) -> ... num_heads seq_len d",
            num_heads=self.num_heads,
            d=d,
        )
        V_multi = rearrange(
            V,
            "... seq_len (num_heads d) -> ... num_heads seq_len d",
            num_heads=self.num_heads,
            d=d,
        )

        # Apply RoPE to Q and K, if needed
        device = x.device
        if self.apply_rope:
            # Because RoPEs encodes relative positions, we can start from 0
            positions = torch.arange(0, seq_len, device=device)
            Q_multi = self.rope(Q_multi, positions)
            K_multi = self.rope(K_multi, positions)

        # The mask should contain 1's in the lower diagonal (including the
        # diagonal) and zeros elsewhere. The diagonal is included because
        # a token can attend to itself.
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()

        # (... num_heads seq_len d)
        multiheaded = scaled_dot_product_attention(Q_multi, K_multi, V_multi, mask)

        # Concatenate heads
        multiheaded_concatenated = rearrange(
            multiheaded,
            "... num_heads seq_len d -> ... seq_len (num_heads d)",
            num_heads=self.num_heads,
            d=d,
        )

        # Apply W_O to concatenated result
        result = einsum(
            self.W_O,
            multiheaded_concatenated,
            "d_model hd, ... seq_len hd -> ... seq_len d_model",
        )

        return result
