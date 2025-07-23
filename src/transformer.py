import torch
import torch.nn as nn
from embedding import Embedding
from transformer_block import Transformer_block
from linear import Linear
from rms_norm import RMSNorm
from transformer_block import Transformer_block


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_heads,
        d_model,
        d_ff,
        rope_theta,
        context_length,
        num_layers,
    ):
        """
        Args:
            vocab_size: for determining the dimension of the token embedding matrix
            context_length: for determining the dimension of position embedding matrix
            num_layers: number of transformer blocks to use
        """
        super().__init__()

        # Input embedding is done using lookup
        self.input_embedding = Embedding(vocab_size, d_model)

        self.transformer_blocks = nn.ModuleList(
            [
                Transformer_block(d_model, num_heads, d_ff, rope_theta, context_length)
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(d_model)

        # Output embeddin is done using a linear layer
        self.output_embedding = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.input_embedding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_embedding(x)
        return x
