import torch
from torch import nn
from einops import einsum, rearrange


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            num_embeddings: size of vocabulary
            embedding_dim: dimension of the embedding vectors
        """
        super().__init__()
        init = torch.zeros((num_embeddings, embedding_dim))
        init = nn.init.trunc_normal_(init, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(init)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, sequence_length)

        Output:
            size should be (batch_size, sequence_length, embedding_dim)
        """
        b = token_ids.shape[0]
        token_ids_flattened = rearrange(
            token_ids, "b sequence_length -> (b sequence_length)"
        )
        # Perform the lookup using direct indexing
        # PyTorch's advanced indexing allows you to pass a LongTensor of indices
        # to select rows from a tensor.
        embedded_flattended = self.W[token_ids_flattened.long()]
        embedded = rearrange(
            embedded_flattended,
            "(b sequence_length) embedding_dim -> b sequence_length embedding_dim",
            b=b,
        )

        return embedded
