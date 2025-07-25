import numpy as np
from einops import rearrange, einsum
import torch


def get_batch(
    dataset: np.typing.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args
        dataset: 1d numpy array of integer tokens
    Returns
        - (sampled input sequence, corresponding next-token targest). Both
          should have shape (batch_size, context_length)
    """
    dataset_len = dataset.shape[0]
    max_valid_starting_index = dataset_len - context_length - 1

    starting_indices = np.random.randint(0, max_valid_starting_index + 1, (batch_size,))
    seq_offsets = np.arange(context_length)

    # Adding (b, 1) and (1, s) to get (b, s)
    seq_indices = rearrange(starting_indices, "(b s) -> b s", s=1) + rearrange(
        seq_offsets, "(b s) -> b s", b=1
    )

    input_batch = dataset[seq_indices]
    output_batch = dataset[seq_indices + 1]

    return (
        torch.from_numpy(input_batch).to(device),
        torch.from_numpy(output_batch).to(device),
    )


# Manual loop
def get_batch_(
    dataset: np.typing.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args
        dataset: 1d numpy array of integer tokens
    Returns
        - (sampled input sequence, corresponding next-token targest). Both
          should have shape (batch_size, context_length)
    """
    dataset_len = dataset.shape[0]
    max_valid_starting_index = dataset_len - context_length - 1

    starting_indices = np.random.randint(
        low=0, high=max_valid_starting_index + 1, size=(batch_size,)
    )

    input_batch = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    output_batch = np.zeros((batch_size, context_length), dtype=dataset.dtype)

    for i in range(batch_size):
        input_batch[i] = dataset[
            starting_indices[i] : starting_indices[i] + context_length
        ]
        output_batch[i] = dataset[
            starting_indices[i] + 1 : starting_indices[i] + context_length + 1
        ]

    return (
        torch.tensor(input_batch, device=device),
        torch.tensor(output_batch, device=device),
    )
