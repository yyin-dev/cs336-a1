import torch
import os
from typing import IO, BinaryIO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    torch.save(
        {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
):
    """
    Returns: iteration number
    """
    states = torch.load(src)
    model.load_state_dict(states["model_state"])
    if optimizer:
        optimizer.load_state_dict(states["optimizer_state"])
    return states["iteration"]
