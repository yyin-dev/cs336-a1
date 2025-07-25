from collections.abc import Iterable
import torch.nn as nn
import math
from einops import einsum


def clip_gradient(parameters: Iterable[nn.Parameter], max_norm):
    # The l2-norm is computed globally among all parameters!
    grad_square_sum = 0

    for param in parameters:
        grad = param.grad
        if grad is None:
            continue

        grad_squares = einsum(grad * grad, "... -> ")
        grad_square_sum += grad_squares

    global_l2_norm = math.sqrt(grad_square_sum)

    if global_l2_norm < max_norm:
        return

    # Clipping
    factor = max_norm / (global_l2_norm + math.pow(10, -6))
    for param in parameters:
        if param.grad is not None:
            param.grad *= factor
