import torch


def softmax(x: torch.Tensor, dim: int):
    assert dim < len(x.shape)

    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_val

    exp = torch.exp(x_stable)
    s = torch.sum(exp, dim=dim, keepdim=True)

    return exp / s
