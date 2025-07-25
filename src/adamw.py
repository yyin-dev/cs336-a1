from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        """
        Args:
            params
            lr: α
            betas: β1,β2
            eps: ϵ
            weight_decay: λ
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if betas[0] < 0 or betas[1] < 0:
            raise ValueError(f"Invalid betas: {betas}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):  # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                g = p.grad

                # Read state
                m = state.get("m", torch.zeros_like(g))
                v = state.get("v", torch.zeros_like(g))
                t: int = state.get("t", 1)  # Starts at 1

                m = betas[0] * m + (1 - betas[0]) * g
                v = betas[1] * v + (1 - betas[1]) * g * g

                # Note: use adjusted learning rate for
                # direct gradient update, but not weight decay
                lr_t = (
                    lr
                    * math.sqrt(1 - math.pow(betas[1], t))
                    / (1 - math.pow(betas[0], t))
                )
                p.data -= lr_t * m / torch.sqrt(v + eps)
                p.data -= lr * weight_decay * p.data

                # Write state
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
