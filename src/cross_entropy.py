import torch
import einops


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    Args
        logits: (..., vocab_size)
        targets: (..., )

    Returns
        avg loss
    """

    # (..., 1)
    logits_max = torch.max(logits, dim=-1, keepdim=True).values

    # (..., vocab_size)
    logits_stable = logits - logits_max

    # (..., vocab_size)
    exp = torch.exp(logits_stable)

    # (..., 1)
    exp_sum = torch.sum(exp, dim=-1, keepdim=True)

    # Gather target prob from [logits_stable]
    # target_logits has shape (...,).
    # target_logits[...] = the targets[...]-th value from logits_stable[...]
    target_logits = torch.gather(
        logits_stable,
        dim=len(logits_stable.shape) - 1,
        index=einops.rearrange(targets, "... (b d) -> ... b d", d=1).long(),
    )

    return torch.mean(-target_logits + torch.log(exp_sum))
