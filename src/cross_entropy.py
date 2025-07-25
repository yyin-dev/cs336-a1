import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    Args
        logits: (b, vocab_size)
        targets: (b, )
    """

    # (b, 1)
    logits_max = torch.max(logits, dim=-1, keepdim=True).values

    # (b, vocab_size)
    logits_stable = logits - logits_max

    # (b, vocab_size)
    exp = torch.exp(logits_stable)

    # (b, 1)
    exp_sum = torch.sum(exp, dim=-1, keepdim=True)

    b = logits.shape[0]
    target_logits = logits[torch.arange(0, b), targets]

    return torch.mean(logits_max - target_logits + torch.log(exp_sum))
