import math


def cosine_lr_schedule_with_warmup(t, lr_max, lr_min, T_w, T_c):
    if t < T_w:
        return t / T_w * lr_max

    if T_w <= t <= T_c:
        return lr_min + 1 / 2 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (
            lr_max - lr_min
        )

    # t > T_c
    return lr_min
