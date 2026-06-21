"""Cosine learning-rate schedule with linear warmup (review fix A3).

The configs/design call for "cosine, warmup 1 epoch"; the trainers build this once the loader
length (hence total steps) is known.
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.0) -> LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, prog)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, fn)
