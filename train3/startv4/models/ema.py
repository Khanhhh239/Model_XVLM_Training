"""Exponential Moving Average of model weights.

EMA weights are smoother and generalise better -- important for sim2real, where we want
the LoRA-adapted SigLIP to keep its real-image knowledge.  Use EMA weights for eval/infer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, s in self.shadow.items():
            sd[k].copy_(s.to(sd[k].dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow
