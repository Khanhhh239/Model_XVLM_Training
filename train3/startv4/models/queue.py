"""MoCo-style FIFO negative queue (He et al., CVPR 2020).

Stores tens of thousands of detached, L2-normalised past keys so InfoNCE sees many more
negatives than the batch provides.  Used by the X-VLM ITC head only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeQueue(nn.Module):
    def __init__(self, dim: int, size: int = 65536):
        super().__init__()
        self.size = size
        self.register_buffer("queue", F.normalize(torch.randn(size, dim), dim=1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """keys: [B, D] (will be detached + L2-normalised before storing)."""
        keys = F.normalize(keys.detach().float(), dim=1)
        b = keys.size(0)
        if b > self.size:
            keys = keys[-self.size :]
            b = self.size
        ptr = int(self.ptr.item())
        end = ptr + b
        if end <= self.size:
            self.queue[ptr:end] = keys
        else:
            first = self.size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: b - first] = keys[first:]
        self.ptr[0] = end % self.size
        self.filled[0] = min(self.size, int(self.filled.item()) + b)

    def get(self) -> torch.Tensor:
        """Return the valid portion of the queue ([min(filled,size), D])."""
        n = int(self.filled.item())
        return self.queue[:n] if 0 < n < self.size else self.queue
