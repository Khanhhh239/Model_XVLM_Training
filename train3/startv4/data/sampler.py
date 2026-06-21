"""Batch samplers.

- PairBatchSampler     : keep each anchor next to its same-video hard partner in a batch
                         (structural hard negatives -- the v3 trick).
- BalancedBucketSampler: roughly equal normal/anomaly per batch.
"""
from __future__ import annotations

import random


class PairBatchSampler:
    def __init__(self, pairs, batch_size: int, seed: int = 42, drop_last: bool = True):
        if batch_size % 2 != 0:
            raise ValueError("PairBatchSampler needs an even batch_size")
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        pairs = self.pairs[:]
        rng.shuffle(pairs)
        for k in range(0, len(pairs), self.pairs_per_batch):
            chunk = pairs[k : k + self.pairs_per_batch]
            if self.drop_last and len(chunk) < self.pairs_per_batch:
                break
            batch = []
            for a, b in chunk:
                batch.extend([a, b])
            yield batch

    def __len__(self) -> int:
        n = len(self.pairs) // self.pairs_per_batch
        return n if self.drop_last else (len(self.pairs) + self.pairs_per_batch - 1) // self.pairs_per_batch


class BalancedBucketSampler:
    """Interleave indices from each bucket so every batch is class-balanced."""

    def __init__(self, buckets: list[int], batch_size: int, seed: int = 42, drop_last: bool = True):
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.groups: dict[int, list[int]] = {}
        for i, b in enumerate(buckets):
            self.groups.setdefault(int(b), []).append(i)
        self._epoch = 0
        self._n = len(buckets)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        pools = {k: v[:] for k, v in self.groups.items()}
        for v in pools.values():
            rng.shuffle(v)
        order: list[int] = []
        keys = list(pools.keys())
        ptrs = {k: 0 for k in keys}
        remaining = sum(len(v) for v in pools.values())
        ki = 0
        while remaining > 0:
            k = keys[ki % len(keys)]
            if ptrs[k] < len(pools[k]):
                order.append(pools[k][ptrs[k]])
                ptrs[k] += 1
                remaining -= 1
            ki += 1
        for s in range(0, len(order), self.batch_size):
            batch = order[s : s + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch

    def __len__(self) -> int:
        return self._n // self.batch_size if self.drop_last else (self._n + self.batch_size - 1) // self.batch_size
