"""Perceptual-hash de-duplication (compliance tool for distractor-val).

External distractors must NOT accidentally include test-gallery images.  We hash every
candidate distractor and drop any that is a near-duplicate of a test-gallery image
(difference-hash + Hamming distance).  This ENFORCES the no-test-data rule.

dHash is robust to mild resize/compression; popcount is order-independent so byte
endianness does not matter.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash (default hash_size=8 -> 8*8 bits)."""
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]  # [hash_size, hash_size]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def _hamming_to_refs(h: int, refs: np.ndarray) -> np.ndarray:
    """Hamming distance of scalar hash `h` to a uint64 array `refs` -> int array."""
    x = np.bitwise_xor(refs, np.uint64(h)).astype(np.uint64)
    bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1)  # [R, 64]
    return bits.sum(axis=1)


def dedup_keep_mask(
    candidate_hashes: list[int], reference_hashes: list[int], threshold: int = 5
) -> list[bool]:
    """Return keep[i] = True iff candidate i has NO reference within `threshold` bits."""
    if not reference_hashes:
        return [True] * len(candidate_hashes)
    refs = np.array([np.uint64(r) for r in reference_hashes], dtype=np.uint64)
    keep = []
    for h in candidate_hashes:
        d = _hamming_to_refs(h, refs)
        keep.append(bool((d > threshold).all()))
    return keep
