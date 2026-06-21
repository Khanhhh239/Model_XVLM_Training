"""STAR-v4: Text-Based Person Anomaly Retrieval (AI City 2026 Track 4).

SigLIP-2-L (retrieval/recall + sim2real) + X-VLM cross-encoder (rerank) + ensemble.
Train on 1x A100 80GB; infer on Kaggle T4 (free) with embedding caching.

COMPLIANCE: the competition test set = real frames from OOPS!.  NEVER train on OOPS!
or any YouTube fail-compilation data.  Train uses ONLY the provided PAB synthetic set.
"""
__version__ = "0.1.0"
