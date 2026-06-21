"""Data: sim2real augmentation, manifest, dataset, samplers, de-dup, grounding."""
from .augment import build_eval_transform, build_train_transform
from .dataset import PABDatasetV4, collate_fn
from .dedup import dedup_keep_mask, dhash, hamming
from .grounding import DummyBoxDetector, extract_noun_phrases, label_boxes
from .manifest import REQUIRED_COLUMNS, load_manifest, validate_manifest
from .sampler import BalancedBucketSampler, PairBatchSampler

__all__ = [
    "build_train_transform",
    "build_eval_transform",
    "PABDatasetV4",
    "collate_fn",
    "load_manifest",
    "validate_manifest",
    "REQUIRED_COLUMNS",
    "PairBatchSampler",
    "BalancedBucketSampler",
    "dhash",
    "hamming",
    "dedup_keep_mask",
    "extract_noun_phrases",
    "DummyBoxDetector",
    "label_boxes",
]
