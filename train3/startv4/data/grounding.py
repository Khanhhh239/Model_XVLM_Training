"""Box pseudo-labels for the bbox-grounding head (Phase 3).

The synthetic data has no box labels.  Useful grounding (failure group #3) needs phrase->box
pairs: extract noun phrases from each caption, then run an open-vocabulary detector
(GroundingDINO) to localise them.  DummyBoxDetector lets the pipeline run/tests pass without
the heavy detector; swap in a real GroundingDINO wrapper for the actual ~10-20h pass.

WARNING: the cheap single-person box is near-trivial (#3 unaffected) -- use phrase->box.
"""
from __future__ import annotations

import re

_STOP = set(
    "a an the of in on at with and or to is are was were be been being this that these those "
    "his her their its he she they it you we i for from by as into onto over under near".split()
)


def extract_noun_phrases(caption: str, max_phrases: int = 8, min_len: int = 3) -> list[str]:
    """Lightweight content-word extraction (no spaCy dependency)."""
    words = re.findall(r"[a-zA-Z]+", str(caption).lower())
    seen, out = set(), []
    for w in words:
        if w in _STOP or len(w) < min_len or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out[:max_phrases]


class DummyBoxDetector:
    """Returns a centred box for every phrase (test / no-GroundingDINO fallback)."""

    def __call__(self, image_path: str, phrases: list[str]):
        return [(p, [0.25, 0.25, 0.75, 0.75], 1.0) for p in phrases]


def label_boxes(image_path: str, caption: str, detector, conf: float = 0.35) -> list[dict]:
    """-> [{phrase, bbox(xyxy norm), conf}] for phrases the detector localised above conf."""
    phrases = extract_noun_phrases(caption)
    out = []
    for phrase, box, c in detector(image_path, phrases):
        if c >= conf:
            out.append({"phrase": phrase, "bbox": list(box), "conf": float(c)})
    return out
