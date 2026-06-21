"""Symmetric InfoNCE with optional MoCo-style negative queue (He et al., CVPR 2020).

Used by the X-VLM ITC head (NOT by SigLIP).  The queue supplies tens of thousands of
extra negatives so the embedding is pushed away from the huge distractor distribution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    logit_scale: torch.Tensor,
    queue_text: torch.Tensor | None = None,
    queue_image: torch.Tensor | None = None,
) -> torch.Tensor:
    """image_feat / text_feat: [B, D] L2-normalised.  queue_*: [K, D] L2-normalised
    detached past keys (or None).  Symmetric i2t + t2i cross-entropy with the matched
    pair on the diagonal of the in-batch block.
    """
    b = image_feat.size(0)
    device = image_feat.device

    txt_bank = text_feat if queue_text is None else torch.cat([text_feat, queue_text], 0)
    img_bank = image_feat if queue_image is None else torch.cat([image_feat, queue_image], 0)

    logits_i2t = logit_scale * (image_feat @ txt_bank.t())  # [B, B+K]
    logits_t2i = logit_scale * (text_feat @ img_bank.t())   # [B, B+K]
    labels = torch.arange(b, device=device)                 # positives on the diagonal
    return 0.5 * (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels))
