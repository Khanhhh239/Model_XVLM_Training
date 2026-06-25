"""Action-keyword alignment loss (wrong-case group C: action / pose mismatch).

Pulls each image embedding toward its short ACTION-PHRASE embedding (the manifest `action` column,
e.g. "playing ping pong", "falling off a skateboard") and pushes it away from other actions in the
batch. Uses IDENTITY SOFT TARGETS — rows sharing the same action label are mutual positives (exactly
the ITC trick in `itc.py`), so different frames of the same action are never treated as negatives.
Reduces to ordinary symmetric InfoNCE when every label is unique.

Why this helps: best (3).pth already retrieves the right *neighbourhood* (R@10 0.977) but mis-ranks
rank-1 (R@1 0.70) — many failures are "same scene, wrong action". Aligning the image with the action
phrase sharpens exactly that axis.

Pure tensors (no backbone import) -> unit-testable on CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def action_alignment_loss(img_feat: Tensor, act_feat: Tensor, group_ids: Tensor,
                          temp: float = 0.07) -> Tensor:
    """Symmetric image<->action-phrase InfoNCE with identity soft targets.

    Args:
        img_feat:  [N, d] image features (re-normalized here).
        act_feat:  [N, d] action-phrase text features (re-normalized here).
        group_ids: [N] long — rows with equal id share the same action (mutual positives).
        temp:      softmax temperature (division), matching itc.py / itc_with_xbm.
    Returns:
        scalar loss. With unique group_ids this equals 0.5*(CE(sim,arange)+CE(sim.T,arange)).
    """
    img = F.normalize(img_feat, dim=-1)
    act = F.normalize(act_feat, dim=-1)
    sim = img @ act.t() / temp                                  # [N, N]
    pos = (group_ids[:, None] == group_ids[None, :]).float()    # symmetric positive mask
    targets = pos / pos.sum(dim=1, keepdim=True).clamp_min(1.0)  # row-normalized soft targets
    loss_i2a = -(F.log_softmax(sim, dim=1) * targets).sum(dim=1).mean()
    loss_a2i = -(F.log_softmax(sim.t(), dim=1) * targets).sum(dim=1).mean()   # pos symmetric -> reuse
    return 0.5 * (loss_i2a + loss_a2i)
