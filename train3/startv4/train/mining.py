"""Hard-negative mining on TRAIN data (Phase 3) -- OFFLINE curriculum utility.

Encode the train set with the current model and, for each text, find the hardest WRONG image
(highest ITC sim, excluding its own GT).  Intended use: write the returned mapping back into the
manifest's `pair_image_id` column between runs so the PairBatchSampler puts the model's own hard
confusions in-batch next epoch (a curriculum).  NOTE: this is a STANDALONE utility -- it is NOT
called inside the training loop (XVLMTrainer._itm_loss samples hard negatives in-batch from the
ITC similarity, ALBEF-style); wire it into your run script if you want cross-batch mining.

CRITICAL: mine on TRAIN only -- NEVER on test/old-test (= train-on-eval = cheating).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def mine_hard_negatives(model, loader, device: str = "cpu") -> dict[int, int]:
    """Return {sample_index -> hardest_wrong_sample_index} using batch['index'] ids."""
    model.eval()
    img_feats, txt_feats, idxs = [], [], []
    for b in loader:
        ip, _ = model.encode_image(
            b["pixel_values"].to(device), b["keypoints"].to(device) if "keypoints" in b else None
        )
        tp, _, _ = model.encode_text(b["input_ids"].to(device), b["attention_mask"].to(device))
        ii, tt = model.itc(ip, tp)
        img_feats.append(ii.cpu())
        txt_feats.append(tt.cpu())
        idxs.append(b["index"])
    I = torch.cat(img_feats)
    T = torch.cat(txt_feats)
    idxs = torch.cat(idxs)
    sim = T @ I.t()  # sim[i, j] = text_i . image_j
    n = sim.size(0)
    sim[torch.arange(n), torch.arange(n)] = float("-inf")  # exclude the true pair
    hard = sim.argmax(dim=1)
    return {int(idxs[i]): int(idxs[hard[i]]) for i in range(n)}
