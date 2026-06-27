"""STARModel — assembles the backbone (+LoRA on image+cross, frozen text, +pose) and computes
the multi-task loss exactly as in the plan:

    L = w_itc * ITC + lambda_itm * ITM(hard-neg) + lambda_smooth_ap * Smooth-AP

Notes vs the earlier draft (per the annotated plan):
  - MLM head + loss REMOVED.
  - Text encoder is FROZEN (no LoRA); only the image encoder + cross-encoder are adapted.
  - Pose branch is fused into the IMAGE-encoder branch (affects f_V) with NO separate pose loss
    (it is trained through the ITC/Smooth-AP gradient on f_V).
See analyze.md for the math.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..losses import ITCLoss, ITMLoss, SmoothAPLoss, action_alignment_loss, build_itm_pairs
from ..losses.weighting import build_weighter
from ..losses.xbm_queue import XBMQueue
from .backbone import build_backbone
from .heads import AnomalyClassificationHead, BoxGroundingHead, PhraseBoxHead
from .lora import count_trainable, mark_only_lora_trainable
from .pose import PoseHeatmapCrossAttn


class STARModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)

        # LoRA on image+cross only + freeze text tower (backbone owns the module layout)
        self.n_lora = self.backbone.setup_finetuning(cfg)

        # pose branch (toggle): SSDC Eq1 heatmap->conv->cross-attn fused into the IMAGE TOKEN stream
        # (img_embeds), sized to the backbone's patch-token width. No separate pose loss — trained
        # through ITC/ITM/SmoothAP gradients on the pose-enhanced tokens.
        self.pose = (PoseHeatmapCrossAttn(self.backbone.vision_width,
                                          heatmap_size=cfg.model.pose_heatmap_size,
                                          conv_ch=cfg.model.pose_conv_ch,
                                          pose_grid=cfg.model.pose_grid,
                                          n_heads=cfg.model.pose_n_heads)
                     if cfg.model.pose_enabled else None)

        # STAGE 1: New heads for box grounding + anomaly classification
        self.bbox_head = BoxGroundingHead(cfg.model.embed_dim) if cfg.model.bbox_enabled else None
        self.anomaly_head = AnomalyClassificationHead(cfg.model.embed_dim) if cfg.model.anomaly_enabled else None
        # phrase-grounded box head (group D): sized to the cross-encoder [CLS] width
        self.phrase_box_head = (PhraseBoxHead(self.backbone.cross_width)
                                if getattr(cfg.model, "phrase_box_enabled", False) else None)

        # XBM Queue for ITC (provides extra negatives from past batches)
        self.xbm_queue = XBMQueue(size=cfg.loss.xbm_size, dim=cfg.model.embed_dim) if cfg.loss.xbm_enabled else None

        # losses. Review fix #6: if the backbone exposes a pretrained `temp` (real X-VLM), reuse it
        # instead of creating a fresh one (the dummy backbone has none -> ITC owns its temp).
        self.itc = ITCLoss(temp_init=cfg.loss.itc_temp_init,
                           external_temp=getattr(self.backbone, "temp", None))
        self.smap = SmoothAPLoss(tau=cfg.loss.smooth_ap_temp)
        self.itm = ITMLoss()

        # multi-task weighting: fixed (default) | uncertainty | dwa  (analyze.md §14)
        self.weighter = build_weighter(cfg.loss)

        # CRITICAL FIX for STAGE 1: Freeze vision encoder 100%, train cross-attention + heads
        # In STAGE 1 (lora_enabled=False): n_lora=0, but we MUST freeze vision encoder
        # In STAGE 2 (lora_enabled=True): n_lora>0, LoRA on vision+cross, rest frozen
        if self.n_lora > 0:
            # STAGE 2: LoRA enabled -> train only LoRA params + task heads
            mark_only_lora_trainable(
                self, train_heads=("itm_head", "img_proj", "vision_proj", "pose", "temp", "gate",
                                   "weighter", "bbox_head", "anomaly_head", "phrase_box_head"),
            )
        else:
            # STAGE 1: No LoRA -> freeze vision encoder + text encoder, train cross-attention + heads
            # Strategy: Freeze all backbone.net params that are vision or text-related
            for name, param in self.named_parameters():
                # Freeze vision encoder (Swin-B or dummy patch/img_pos)
                if "backbone.net.patch" in name or "backbone.net.img_pos" in name or "vision_encoder" in name:
                    param.requires_grad_(False)
                # Freeze text encoder + txt_proj (text side 100% frozen)
                elif any(k in name for k in ("backbone.net.tok_embed", "backbone.net.txt_pos", 
                                              "backbone.net.text_self", "backbone.net.txt_proj",
                                              "text_encoder")):
                    param.requires_grad_(False)
                # Train cross-attention, img_proj, ITM head, new heads, pose, weighter, temp
                else:
                    param.requires_grad_(True)

    # ------------------------------------------------------------------ info
    def trainable_summary(self) -> str:
        tr, tot = count_trainable(self)
        return f"trainable={tr/1e6:.2f}M / total={tot/1e6:.2f}M ({100*tr/tot:.1f}%) | lora_layers={self.n_lora}"

    # ------------------------------------------------------------------ forward / losses
    def forward(self, batch: dict, step: int = 0) -> dict[str, Tensor]:
        image, ids, mask = batch["image"], batch["input_ids"], batch["attention_mask"]
        inst = batch.get("instance_id")

        img_embeds, img_feat = self.backbone.encode_image(image)
        txt_embeds, txt_feat = self.backbone.encode_text(ids, mask)
        if self.pose is not None and "keypoints" in batch:
            # token-level pose fusion (SSDC Eq1): enhance img_embeds, then RE-POOL the bi-encoder
            # feature. The enhanced img_embeds also flow into ITM below -> rerank is pose-aware.
            img_embeds = self.pose(img_embeds, batch["keypoints"])
            img_feat = self.backbone.pool_vision(img_embeds)

        n = img_feat.size(0)
        device = img_feat.device
        if inst is None:
            inst = torch.arange(n, device=device)

        # ---- ITC with XBM Queue (extra negatives from past batches) ----
        queue_img, queue_txt = None, None
        if self.xbm_queue is not None:
            queue_img, queue_txt = self.xbm_queue.get_queue()
        loss_itc = self.itc(img_feat, txt_feat, ids=inst, queue_img=queue_img, queue_txt=queue_txt)
        
        # Update XBM queue AFTER computing loss (detached features)
        if self.xbm_queue is not None:
            self.xbm_queue.enqueue(img_feat.detach(), txt_feat.detach())

        # ---- Smooth-AP (relevance = same-instance) ----
        relevance = (inst[:, None] == inst[None, :]).float()
        loss_smap = self.smap(img_feat, txt_feat, relevance)

        # ---- ITM with hard negatives ----
        # Review fix #1: sample negatives from softmax(sim / temp) like X-VLM (peaked on the
        # HARDEST negatives), not softmax(cos / 0.5). temp is already clamped by the ITC call above.
        with torch.no_grad():
            temp = self.itc.temp.detach().clamp(min=1e-3)
            sim_i2t = (img_feat @ txt_feat.t()) / temp
            forbid = inst[:, None] == inst[None, :]          # same instance => not a negative
        pairs = build_itm_pairs(sim_i2t, dup_mask=forbid, temperature=1.0)
        itm_logits = self.backbone.itm_logits(
            img_embeds[pairs["img_idx"]], txt_embeds[pairs["txt_idx"]], mask[pairs["txt_idx"]]
        )
        loss_itm = self.itm(itm_logits, pairs["label"])

        # ---- STAGE 1: Box Grounding Loss (only for rows with bbox) ----
        loss_box = torch.tensor(0.0, device=device)
        if self.bbox_head is not None and "bbox" in batch:
            bbox_pred = self.bbox_head(img_feat)
            bbox_gt = batch["bbox"]
            bbox_mask = batch.get("bbox_mask")  # [B] binary mask (1=valid, 0=skip)
            loss_box = BoxGroundingHead.compute_loss(
                bbox_pred, bbox_gt, mask=bbox_mask,
                w_giou=self.cfg.loss.w_box_giou,
                w_l1=self.cfg.loss.w_box_l1,
            )
            if self.cfg.loss.box_rampup_steps > 0:           # ramp fresh head 0->full
                loss_box = loss_box * min(1.0, step / self.cfg.loss.box_rampup_steps)

        # ---- STAGE 1: Anomaly Classification Loss ----
        loss_anomaly = torch.tensor(0.0, device=device)
        if self.anomaly_head is not None and "anomaly_label" in batch:
            anomaly_logits = self.anomaly_head(img_feat)
            anomaly_labels = batch["anomaly_label"]  # [B] 0=normal, 1=anomaly
            anomaly_mask = batch.get("anomaly_mask")  # [B] binary mask (1=valid, 0=skip)
            loss_anomaly = AnomalyClassificationHead.compute_loss(
                anomaly_logits, anomaly_labels, mask=anomaly_mask
            )
            # Ramp-up: gradually increase weight from 0 to full over first N steps
            if self.cfg.loss.anomaly_rampup_steps > 0:
                rampup_factor = min(1.0, step / self.cfg.loss.anomaly_rampup_steps)
                loss_anomaly = loss_anomaly * rampup_factor

        # ---- Action-keyword alignment (group C: action/pose mismatch) ----
        # Align the (pose-fused) image feature with its action-phrase embedding; mask out the
        # hard-partner rows (action="hard_pair") which carry no real action label.
        loss_action = torch.tensor(0.0, device=device)
        if (self.cfg.loss.lambda_action > 0 and "action_input_ids" in batch
                and batch.get("action_valid") is not None):
            vmask = batch["action_valid"].bool()
            if int(vmask.sum()) >= 2:
                _, act_feat = self.backbone.encode_text(
                    batch["action_input_ids"], batch["action_attention_mask"])
                loss_action = action_alignment_loss(
                    img_feat[vmask], act_feat[vmask], batch["action_group"][vmask],
                    temp=self.cfg.loss.action_temp)
                if self.cfg.loss.action_rampup_steps > 0:    # ramp 0->full
                    loss_action = loss_action * min(1.0, step / self.cfg.loss.action_rampup_steps)

        # ---- Phrase-grounded box (group D: multi-person / #3) ----
        # cross-encode (image, each caption noun-phrase) -> [CLS] -> box; masked GIoU+L1 over P phrases.
        loss_pbox = torch.tensor(0.0, device=device)
        if (self.phrase_box_head is not None and "phrase_input_ids" in batch
                and batch.get("phrase_mask") is not None and bool(batch["phrase_mask"].any())):
            B, P, Lp = batch["phrase_input_ids"].shape
            flat_ids = batch["phrase_input_ids"].reshape(B * P, Lp)
            flat_mask = batch["phrase_attention_mask"].reshape(B * P, Lp)
            ph_embeds, _ = self.backbone.encode_text(flat_ids, flat_mask)
            ie = img_embeds.unsqueeze(1).expand(B, P, *img_embeds.shape[1:]).reshape(B * P, *img_embeds.shape[1:])
            cross_cls = self.backbone.cross_feature(ie, ph_embeds, flat_mask)      # [B*P, cross_width]
            pred = self.phrase_box_head(cross_cls)                                  # [B*P, 4]
            gt = batch["phrase_box"].reshape(B * P, 4)
            m = batch["phrase_mask"].reshape(B * P).bool()
            loss_pbox = PhraseBoxHead.compute_loss(pred, gt, mask=m,
                                                   w_giou=self.cfg.loss.w_box_giou,
                                                   w_l1=self.cfg.loss.w_box_l1)
            if self.cfg.loss.phrase_box_rampup > 0:
                loss_pbox = loss_pbox * min(1.0, step / self.cfg.loss.phrase_box_rampup)

        # ---- total: weighter combines the tasks ----
        losses = {
            "itc": loss_itc,
            "itm": loss_itm,
            "smap": loss_smap,
            "box": loss_box,
            "anomaly": loss_anomaly,
            "action": loss_action,
            "pbox": loss_pbox,
        }
        total = self.weighter(losses)

        # components are returned NON-detached so the trainer's per-loss grad-norm diagnostic
        # (train.grad_norm_every) can backprop through them; harmless for logging (.item()).
        return {
            "loss": total,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_smap": loss_smap,
            "loss_box": loss_box,
            "loss_anomaly": loss_anomaly,
            "loss_action": loss_action,
            "loss_pbox": loss_pbox,
        }

    @torch.no_grad()
    def encode_for_eval(self, image, ids, mask, keypoints=None):
        """Return (img_feat, txt_feat) for retrieval evaluation (no augmentation).

        If the pose branch is enabled, keypoints MUST be fused here too — otherwise the
        eval embedding space differs from the trained one (train/eval mismatch).
        """
        img_embeds, img_feat = self.backbone.encode_image(image)
        if self.pose is not None and keypoints is not None:
            img_embeds = self.pose(img_embeds, keypoints)          # token-level pose fusion (SSDC Eq1)
            img_feat = self.backbone.pool_vision(img_embeds)
        _, txt_feat = self.backbone.encode_text(ids, mask)
        return img_feat, txt_feat
