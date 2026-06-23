"""SearchStage1 = CMP Search + box_head + anomaly_head + XBM-ITC, for the Stage-1 Kaggle fine-tune:
FREEZE Swin (all) + BERT text-lower layers 0..5; FULL-FT BERT fusion layers 6..11 + itm_head + proj +
temp + pose_block + box_head + anomaly_head. Losses: ITC(+XBM) + ITM(+ID-hard +mined) + anomaly + box.
NO MLM (text-lower frozen). Vision runs in no_grad (frozen) -> tiny memory -> no OOM on T4 @384.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_search import Search
import xbm as _xbm


def giou_l1_loss(pred, tgt):
    """pred,tgt: [N,4] xyxy in [0,1]. Returns mean (GIoU 2x + L1 5x) DETR-style weights."""
    l1 = F.l1_loss(pred, tgt, reduction="mean")
    # iou
    x1 = torch.max(pred[:, 0], tgt[:, 0]); y1 = torch.max(pred[:, 1], tgt[:, 1])
    x2 = torch.min(pred[:, 2], tgt[:, 2]); y2 = torch.min(pred[:, 3], tgt[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    ap = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
    at = (tgt[:, 2] - tgt[:, 0]).clamp(0) * (tgt[:, 3] - tgt[:, 1]).clamp(0)
    union = ap + at - inter + 1e-6
    iou = inter / union
    # enclosing
    cx1 = torch.min(pred[:, 0], tgt[:, 0]); cy1 = torch.min(pred[:, 1], tgt[:, 1])
    cx2 = torch.max(pred[:, 2], tgt[:, 2]); cy2 = torch.max(pred[:, 3], tgt[:, 3])
    carea = (cx2 - cx1).clamp(0) * (cy2 - cy1).clamp(0) + 1e-6
    giou = iou - (carea - union) / carea
    return 2.0 * (1 - giou).mean() + 5.0 * l1


class SearchStage1(Search):
    def __init__(self, config):
        super().__init__(config)
        vw = config.get("vision_width", 1024)
        self.anomaly_head = nn.Linear(vw, 2)
        self.box_head = nn.Sequential(nn.Linear(self.text_width, 256), nn.ReLU(),
                                      nn.Linear(256, 4), nn.Sigmoid())
        self.w_anom = config.get("w_anom", 0.2)
        self.w_box = config.get("w_box", 0.1)
        self.xbm = None  # set by trainer after .to(device)

    # ---------- freeze policy ----------
    def freeze_stage1(self, n_text_frozen=6, freeze_vision=True):
        for p in self.parameters():
            p.requires_grad = False
        te = self.text_encoder.bert.encoder.layer
        for i in range(n_text_frozen, len(te)):              # fusion layers 6..11 (self+cross attn)
            for p in te[i].parameters():
                p.requires_grad = True
        for name in ("vision_proj", "text_proj", "itm_head", "pose_block", "pose_conv",
                     "box_head", "anomaly_head"):
            m = getattr(self, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True
        self.temp.requires_grad = True
        if not freeze_vision:                                # optional Stage-2-ish: LoRA handled outside
            pass
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[stage1] trainable {tr/1e6:.1f}M (vision frozen={freeze_vision}, text-lower 0..{n_text_frozen-1} frozen)")

    @torch.no_grad()
    def _vis(self, image, pose=None):
        emb, atts = self.get_vision_embeds(image)            # frozen -> no_grad
        return emb, atts, pose

    def _vis_pose(self, image, pose):
        """image_embed (detached, frozen vision) fused with pose via the TRAINABLE pose_block."""
        emb, atts = self.get_vision_embeds(image)            # this runs under no_grad via caller
        emb = emb.detach()
        if self.be_pose_img and pose is not None:
            p = self.pose_conv(pose.detach() if False else pose) if self.be_pose_conv else pose
            pe, _ = self.get_vision_embeds(pose)
            emb = self.pose_block(emb, pe.detach())           # pose_block trains; inputs detached (vision frozen)
        return emb, atts

    def forward(self, image, text_ids, text_atts, idx=None, pose=None,
                hard_i=None, hard_pose=None, hard_text_ids=None, hard_text_atts=None,
                mined_i=None, mined_pose=None, mined_text_ids=None, mined_text_atts=None,
                anomaly_label=None, box_text_ids=None, box_text_atts=None, box_tgt=None, box_mask=None):
        dev = image.device
        with torch.no_grad():                                # frozen vision
            img_emb_raw, img_atts = self.get_vision_embeds(image)
            hard_emb_raw = self.get_vision_embeds(hard_i)[0] if hard_i is not None else None
            mined_emb_raw = self.get_vision_embeds(mined_i)[0] if mined_i is not None else None
            pose_e = self.get_vision_embeds(pose)[0] if (self.be_pose_img and pose is not None) else None
        # pose fusion (pose_block TRAINS; vision feats detached)
        img_emb = img_emb_raw
        if pose_e is not None:
            img_emb = self.pose_block(img_emb_raw, self.pose_conv(pose_e) if self.be_pose_conv else pose_e)
        text_emb = self.get_text_embeds(text_ids, text_atts)         # fusion layers 6..11 train
        img_feat = self.get_image_feat(img_emb)
        text_feat = self.get_text_feat(text_emb)

        # ITC with XBM queue
        loss_itc = _xbm.itc_with_xbm(img_feat, text_feat, self.temp, self.xbm, idx)
        if self.xbm is not None:
            self.xbm.enqueue(img_feat, text_feat)

        # ITM (positive + in-batch) + ID-hard + mined cross-ID
        loss_itm = self.get_matching_loss(img_emb, img_atts, img_feat, text_emb, text_atts, text_feat, idx=idx)
        for he_raw, hid, hat in ((hard_emb_raw, hard_text_ids, hard_text_atts),
                                 (mined_emb_raw, mined_text_ids, mined_text_atts)):
            if he_raw is not None and hid is not None:
                he = self.pose_block(he_raw, self.pose_conv(pose_e) if self.be_pose_conv else pose_e) if (pose_e is not None) else he_raw
                hatts = torch.ones(he.shape[:-1], dtype=torch.long, device=dev)
                hte = self.get_text_embeds(hid, hat)
                loss_itm = loss_itm + self.get_matching_loss_hard(img_emb, img_atts, he, hatts,
                                                                  text_emb, text_atts, hte, hat)
        # anomaly head on the (pose-fused) IMAGE cls — image-only, no text leak
        loss_anom = img_emb.new_zeros(())
        if anomaly_label is not None:
            loss_anom = self.w_anom * F.cross_entropy(self.anomaly_head(img_emb[:, 0, :]), anomaly_label)

        # box head: cross-encode (image, phrase) -> cls -> box; over up to P phrases, masked
        loss_box = img_emb.new_zeros(())
        if self.w_box > 0 and box_text_ids is not None and box_mask is not None and box_mask.any():
            B, P, L = box_text_ids.shape
            flat_ids = box_text_ids.reshape(B * P, L); flat_at = box_text_atts.reshape(B * P, L)
            ph_emb = self.get_text_embeds(flat_ids, flat_at)
            ie = img_emb.unsqueeze(1).expand(B, P, *img_emb.shape[1:]).reshape(B * P, *img_emb.shape[1:])
            ia = img_atts.unsqueeze(1).expand(B, P, img_atts.shape[1]).reshape(B * P, img_atts.shape[1])
            cross = self.get_cross_embeds(ie, ia, text_embeds=ph_emb, text_atts=flat_at)[:, 0, :]
            pred = self.box_head(cross)                                  # [B*P,4]
            m = box_mask.reshape(-1).bool()
            if m.any():
                loss_box = self.w_box * giou_l1_loss(pred[m], box_tgt.reshape(B * P, 4)[m])
        return {"itc": loss_itc, "itm": loss_itm, "anom": loss_anom, "box": loss_box}
