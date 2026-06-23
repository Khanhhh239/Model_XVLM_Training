"""Stage-1 dataset over the data team's 30K hard subset (train_30k_hard.jsonl).
Returns image + caption + anomaly + ID-hard + (mined) + pose-skeleton + up-to-P grounded boxes.
Image path: row['image']='train/imgs_X/..webp' -> WEBP_ROOT/imgs_X/..webp (strip 'train/').
Boxes from boxes_30k.jsonl keyed by image_id (pixel xyxy -> normalized [0,1]).
Pose from a rendered pose dir (render_pose.py), same rel path as the image.
"""
import json, os, random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from dataset.utils import pre_caption


def _strip(rel):
    return rel[len("train/"):] if rel.startswith("train/") else rel


def load_boxes(path, topk=2, min_score=0.35):
    """image_id -> [(phrase, [x1,y1,x2,y2] pixel, score)] top-k by score."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        bs = sorted(d.get("boxes", []), key=lambda b: -b["score"])
        bs = [b for b in bs if b["score"] >= min_score][:topk]
        if bs:
            out[d["image_id"]] = bs
    return out


class Stage1Dataset(Dataset):
    def __init__(self, anns, webp_root, pose_root, boxes, img_tf, pose_tf, config, P=2):
        self.anns = anns
        self.webp_root, self.pose_root = webp_root, pose_root
        self.boxes = boxes
        self.img_tf, self.pose_tf = img_tf, pose_tf
        self.max_words = config["max_words"]
        self.size = config["h"]
        self.box_src = config.get("box_src_size", 384)        # res the boxes were generated at (webp=384)
        self.be_pose = config.get("be_pose_img", True)
        self.be_hard = config.get("be_hard", True)
        self.use_mined = config.get("use_mined_neg", True)
        self.P = P
        ids = {}
        for a in anns:
            i = a["image_id"]
            if i not in ids: ids[i] = len(ids)
        self.img_ids = ids

    def __len__(self): return len(self.anns)

    def _anom(self, a):
        v = str(a.get("label_type", a.get("action", ""))).lower()
        return 0 if v in ("normal", "cn") else 1

    def _load(self, rel, flip):
        img = Image.open(os.path.join(self.webp_root, _strip(rel))).convert("RGB")
        if flip: img = TF.hflip(img)
        x = self.img_tf(img)
        pose = 0
        if self.be_pose and self.pose_root:
            pp = os.path.join(self.pose_root, _strip(rel))
            if os.path.exists(pp):
                p = Image.open(pp).convert("RGB")
                if flip: p = TF.hflip(p)
                pose = self.pose_tf(p)
        return x, pose

    def _cap(self, c): return c if isinstance(c, str) else (c[0] if c else "")

    def __getitem__(self, i):
        a = self.anns[i]
        flip = random.random() < 0.5
        image, pose = self._load(a["image"], flip)
        caption = pre_caption(self._cap(a["caption"]), self.max_words)
        anom = self._anom(a)

        hard_i, hard_pose, hard_c = 0, 0, ""
        if self.be_hard and a.get("hard_i"):
            hard_i, hard_pose = self._load(a["hard_i"], random.random() < 0.5)
            hard_c = pre_caption(self._cap(a.get("hard_c", "")), self.max_words)

        mined_i, mined_pose, mined_c = 0, 0, ""
        if self.use_mined and a.get("mined_i"):
            mined_i, mined_pose = self._load(a["mined_i"], random.random() < 0.5)
            mined_c = pre_caption(self._cap(a.get("mined_c", "")), self.max_words)

        # boxes (normalized; flip x if image flipped); pad to P
        phr = [""] * self.P
        tgt = torch.zeros(self.P, 4); msk = torch.zeros(self.P)
        for k, b in enumerate(self.boxes.get(a["image_id"], [])[: self.P]):
            x1, y1, x2, y2 = [c / self.box_src for c in b["box"]]   # normalize by box-gen res (384), not train res
            if flip: x1, x2 = 1 - x2, 1 - x1
            phr[k] = b["phrase"]; tgt[k] = torch.tensor([x1, y1, x2, y2]); msk[k] = 1.0
        return (image, caption, self.img_ids[a["image_id"]], pose, hard_i, hard_pose, hard_c,
                anom, mined_i, mined_pose, mined_c, phr, tgt, msk)


def collate(batch):
    (img, cap, idx, pose, hi, hp, hc, anom, mi, mp, mc, phr, tgt, msk) = zip(*batch)
    st = lambda xs: torch.stack([x for x in xs if torch.is_tensor(x)]) if any(torch.is_tensor(x) for x in xs) else None
    return {
        "image": torch.stack(img), "caption": list(cap), "idx": torch.tensor(idx),
        "pose": st(pose), "hard_i": st(hi), "hard_pose": st(hp), "hard_caption": list(hc),
        "anomaly": torch.tensor(anom), "mined_i": st(mi), "mined_pose": st(mp), "mined_caption": list(mc),
        "box_phrases": [p for sub in phr for p in sub],          # flattened B*P phrases
        "box_tgt": torch.stack(tgt), "box_mask": torch.stack(msk),
    }
