"""Fine-tune dataset: subset anns -> (image, caption, eda, idx, pose, hard_i, hard_pose, hard_caption,
anomaly_label, mined_i, mined_pose, mined_caption). Joint horizontal-flip on image+pose for consistency.
Reuses CMP's pre_caption + EDA. Missing hard/mined/pose fields degrade gracefully (return zeros/None).
"""
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from dataset.utils import pre_caption          # CMP util


class FTTrainDataset(Dataset):
    def __init__(self, anns, image_root, img_tf, pose_tf, config):
        self.anns = anns
        self.image_root = image_root
        self.img_tf, self.pose_tf = img_tf, pose_tf
        self.max_words = config["max_words"]
        self.eda_p = config.get("eda_p", 1)
        self.be_hard = config.get("be_hard", True)
        self.be_pose = config.get("be_pose_img", True)
        self.use_mined = config.get("use_mined_neg", True)
        ids = {}
        for a in anns:
            i = a["image_id"]
            if i not in ids: ids[i] = len(ids)
        self.img_ids = ids

    def __len__(self): return len(self.anns)

    def _load(self, rel, flip):
        img = Image.open(os.path.join(self.image_root, rel)).convert("RGB")
        if flip: img = TF.hflip(img)
        x = self.img_tf(img)
        pose = 0
        if self.be_pose:
            p = Image.open(os.path.join(self.image_root, "pose/" + rel)).convert("RGB")
            if flip: p = TF.hflip(p)
            pose = self.pose_tf(p)
        return x, pose

    def _cap(self, c): return c if isinstance(c, str) else (c[0] if c else "")

    def __getitem__(self, i):
        a = self.anns[i]
        flip = random.random() < 0.5
        image, pose = self._load(a["image"], flip)
        cap = self._cap(a["caption"])
        caption = pre_caption(cap, self.max_words)
        caption_eda = pre_caption(cap, self.max_words, True, self.eda_p)
        anom = int(a.get("anomaly") or 0)

        hard_i, hard_pose, hard_c = 0, 0, ""
        if self.be_hard and a.get("hard_i"):
            hard_i, hard_pose = self._load("train/" + a["hard_i"], random.random() < 0.5)
            hard_c = pre_caption(self._cap(a.get("hard_c", "")), self.max_words)

        mined_i, mined_pose, mined_c = 0, 0, ""
        if self.use_mined and a.get("mined_i"):
            mined_i, mined_pose = self._load(a["mined_i"], random.random() < 0.5)
            mined_c = pre_caption(self._cap(a.get("mined_c", "")), self.max_words)

        return (image, caption, caption_eda, self.img_ids[a["image_id"]], pose,
                hard_i, hard_pose, hard_c, anom, mined_i, mined_pose, mined_c)


def collate(batch):
    """Stack tensors; keep text lists for the tokenizer. Handles optional (0 placeholder) fields."""
    (img, cap, eda, idx, pose, hi, hp, hc, anom, mi, mp, mc) = zip(*batch)
    def stack_imgs(xs):
        xs = [x for x in xs if torch.is_tensor(x)]
        return torch.stack(xs) if xs else None
    out = {
        "image": torch.stack(img), "caption": list(cap), "caption_eda": list(eda),
        "idx": torch.tensor(idx), "pose": stack_imgs(pose),
        "hard_i": stack_imgs(hi), "hard_pose": stack_imgs(hp), "hard_caption": list(hc),
        "anomaly": torch.tensor(anom),
        "mined_i": stack_imgs(mi), "mined_pose": stack_imgs(mp), "mined_caption": list(mc),
    }
    return out
