"""Distractor-val: queries+GT from the OLD labeled test (attr.json) + N distractors from the
competition gallery (never a GT for the old queries) -> single-GT mAP/R@k. Used as the SAFETY
gauge: training must keep this >= the 80-mAP baseline, else we ship the original checkpoint.
Pose-OFF by default (no pose maps needed for old-test/distractors); set pose if you must match a
pose-on baseline protocol.
"""
import json
import torch
import torch.nn.functional as F
from PIL import Image


def _read_json_any(p):
    txt = open(p, encoding="utf-8").read().strip()
    try: return json.loads(txt)
    except Exception: return [json.loads(l) for l in txt.splitlines() if l.strip()]


@torch.no_grad()
def _encode_imgs(model, paths, tf, device, bs=64):
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(device).half()
        e, _ = model.get_vision_embeds(x)
        feats.append(F.normalize(model.get_image_feat(e), dim=-1).float().cpu())
    return torch.cat(feats)


@torch.no_grad()
def _encode_txt(model, caps, tok, device, max_tokens, bs=128):
    feats = []
    for i in range(0, len(caps), bs):
        t = tok(caps[i:i + bs], padding="max_length", truncation=True, max_length=max_tokens,
                return_tensors="pt").to(device)
        e = model.get_text_embeds(t.input_ids, t.attention_mask)
        feats.append(F.normalize(model.get_text_feat(e), dim=-1).float().cpu())
    return torch.cat(feats)


def build_val(val_dir, gal_dir, gal_names, n_distract=5000):
    rows = _read_json_any(f"{val_dir}/attr.json")
    qcaps, gpaths, qgid, seen = [], [], [], {}
    for r in rows:
        img = r["image"]
        if img not in seen:
            seen[img] = len(gpaths); gpaths.append(f"{val_dir}/{img}")
        caps = r.get("caption"); caps = caps if isinstance(caps, list) else [caps] if caps else []
        for c in caps:
            qcaps.append(c); qgid.append(seen[img])
    gpaths += [f"{gal_dir}/{n}" for n in gal_names[:n_distract]]      # distractors (no GT among old queries)
    return qcaps, gpaths, torch.tensor(qgid)


@torch.no_grad()
def evaluate(model, tok, tf, device, max_tokens, qcaps, gpaths, qgid):
    model.eval()
    qf = _encode_txt(model, qcaps, tok, device, max_tokens)
    gf = _encode_imgs(model, gpaths, tf, device)
    sim = qf @ gf.t()                                  # [Q, G]
    order = sim.argsort(dim=1, descending=True)
    ranks = (order == qgid.view(-1, 1)).float().argmax(dim=1)   # rank of GT (0-based)
    r1 = (ranks < 1).float().mean().item()
    r5 = (ranks < 5).float().mean().item()
    r10 = (ranks < 10).float().mean().item()
    mAP = (1.0 / (ranks.float() + 1.0)).mean().item()           # single-GT AP = 1/rank
    return {"mAP": mAP, "R@1": r1, "R@5": r5, "R@10": r10}
