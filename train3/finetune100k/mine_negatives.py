"""ANCE-style hard-negative mining: with the CURRENT model, find for each sample the hardest
CROSS-ID confuser (highest text_i->image_j similarity, j != i and different ID) and attach it as
mined_i / mined_c. Re-run before each epoch so negatives track the improving model.

These are exactly the look-alike-different-person confusions that the 34,795 gallery distractors are,
so training to reject them directly hardens distractor recall.
"""
import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def _encode(model, anns, image_root, transform, tokenizer, device, max_tokens, bs=64):
    img_feats, txt_feats = [], []
    for i in range(0, len(anns), bs):
        chunk = anns[i:i + bs]
        imgs = torch.stack([transform(Image.open(f"{image_root}/{a['image']}").convert("RGB"))
                            for a in chunk]).to(device).half()
        ie, _ = model.get_vision_embeds(imgs)
        img_feats.append(F.normalize(model.get_image_feat(ie), dim=-1).float().cpu())
        t = tokenizer([a["caption"] if isinstance(a["caption"], str) else a["caption"][0] for a in chunk],
                      padding="max_length", truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        te = model.get_text_embeds(t.input_ids, t.attention_mask)
        txt_feats.append(F.normalize(model.get_text_feat(te), dim=-1).float().cpu())
    return torch.cat(img_feats), torch.cat(txt_feats)


@torch.no_grad()
def mine(model, anns, image_root, transform, tokenizer, device, max_tokens, topk=10, block=2048):
    """Returns anns with 'mined_i','mined_c','mined_i_id' filled (hardest cross-ID confuser)."""
    model.eval()
    img_f, txt_f = _encode(model, anns, image_root, transform, tokenizer, device, max_tokens)
    ids = [a.get("image_id") for a in anns]
    n = len(anns)
    for s in range(0, n, block):                      # text[s:e] @ all images, pick hardest cross-ID
        e = min(n, s + block)
        sim = (txt_f[s:e] @ img_f.t())                # [b, N]
        order = sim.argsort(dim=1, descending=True)
        for r, qi in enumerate(range(s, e)):
            picked = None
            for j in order[r].tolist():
                if j == qi:           continue        # skip the GT image
                if ids[j] == ids[qi]: continue        # skip same-ID (that's the CMP hard_i already)
                picked = j; break
            if picked is None: picked = (qi + 1) % n
            a = anns[qi]
            a["mined_i"] = anns[picked]["image"]
            a["mined_c"] = anns[picked]["caption"] if isinstance(anns[picked]["caption"], str) else anns[picked]["caption"][0]
            a["mined_i_id"] = anns[picked].get("image_id")
    print(f"mined cross-ID hard negatives for {n} samples (topk-scan, skipped GT + same-ID)")
    return anns
