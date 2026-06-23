"""Sample a BALANCED 100K subset from PAB train annotations for a SAFE fine-tune.

Rules (smart sampling):
  - 50/50 normal vs anomaly (from the action/scene attribute) so both retrieval + anomaly head learn.
  - Spread across video/scene to avoid near-duplicate over-representation.
  - ALWAYS keep each item's hard_i (same-ID counterpart) -> preserves CMP's ID-hard-pair.
  - Carry an explicit `anomaly` label (1=anomaly, 0=normal) derived from the attribute.

Run on the machine that has PAB downloaded (OneDrive/Baidu via github Shuyu-XJTU/CMP):
  python sample_100k.py --ann-glob "data/PAB/annotation/train/attr_*.json" --out data/PAB/annotation/train/attr_100k.json --n 100000
"""
import argparse, glob, json, os, random
from collections import defaultdict


def read_anns(paths):
    anns = []
    for p in paths:
        txt = open(p, encoding="utf-8").read().strip()
        try:
            rows = json.loads(txt)
        except Exception:
            rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
        anns.extend(rows)
    return anns


def is_anomaly(ann):
    """Best-effort anomaly flag from the attribute fields (Ca=anomaly, Cn=normal)."""
    for k in ("anomaly", "is_anomaly", "label", "bucket", "type", "caption_type"):
        if k in ann:
            v = str(ann[k]).lower()
            if v in ("1", "true", "anomaly", "abnormal", "ca", "a"):
                return 1
            if v in ("0", "false", "normal", "cn", "n"):
                return 0
    # fallback: many PAB rows encode anomaly in the image path (wentrong/ vs goal/)
    img = str(ann.get("image", "")).lower()
    if "wentrong" in img or "anomaly" in img:
        return 1
    if "goal" in img or "normal" in img:
        return 0
    return None  # unknown


def scene_key(ann):
    # group key to spread sampling (prefer a video/scene id, else the image dir)
    for k in ("video_id", "vid", "scene", "clip", "source"):
        if k in ann:
            return str(ann[k])
    return os.path.dirname(str(ann.get("image", "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    anns = read_anns(sorted(glob.glob(args.ann_glob)))
    print(f"loaded {len(anns)} train anns")
    for a in anns:
        a["anomaly"] = is_anomaly(a)

    pos = [a for a in anns if a["anomaly"] == 1]
    neg = [a for a in anns if a["anomaly"] == 0]
    unk = [a for a in anns if a["anomaly"] is None]
    print(f"anomaly={len(pos)} normal={len(neg)} unknown={len(unk)}")
    if not pos or not neg:                       # attribute parse failed -> fall back to random, label all 0
        print("WARN: could not split by anomaly -> random sample, anomaly label set 0 (fix is_anomaly()).")
        random.shuffle(anns); sub = anns[:args.n]
        for a in sub: a["anomaly"] = 0
    else:
        half = args.n // 2
        def spread(rows, k):
            by = defaultdict(list)
            for a in rows: by[scene_key(a)].append(a)
            for v in by.values(): random.shuffle(v)
            out, buckets = [], list(by.values()); random.shuffle(buckets)
            i = 0
            while len(out) < k and any(buckets):
                b = buckets[i % len(buckets)]
                if b: out.append(b.pop())
                i += 1
                if i % len(buckets) == 0: buckets = [b for b in buckets if b]
            return out[:k]
        sub = spread(pos, half) + spread(neg, args.n - half)
        random.shuffle(sub)

    # verify hard_i present (CMP ID-hard-pair)
    miss = sum(1 for a in sub if not a.get("hard_i"))
    print(f"subset={len(sub)}  rows missing hard_i: {miss} (those fall back to in-batch negs)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(sub, open(args.out, "w", encoding="utf-8"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
