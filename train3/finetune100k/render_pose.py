"""Render ViTPose keypoints (already extracted by the data team in train_30k_hard_vitpose.json)
to COCO-17 skeleton images, so CMP's pose_block can consume them. Output mirrors the image path:
  <out_dir>/<image rel path>.webp   (e.g. out/imgs_0/goal/31.webp)
Fast (pure drawing) — ~minutes for 35,677 on CPU.

Run (on the extracted data team folder):
  python render_pose.py --vitpose .../train_30k_hard_vitpose.json --out /kaggle/working/pose --size 384
"""
import argparse, json, os
import numpy as np
import cv2

# COCO-17 skeleton edges + per-edge colors
SK = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12),(5,11),(6,12),(0,1),(0,2),(1,3),(2,4),(0,5),(0,6)]
COL = [(0,255,0)]*8 + [(0,128,255)]*4 + [(255,0,0)]*6


def get_kps(item):
    """Return 17x3 keypoints from a vitpose item (handles instances[] or flat keypoints)."""
    insts = item.get("instances")
    kp = None
    if insts:
        kp = insts[0].get("keypoints")
    kp = kp or item.get("keypoints")
    if kp is None:
        return None
    a = np.asarray(kp, dtype=np.float32).reshape(-1, 3)   # [17,3] = x,y,score
    return a if a.shape[0] >= 17 else None


def render(kps, w, h, size, conf=0.3):
    canvas = np.zeros((h, w, 3), np.uint8)
    pts = kps[:, :2]; sc = kps[:, 2]
    for (a, b), c in zip(SK, COL):
        if a < len(pts) and b < len(pts) and sc[a] > conf and sc[b] > conf:
            cv2.line(canvas, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), c, 3)
    for i, p in enumerate(pts):
        if sc[i] > conf:
            cv2.circle(canvas, tuple(p.astype(int)), 3, (255, 255, 255), -1)
    if (w, h) != (size, size):
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitpose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    data = json.load(open(args.vitpose, encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    items = items.values() if isinstance(items, dict) else items
    n, miss = 0, 0
    for it in items:
        rel = it["image"]                                # e.g. 'imgs_0/full/1040.webp'
        op = os.path.join(args.out, rel)
        if os.path.exists(op):
            n += 1; continue
        kps = get_kps(it)
        w, h = it.get("width", args.size), it.get("height", args.size)
        canvas = render(kps, w, h, args.size) if kps is not None else np.zeros((args.size, args.size, 3), np.uint8)
        if kps is None: miss += 1
        os.makedirs(os.path.dirname(op), exist_ok=True)
        cv2.imwrite(op, canvas)
        n += 1
        if n % 5000 == 0: print(f"{n} rendered")
    print(f"DONE: {n} pose images -> {args.out} (no-keypoint: {miss})")


if __name__ == "__main__":
    main()
