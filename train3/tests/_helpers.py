"""Build a tiny on-disk dataset (images + manifest) for tests -- no downloads."""
import json
from pathlib import Path

from PIL import Image


def make_dataset(tmp_path, n: int = 8):
    img_root = Path(tmp_path) / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        fn = f"img{i}.jpg"
        Image.new("RGB", (40, 40), ((i * 17) % 255, 100, 150)).save(img_root / fn)
        rows.append(
            {
                "image_path": fn,
                "caption": f"a person number {i} doing action {i % 3}",
                "video_id": f"vid{i // 2}",
                "bucket": "anomaly" if i % 2 else "normal",
                "keypoints": [[float(j), float(j), 0.9] for j in range(17)],
                "bbox": [0.1, 0.1, 0.5, 0.6],
                "image_id": f"id{i}",
                "pair_image_id": f"id{i ^ 1}",
                "split": "train",
            }
        )
    mani = Path(tmp_path) / "manifest.jsonl"
    mani.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return mani, img_root
