"""Import-safety for CLI scripts + a full Phase 0 -> 1 -> 2 run on the dummy backbone."""
import importlib
from pathlib import Path

import numpy as np
from PIL import Image

from startv4.config import load_config
from startv4.data.dataset import DummyTokenizer
from startv4.eval.distractor_val import build_distractor_index
from startv4.infer.encode import (
    average_features,
    encode_captions,
    encode_image_paths,
    load_embeddings,
    save_embeddings,
)
from startv4.infer.pipeline import evaluate_with_pipeline
from startv4.models.siglip_retrieval import build_siglip

ROOT = Path(__file__).resolve().parents[1]


def test_all_scripts_import_with_main():
    for mod in (
        "startv4.scripts.build_distractor_val",
        "startv4.scripts.encode_cache",
        "startv4.scripts.run_phase2",
        "startv4.train.train_siglip",
    ):
        assert hasattr(importlib.import_module(mod), "main")


def _img(p, seed):
    Image.fromarray((np.random.default_rng(seed).random((40, 40, 3)) * 255).astype("uint8")).save(p)


def test_phase0_1_2_end_to_end(tmp_path):
    cfg = load_config(ROOT / "configs" / "_test_dummy.yaml")
    model = build_siglip(cfg)
    tok = DummyTokenizer(max_len=16)

    # ---- Phase 0: build distractor-val with perceptual-hash de-dup ----
    gtd, dis, tg = tmp_path / "gt", tmp_path / "dis", tmp_path / "tg"
    for d in (gtd, dis, tg):
        d.mkdir()
    gt_paths = []
    for i in range(4):
        p = gtd / f"g{i}.png"
        _img(p, 100 + i)
        gt_paths.append(str(p))
    _img(tg / "t0.png", 100)            # == g0  -> any distractor copy must be dropped
    _img(dis / "leak.png", 100)         # leaked test image
    dis_paths = [str(dis / "leak.png")]
    for j in range(6):
        p = dis / f"d{j}.png"
        _img(p, 500 + j)
        dis_paths.append(str(p))
    idx = build_distractor_index(gt_paths, dis_paths, [str(tg / "t0.png")])
    assert idx["removed"] == 1
    assert all("leak" not in p for p in idx["distractors"])

    # ---- Phase 1: encode at two "scales" -> cache (.pt) ----
    captions = [f"a person number {i} doing action {i}" for i in range(4)]
    for scale in (32, 32):  # same size twice is fine for the TTA smoke
        save_embeddings(
            tmp_path / f"siglip_{scale}.pt",
            query=encode_captions(model, captions, tok, "cpu"),
            gt=encode_image_paths(model, idx["gt"], scale, "cpu", batch_size=2),
            distractor=encode_image_paths(model, idx["distractors"], scale, "cpu", batch_size=2),
        )
    cache_a = load_embeddings(tmp_path / "siglip_32.pt")
    assert cache_a["gt"].shape[0] == 4 and cache_a["query"].shape[0] == 4

    # ---- Phase 2: TTA average + ensemble + QE/k-reciprocal -> metrics ----
    files = [tmp_path / "siglip_32.pt", tmp_path / "siglip_32.pt"]
    q = average_features([load_embeddings(f)["query"] for f in files])
    gt = average_features([load_embeddings(f)["gt"] for f in files])
    ds = average_features([load_embeddings(f)["distractor"] for f in files])
    metrics, score = evaluate_with_pipeline(q, gt, ds, fuse="rrf", use_qe_kr=True, k1=4, k2=2)
    assert score.shape == (4, 4 + len(idx["distractors"]))
    for k in ("R@1", "R@5", "mAP"):
        assert 0.0 <= metrics[k] <= 1.0
