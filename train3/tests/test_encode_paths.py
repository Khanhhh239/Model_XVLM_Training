from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from startv4.config import load_config
from startv4.data.dataset import DummyTokenizer
from startv4.infer.encode import average_features, encode_captions, encode_image_paths
from startv4.models.siglip_retrieval import build_siglip

ROOT = Path(__file__).resolve().parents[1]


def test_encode_image_paths_and_captions(tmp_path):
    cfg = load_config(ROOT / "configs" / "_test_dummy.yaml")
    m = build_siglip(cfg)
    paths = []
    for i in range(5):
        p = tmp_path / f"i{i}.png"
        Image.new("RGB", (40, 40), (i * 20, 80, 60)).save(p)
        paths.append(str(p))
    feat = encode_image_paths(m, paths, image_size=32, device="cpu", batch_size=2)
    assert feat.shape == (5, 64)
    assert torch.allclose(feat.norm(dim=1), torch.ones(5), atol=1e-4)

    qf = encode_captions(m, ["a b c", "d e"], DummyTokenizer(max_len=16), "cpu")
    assert qf.shape == (2, 64)


def test_average_features_identity_and_norm():
    f = F.normalize(torch.randn(4, 8), dim=1)
    avg = average_features([f, f.clone()])
    assert torch.allclose(avg, f, atol=1e-5)
    mixed = average_features([F.normalize(torch.randn(4, 8), dim=1) for _ in range(3)])
    assert torch.allclose(mixed.norm(dim=1), torch.ones(4), atol=1e-4)
