"""Unit tests for the phrase-grounded box head + loader + multi-box transform (group D / #3)."""
import json

import torch
from PIL import Image

from star.data.dataset import load_phrase_boxes
from star.data.transforms import LHPTransform
from star.models.heads import PhraseBoxHead


def test_phrase_box_head_forward_shape_and_range():
    head = PhraseBoxHead(cross_dim=32)
    out = head(torch.randn(5, 32))
    assert out.shape == (5, 4)
    assert (out >= 0).all() and (out <= 1).all()           # sigmoid output


def test_phrase_box_loss_perfect_zero_and_masking():
    pred = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]])
    assert PhraseBoxHead.compute_loss(pred, pred.clone()).item() < 1e-5
    # second row garbage but masked out -> still ~0
    bad = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.9, 0.9, 0.05, 0.05]])
    assert PhraseBoxHead.compute_loss(bad, pred, mask=torch.tensor([True, False])).item() < 1e-5
    # no valid box -> exactly 0
    assert PhraseBoxHead.compute_loss(bad, pred, mask=torch.tensor([False, False])).item() == 0.0


def test_phrase_box_loss_far_greater_than_near():
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    near = PhraseBoxHead.compute_loss(torch.tensor([[0.52, 0.52, 0.2, 0.2]]), gt)
    far = PhraseBoxHead.compute_loss(torch.tensor([[0.1, 0.1, 0.1, 0.1]]), gt)
    assert far > near


def test_load_phrase_boxes_jsonl_topk_and_score(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text("\n".join([
        json.dumps({"image_id": "a", "boxes": [
            {"phrase": "man", "box": [0, 0, 10, 10], "score": 0.9},
            {"phrase": "dog", "box": [5, 5, 20, 20], "score": 0.4},
            {"phrase": "low", "box": [1, 1, 2, 2], "score": 0.1}]}),
        json.dumps({"image_id": "b", "boxes": [{"phrase": "x", "box": [0, 0, 5, 5], "score": 0.5}]}),
    ]), encoding="utf-8")
    d = load_phrase_boxes(str(p), topk=2, min_score=0.3)
    assert set(d) == {"a", "b"}
    assert len(d["a"]) == 2                                 # topk=2; low-score 0.1 filtered out
    assert d["a"][0]["phrase"] == "man"                     # sorted by score desc


def test_load_phrase_boxes_json_array(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps([{"image_id": "a",
                              "boxes": [{"phrase": "m", "box": [0, 0, 3, 3], "score": 0.8}]}]),
                 encoding="utf-8")
    d = load_phrase_boxes(str(p))
    assert "a" in d and len(d["a"]) == 1


def test_lhp_transform_carries_phrase_boxes():
    # enabled=False -> no crop -> phrase boxes just rescale with the image (deterministic geometry).
    t = LHPTransform(size=100, enabled=False)
    img = Image.new("RGB", (200, 200), (120, 100, 80))
    out = t(img, bbox=[20, 20, 180, 180], keypoints=None,
            extra_boxes=[[20, 20, 120, 120], [60, 60, 180, 180]])
    assert len(out) == 5                                    # (image, person, kpts, ph_box, ph_valid)
    image, _person, _kpts, ph_box, ph_valid = out
    assert image.shape == (3, 100, 100)
    assert ph_box.shape == (2, 4) and ph_valid.shape == (2,)
    assert ph_valid.sum().item() == 2                       # both boxes inside -> valid
    assert (ph_box >= 0).all() and (ph_box <= 1).all()
    # [20,20,120,120] @200 -> /200 -> center xywh (0.35, 0.35, 0.5, 0.5)
    assert torch.allclose(ph_box[0], torch.tensor([0.35, 0.35, 0.5, 0.5]), atol=2e-2)
