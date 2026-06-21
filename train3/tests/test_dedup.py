import numpy as np
from PIL import Image

from startv4.data.dedup import dedup_keep_mask, dhash, hamming
from startv4.eval.distractor_val import build_distractor_index


def _img(seed):
    a = (np.random.default_rng(seed).random((40, 40, 3)) * 255).astype("uint8")
    return Image.fromarray(a)


def test_dhash_identical_is_zero():
    im = _img(0)
    assert hamming(dhash(im), dhash(im.copy())) == 0


def test_dedup_removes_dup_keeps_distinct():
    a, b = _img(1), _img(2)
    keep = dedup_keep_mask([dhash(a.copy()), dhash(b)], [dhash(a)], threshold=5)
    assert keep == [False, True]


def test_dedup_empty_refs_keeps_all():
    assert dedup_keep_mask([1, 2, 3], [], threshold=5) == [True, True, True]


def test_build_distractor_index_drops_overlap(tmp_path):
    def save(p, seed):
        _img(seed).save(p)

    gtd, dis, tg = tmp_path / "gt", tmp_path / "dis", tmp_path / "tg"
    for d in (gtd, dis, tg):
        d.mkdir()
    save(gtd / "g0.png", 10)
    save(tg / "t0.png", 99)               # a test-gallery image
    save(dis / "d_dup.png", 99)           # identical to test gallery -> must drop
    save(dis / "d_ok.png", 50)            # distinct -> keep
    idx = build_distractor_index(
        [str(gtd / "g0.png")],
        [str(dis / "d_dup.png"), str(dis / "d_ok.png")],
        [str(tg / "t0.png")],
    )
    assert idx["removed"] == 1
    assert any("d_ok" in p for p in idx["distractors"])
    assert all("d_dup" not in p for p in idx["distractors"])
