import importlib

from startv4.data.grounding import DummyBoxDetector, extract_noun_phrases, label_boxes


def test_extract_noun_phrases_drops_stopwords():
    ph = extract_noun_phrases("A man in a red shirt is falling near the trash can")
    assert "man" in ph and "shirt" in ph and "trash" in ph
    assert "the" not in ph and "is" not in ph and "a" not in ph


def test_extract_dedup_and_limit():
    ph = extract_noun_phrases("dog dog dog cat cat bird", max_phrases=2)
    assert ph == ["dog", "cat"]


def test_label_boxes_with_dummy_detector():
    boxes = label_boxes("img.jpg", "a person holding a bottle", DummyBoxDetector(), conf=0.35)
    assert len(boxes) >= 1
    for b in boxes:
        assert len(b["bbox"]) == 4 and 0.0 <= b["conf"] <= 1.0


def test_phase3_scripts_import_with_main():
    for mod in (
        "startv4.scripts.build_box_pseudolabels",
        "startv4.train.train_xvlm",
    ):
        assert hasattr(importlib.import_module(mod), "main")
