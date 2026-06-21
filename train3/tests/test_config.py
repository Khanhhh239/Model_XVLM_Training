from pathlib import Path

from startv4.config import Cfg, apply_overrides, load_config, parse_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_attribute_access_and_nested():
    cfg = Cfg({"a": 1, "b": {"c": 2}})
    assert cfg.a == 1
    assert cfg.b.c == 2
    assert isinstance(cfg.b, Cfg)


def test_parse_and_coerce():
    ov = parse_overrides(["train.batch_size=128", "optim.lr=5e-5", "x.flag=true", "x.none=none"])
    assert ov["train.batch_size"] == 128
    assert abs(ov["optim.lr"] - 5e-5) < 1e-12
    assert ov["x.flag"] is True
    assert ov["x.none"] is None


def test_apply_overrides_creates_path():
    cfg = Cfg({"a": {"b": 1}})
    apply_overrides(cfg, {"a.b": 9, "a.c.d": 5})
    assert cfg.a.b == 9
    assert cfg.a.c.d == 5


def test_load_test_config():
    cfg = load_config(ROOT / "configs" / "_test_dummy.yaml")
    assert cfg.model.name == "dummy"
    assert cfg.train.batch_size == 8
    assert cfg.optim.get_path("ema") is True
