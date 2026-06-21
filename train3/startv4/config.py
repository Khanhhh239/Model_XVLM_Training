"""Tiny YAML config with attribute access + dotted overrides.

    cfg = load_config("configs/siglip_a100_1m.yaml",
                      parse_overrides(["train.batch_size=128", "optim.lr=5e-5"]))
    cfg.train.batch_size            # -> 128
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict that also supports attribute access, recursively."""

    def __init__(self, d: dict | None = None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    __setattr__ = dict.__setitem__  # type: ignore[assignment]

    def get_path(self, dotted: str, default: Any = None) -> Any:
        d: Any = self
        for p in dotted.split("."):
            if not isinstance(d, dict) or p not in d:
                return default
            d = d[p]
        return d


def _coerce(v: str) -> Any:
    """'128' -> 128, '5e-5' -> 5e-05, 'true' -> True, 'none' -> None, else str."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"override must be key=value, got: {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def apply_overrides(cfg: Cfg, overrides: dict[str, Any]) -> Cfg:
    for k, v in overrides.items():
        parts = k.split(".")
        d: Any = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = Cfg()
            d = d[p]
        d[parts[-1]] = v
    return cfg


def to_plain(cfg: Any) -> Any:
    """Recursively convert Cfg -> plain dict (so checkpoints don't pickle the Cfg class)."""
    if isinstance(cfg, dict):
        return {k: to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_plain(v) for v in cfg]
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Cfg:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Cfg(data)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg
