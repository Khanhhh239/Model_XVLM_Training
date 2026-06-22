"""Tests for the full-1M/40GB + wrong-case-proof additions."""
import importlib
from pathlib import Path

import torch
import torch.nn.functional as F

from startv4.config import load_config
from startv4.eval.distractor_val import evaluate_by_category
from startv4.models.siglip_retrieval import build_siglip
from startv4.train.trainer import SiglipTrainer

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_by_category_groups_and_overall():
    q = F.normalize(torch.randn(6, 8), dim=1)
    gt = q.clone()  # perfect -> R@1 = 1 in every group
    dis = F.normalize(torch.randn(20, 8), dim=1)
    cats = ["#1", "#1", "#2", "#2", "#3", "#3"]
    res = evaluate_by_category(q, gt, dis, cats)
    assert {"#1", "#2", "#3", "__overall__"} <= set(res)
    assert res["#1"]["n"] == 2 and res["__overall__"]["n"] == 6
    assert res["#1"]["R@1"] == 1.0 and res["__overall__"]["R@1"] == 1.0


def test_siglip_filip_chunk_runs():
    cfg = load_config(ROOT / "configs" / "_test_dummy.yaml")
    cfg.optim["use_filip"] = True
    cfg.optim["w_filip"] = 0.5
    cfg.optim["filip_chunk"] = 2
    trainer = SiglipTrainer(build_siglip(cfg), cfg, "cpu")
    logs = trainer.train_step({
        "pixel_values": torch.randn(8, 3, 32, 32),
        "input_ids": torch.randint(1, 1000, (8, 16)),
        "attention_mask": torch.ones(8, 16, dtype=torch.long),
    })
    assert "filip" in logs and logs["filip"] == logs["filip"]  # present + not NaN


def test_full1m_and_wrongcase_configs_load():
    for name in ("siglip_full1m_a100_40g", "xvlm_full_wrongcase"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.model.name and cfg.optim is not None
    # the wrong-case X-VLM config turns the heads ON
    xc = load_config(ROOT / "configs" / "xvlm_full_wrongcase.yaml")
    assert xc.model.use_box and xc.model.use_pose and xc.model.use_filip


def test_eval_wrong_cases_script_has_main():
    assert hasattr(importlib.import_module("startv4.scripts.eval_wrong_cases"), "main")
