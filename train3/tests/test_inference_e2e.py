import torch
import torch.nn.functional as F

from startv4.infer.pipeline import build_ranking, write_answer


def test_build_ranking_base_top1():
    g = F.normalize(torch.randn(12, 8), dim=1)
    q = g[[2, 5, 7]].clone()
    rank = build_ranking(q, g)
    assert rank.shape == (3, 12)
    assert rank[:, 0].tolist() == [2, 5, 7]


def test_build_ranking_qe_kr_keeps_gt_near_top():
    g = F.normalize(torch.randn(10, 8), dim=1)
    q = g[[1, 4, 8]].clone()
    rank = build_ranking(q, g, use_qe_kr=True, k1=4, k2=2)
    top3 = rank[:, :3].tolist()
    for i, gt in enumerate([1, 4, 8]):
        assert gt in top3[i]


def test_ensemble_rrf_runs():
    g = F.normalize(torch.randn(10, 8), dim=1)
    q = F.normalize(torch.randn(3, 8), dim=1)
    extra = q @ g.t() + 0.01 * torch.randn(3, 10)
    rank = build_ranking(q, g, extra_sims=[extra], fuse="rrf")
    assert rank.shape == (3, 10)


def test_write_answer(tmp_path):
    g = F.normalize(torch.randn(6, 8), dim=1)
    q = g[[0, 3]].clone()
    rank = build_ranking(q, g)
    ids = [f"g{i}" for i in range(6)]
    out = tmp_path / "answer.txt"
    write_answer(rank, ids, out, topk=4)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].split()[0] == "g0"
    assert len(lines[0].split()) == 4
