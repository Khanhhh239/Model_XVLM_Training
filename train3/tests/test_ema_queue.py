import torch
import torch.nn as nn

from startv4.models import ModelEMA, NegativeQueue


def test_ema_tracks_weights():
    m = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        m.weight.zero_()
    ema = ModelEMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.add_(1.0)  # weights now all 1, shadow still 0
    ema.update(m)
    # shadow = 0.5*0 + 0.5*1 = 0.5
    assert torch.allclose(ema.state_dict()["weight"], torch.full((3, 3), 0.5), atol=1e-6)
    ema.copy_to(m)
    assert torch.allclose(m.weight, torch.full((3, 3), 0.5), atol=1e-6)


def test_queue_enqueue_and_wraparound():
    q = NegativeQueue(dim=4, size=8)
    q.enqueue(torch.randn(5, 4))
    assert int(q.filled.item()) == 5
    assert int(q.ptr.item()) == 5
    q.enqueue(torch.randn(5, 4))  # wraps: 8 - 5 = 3 at end, 2 at start
    assert int(q.filled.item()) == 8
    assert int(q.ptr.item()) == 2
    out = q.get()
    assert out.shape == (8, 4)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-4)
