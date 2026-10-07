"""Shapes, the quality-weighted fingerprint and the masked losses."""
import torch

import _paths  # noqa: F401
from model import HerdModel, masked_ce, supcon


def test_forward_and_losses():
    m = HerdModel(48, d=32, layers=1, heads=2)
    x, t = torch.randn(3, 20, 48), torch.arange(20).float().repeat(3, 1) / 25
    valid = torch.ones(3, 20, dtype=torch.bool)
    valid[1, 10:] = False                     # a burst with hidden frames
    o = m.temporal(x, t, valid)
    assert o["fingerprint"].shape == (3, 512)
    assert torch.allclose(o["fingerprint"].norm(dim=-1), torch.ones(3), atol=1e-4)
    assert torch.allclose(o["weights"].sum(1), torch.ones(3), atol=1e-4)
    assert o["weights"][1, 10:].abs().max() < 1e-6      # hidden frames get no weight
    f = m.frame(torch.randn(5, 48))
    assert f["posture"].shape == (5, 2) and f["activity"].shape == (5, 3)
    assert masked_ce(f["posture"], torch.tensor([-1] * 5)).item() == 0.0   # no labels: head off
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    assert supcon(z, torch.tensor([0, 0, 1, 1])).item() > 0
    assert supcon(z, torch.tensor([0, 1, 2, 3])).item() == 0.0              # no positives
