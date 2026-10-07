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


def test_motion_branch_and_old_checkpoints():
    import os
    import tempfile
    from motion import DIM
    m = HerdModel(48, d=32, layers=1, heads=2, motion_dim=DIM)
    x, t = torch.randn(2, 20, 48), torch.arange(20).float().repeat(2, 1) / 25
    valid = torch.ones(2, 20, dtype=torch.bool)
    with_m = m.temporal(x, t, valid, motion=torch.randn(2, DIM))
    without = m.temporal(x, t, valid)                  # a cow seen too briefly: zeros, still an answer
    assert with_m["rumination"].shape == without["rumination"].shape == (2,)
    assert with_m["activity"].shape == (2, 3)
    # a model from before motion loads and runs as before
    old = HerdModel(48, d=32, layers=1, heads=2)
    with tempfile.TemporaryDirectory() as d:
        old.save(os.path.join(d, "m.pt"))
        back, _ = HerdModel.load(os.path.join(d, "m.pt"))
    assert back.motion_dim == 0 and back.temporal.motion is None
    assert back.temporal(x, t, valid)["rumination"].shape == (2,)


def test_motion_features_find_a_rhythm():
    import numpy as np
    from motion import BANDS, DIM, GRID, motion_features
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (224, 224, 3)).astype(np.float32)

    def burst(freq):
        out = []
        for k in range(175):
            img = base + rng.normal(0, 6, base.shape)
            if freq:
                img[150:175, 30:60] += 25 * np.sin(2 * np.pi * freq * k / 25)   # a jaw, ~1 chew a second
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return np.stack(out)

    still, ok1 = motion_features(burst(0))
    chew, ok2 = motion_features(burst(1.1))
    assert ok1 and ok2 and still.shape == chew.shape == (DIM,)
    band = [i for i, (lo, hi) in enumerate(BANDS) if lo <= 1.1 < hi][0]
    top = 2 * GRID * GRID * len(BANDS)                  # max over cells of each band's share
    assert chew[top + band] > 0.6 > still[top + band]
    short = np.zeros(175, bool)
    short[:40] = True                                   # 1.6 s seen: too short to tell
    assert motion_features(burst(1.1), valid=short)[1] is False
