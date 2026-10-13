"""beef_bursts: one-cow clips -> herd bursts with labels, readable by train.Split."""
import json
import os
import tempfile

import numpy as np
import pytest

import _paths  # noqa: F401

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")


class StubEncoder(torch.nn.Module):
    """Stands in for DINOv2: a crop -> 12 numbers (its mean colour, repeated)."""
    out_dim = 12

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m = x.float().mean((1, 2)) / 255.0
        return m.repeat(1, 4)


def clip(path, seconds, fps=25, size=(96, 64), colour=(40, 120, 200)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(int(seconds * fps)):
        img = np.full((size[1], size[0], 3), colour, np.uint8)
        img[20:30, 10 + (k % 20):20 + (k % 20)] = 255
        w.write(img)
    w.release()


def test_build_writes_a_feature_folder_train_can_read():
    import beef_bursts as bb
    from train import Split
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "clips")
        for b in ("drink", "lie", "ruminate"):
            for k in range(3):
                clip(os.path.join(src, b, f"{b}_{k}.mp4"), 8)
        clip(os.path.join(src, "lie", "lie_short.mp4"), 3)          # too short: replaced, never stretched
        clip(os.path.join(src, "stand", "stand_0.mp4"), 7.5, fps=30)  # another frame rate: real timing kept
        out = os.path.join(d, "beef")
        import argparse
        a = argparse.Namespace(out=out, per_class=2, behaviours="drink,lie,ruminate,stand", seed=0, like=None,
                               encoder="stub", grid=2, crop=32, margin=0.0, burst_seconds=7.0, work=None,
                               local_dir=src)
        bb.cmd_build(a, encoder_factory=lambda meta: StubEncoder())
        meta = json.load(open(os.path.join(out, "meta.json")))
        assert meta["dim"] == 12 and meta["source"] == "beef"
        s = Split(out)
        by = {}
        for r in s.bursts:
            by.setdefault(r["behaviour"], []).append(r)
        assert {b: len(v) for b, v in by.items()} == {"drink": 2, "lie": 2, "ruminate": 2, "stand": 1}
        assert all("short" not in r["file"] for r in s.bursts)
        assert s.burst_feats.shape == (7, 175, 12)
        assert np.allclose(s.burst_times[0, :3], [0, 0.04, 0.08])
        assert s.burst_motion is not None and s.burst_motion.shape[0] == 7
        assert len(set(s.identity.tolist())) == 7                    # every clip its own cow
        r = {b: v[0] for b, v in by.items()}
        assert (r["drink"]["activity"], r["drink"]["rumination"], r["drink"]["posture"]) == (1, 0, -1)
        assert (r["lie"]["posture"], r["lie"]["activity"]) == (1, 2)
        assert (r["ruminate"]["rumination"], r["ruminate"]["posture"]) == (1, -1)
        assert r["stand"]["posture"] == 0
        # a second build reuses the per-clip cache instead of decoding again
        bb.cmd_build(a, encoder_factory=lambda meta: (_ for _ in ()).throw(AssertionError("re-encoded")))
        a2 = argparse.Namespace(out=out, work=None, local_dir=src, n=2)
        bb.cmd_sheet(a2)
        assert os.path.exists(os.path.join(out, "check.png"))
