"""Rumination from video: label maps, 7 s windows, the splits, the scores, and a short training run."""
import _paths  # noqa: F401
import json
import os
import tempfile

import numpy as np

import rumination as r


def test_labels_and_splits():
    assert r.label_of("kaggle", "ruminate") == (0, -1)
    assert r.label_of("cvb", "ruminating-lying") == (0, r.LYING)
    assert r.label_of("cvb", "grazing") == (r.CLASSES.index("feeding"), r.STANDING)
    assert r.label_of("cvb", "hidden") == (-1, -1)
    assert r.kaggle_split("00000000514000000_3_clip_7") == ("00000000514000000", "val")
    assert r.kaggle_split("00000000082000000_5_clip_1") == ("00000000082000000", "train")
    assert r.kaggle_split("15_2_clip_0")[1] == "val" and r.kaggle_split("16_2_clip_0")[1] == "train"


def test_windows_agree_and_seen():
    t = np.arange(375) / 25.0                                    # 15 s
    lab = np.zeros(375, int)
    lab[200:] = 3                                                # rumination, then resting from 8 s
    post = np.full(375, 1)
    w = r.windows(t, lab, post)
    assert w[0][:4] == (0.0, 7.0, 0, 1)                          # all ruminating
    assert all(x[4] >= r.MIN_AGREE for x in w)
    assert w[-1][2] == 3
    hidden = np.zeros(375, bool)                                 # nobody seen: no window
    assert r.windows(t, lab, post, hidden) == []
    short = r.windows(np.arange(100) / 25.0, np.zeros(100, int), np.full(100, -1))
    assert len(short) == 1 and short[0][2] == 0 and short[0][3] == -1


def test_cvb_tracks():
    ann = {"images": [{"id": 1, "width": 100, "height": 50, "file_name": "x/img_00002.jpg"},
                      {"id": 2, "width": 100, "height": 50, "file_name": "x/img_00001.jpg"}],
           "annotations": [{"image_id": 1, "bbox": [10, 5, 20, 10], "attributes": {"track_id": 3, "behavior": "grazing"}},
                           {"image_id": 2, "bbox": [0, 0, 10, 10], "attributes": {"track_id": 3, "behavior": "walking"}}]}
    tr = r.cvb_tracks(ann)
    assert list(tr) == [3] and [n for n, _, _ in tr[3]] == [1, 2]
    assert tr[3][1][1] == [0.1, 0.1, 0.3, 0.3] and tr[3][1][2] == "grazing"


def test_scores_auc():
    p = np.array([0.9, 0.8, 0.3, 0.2])
    s = r.scores(p, np.array([True, False, True, False]), 0.5)
    assert abs(s["auc"] - 0.75) < 1e-9 and s["at_threshold"]["tp"] == 1 and s["at_threshold"]["fp"] == 1
    assert r.scores(p, np.array([True, True, False, False]), 0.5)["best"]["f1"] == 1.0


def test_train_on_synthetic_sequences():
    """Sequences whose frames carry the label in one dimension: the transformer must find it."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        feat, out = os.path.join(d, "feat"), os.path.join(d, "model")
        for src in ("kaggle", "cvb"):
            os.makedirs(os.path.join(feat, src))
        json.dump({"encoder": "x", "grid": 2, "crop": 224, "margin": 0.1, "dim": 16}, open(os.path.join(feat, "features.json"), "w"))
        for k in range(40):
            src = ("kaggle", "cvb")[k % 2]
            c = (0, 3)[(k // 2) % 2]
            t = np.arange(200, dtype=np.float32) / 25
            emb = rng.normal(size=(200, 16)).astype(np.float16)
            emb[:, 0] += 3.0 if c == 0 else -3.0
            lab, post, seen = np.full(200, c), np.full(200, 1), np.ones(200, bool)
            wins = r.windows(t, lab, post, seen)
            r._save_seq(os.path.join(feat, src, f"s{k}.npz"), emb, t, lab, post, seen, wins,
                        np.zeros((len(wins), 201), np.float32),
                        {"source": src, "seq": f"s{k}", "group": "g", "split": "val" if k % 5 == 0 else "train"})
        r.main(["train", "--feat", feat, "--out", out, "--epochs", "3", "--steps", "15", "--batch", "8",
                "--d", "32", "--layers", "1"])
        ev = json.load(open(os.path.join(out, "eval.json")))
        assert ev["val"]["all"]["best"]["f1"] > 0.9
        assert os.path.exists(os.path.join(out, "rumination.pt"))
