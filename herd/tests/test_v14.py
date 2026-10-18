"""v14 (A) on made-up data: helpers, scores, the two transformers' training, the onboarding steps."""
import argparse
import collections
import json
import os
import sys
import tempfile

import numpy as np
import pytest

import _paths  # noqa: F401

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v14"))

import metrics  # noqa: E402
import v14common as C  # noqa: E402


def test_names_and_labels():
    assert C.unix_from_name("1690304400_0001.jpg") == 1690304400.0
    t = C.unix_from_name("cam_1_12-30-15.jpg")
    assert t is not None and abs(C.local_hour(t) - 12.5042) < 1e-3
    boxes = C.parse_id_boxes("3 0.5 0.5 0.2 0.1\n7 2240 1400 448 280\nbad line\n")
    assert boxes[0][0] == 3 and np.allclose(boxes[0][1], [0.4, 0.45, 0.6, 0.55])
    assert boxes[1][0] == 7 and np.allclose(boxes[1][1], [0.45, 0.45, 0.55, 0.55])


def test_mcc_names():
    import data_mcc
    rows = data_mcc.parse(["x/MultiCamCows2024Root/2023Aug14/001/cam1_track3_000010.jpg",
                           "x/MultiCamCows2024Root/2023Aug14/001/cam1_track3_000015.jpg",
                           "x/MultiCamCows2024Root/2023Aug15/012/seqA/img_2.png",
                           "x/MultiCamCows2024Root/videos/cam1.mp4", "readme.txt"])
    assert len(rows) == 3
    a, b, c = rows
    assert (a["day"], a["cow"], a["seq"], a["cam"]) == ("2023Aug14", 1, "cam1_track3", 1)
    assert b["t"] - a["t"] == pytest.approx(1.0)                  # 5 frames at 5 Hz
    assert (c["cow"], c["seq"]) == (12, "seqA")


def test_scores():
    r = metrics.id_prf(["a", None, "b", "x"], ["a", "b", "b", "c"])
    assert (r["right"], r["answered"]) == (2, 3)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(0.5)
    f = metrics.idf1(["a", "a", "b", "b"], [1, 1, 2, None], n_pred=4)
    assert f["idtp"] == 3 and f["idfn"] == 1 and f["idfp"] == 1 and f["idf1"] == pytest.approx(0.75)
    al = metrics.album_quality(["A1", "A1", "A2", "A3", None], ["c1", "c1", "c2", "c2", "c3"], 3)
    assert al["album_cows"] == 3 and al["duplicates"] == 1 and al["coverage"] == pytest.approx(2 / 3)
    co = metrics.continuity([("c", 0, "cam1", 5), ("c", 15, "cam1", 5), ("c", 30, "cam2", 5), ("c", 45, "cam2", 9)])
    assert co["kept_same_camera"] == pytest.approx(0.5) and co["kept_camera_change"] == 1.0
    q = np.eye(3, dtype=np.float32)
    rr = metrics.retrieval(q, ["a", "b", "c"], q, ["a", "b", "c"])
    assert rr["top1"] == 1.0 and rr["mAP"] == 1.0


def fake_mcc(folder, days=4, cows=8, seqs=3, frames=10, dim=48, seed=0):
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(cows, dim))
    rows, feats = [], []
    for d in range(days):
        for c in range(cows):
            for s in range(seqs):
                for f in range(frames):
                    feats.append(proto[c] + rng.normal(scale=0.6, size=dim))
                    rows.append({"name": f"{d}/{c}/{s}/{f}", "day": f"2023Aug1{d + 4}", "cow": c, "seq": f"s{s}",
                                 "t": f / 5.0, "cam": 1})
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "feats.npy"), np.array(feats, np.float16))
    with open(os.path.join(folder, "crops.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump({"encoder": "stub", "grid": 2, "crop": 224, "dim": dim}, open(os.path.join(folder, "meta.json"), "w"))


def test_idnet_trains_and_scores_a_new_day():
    import idnet
    with tempfile.TemporaryDirectory() as d:
        fake_mcc(os.path.join(d, "mcc"))
        a = argparse.Namespace(feats=os.path.join(d, "mcc"), out=os.path.join(d, "idnet"), max_error=0.05, epochs=6,
                               patience=6, P=8, min_len=3, max_len=8, drop=0.1, noise=0.01, d=32, layers=1, heads=2,
                               dropout=0.0, lr=2e-3, wd=0.0, temperature=0.1, seed=0)
        idnet.cmd_train(a)
        ev = json.load(open(os.path.join(d, "idnet", "eval_mcc.json")))
        assert ev["days"]["test"] == "2023Aug17" and ev["test_day"]["queries"] == 8 * 3
        assert ev["test_day"]["top1"] >= 0.8                          # easy made-up cows
        assert ev["test_id_with_not_sure"]["f1"] is not None
        ck = torch.load(os.path.join(d, "idnet", "idnet.pt"), weights_only=False)
        assert "nan" in ck and "enrol_threshold" in ck


def fake_mmcows(folder, cows=6, dim=48, sdim=16, keys=40, seed=0):
    """Two cameras over overlapping halves of one pen, a labelled frame every 15 s."""
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(cows, dim))
    pos = rng.uniform(0.1, 0.9, size=(cows, 2))
    frames, rows, feats, scenes = [], [], [], []
    for k in range(keys):
        pos = np.clip(pos + rng.normal(scale=0.02, size=pos.shape), 0.05, 0.95)
        split = "window" if k >= keys - 10 else "train"
        for cam, (x0, x1) in (("cam_1", (0.0, 0.6)), ("cam_2", (0.4, 1.0))):
            boxes = []
            for c in range(cows):
                if x0 <= pos[c, 0] <= x1:
                    cx = (pos[c, 0] - x0) / (x1 - x0)
                    b = [max(0, cx - 0.05), max(0, pos[c, 1] - 0.04), min(1, cx + 0.05), min(1, pos[c, 1] + 0.04)]
                    boxes.append([c, b])
                    rows.append({"frame": len(frames), "cow": c, "box": b})
                    feats.append(proto[c] + rng.normal(scale=0.5, size=dim))
            frames.append({"cam": cam, "ts": 1690300000.0 + 15 * k, "split": split, "boxes": boxes, "path": ""})
            scenes.append(rng.normal(size=(144, sdim)))
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, "assoc.npz"), box_feats=np.array(feats, np.float16), scene=np.array(scenes, np.float16))
    for name, data in (("frames.jsonl", frames), ("assoc_rows.jsonl", rows)):
        with open(os.path.join(folder, name), "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")


def test_assoc_trains_on_real_and_virtual_cameras():
    import assoc
    with tempfile.TemporaryDirectory() as d:
        fake_mmcows(os.path.join(d, "mm"))
        data = assoc.Data(os.path.join(d, "mm"))
        import random
        e = data.real("train", data.keys["train"][5], random.Random(0))
        assert (e["kind"] == assoc.CUR).any() and (e["kind"] == assoc.SCENE).sum() == 2 * 64
        assert set(np.unique(e["slot"])) <= set(range(assoc.SLOTS))
        v = data.virtual("train", data.keys["train"][5], random.Random(1))
        ov = v["overlap"][v["kind"] == assoc.SCENE]
        assert ((ov == 0) | (ov == 1) | (ov == -1)).all() and (ov == 1).any()
        a = argparse.Namespace(data=os.path.join(d, "mm"), out=os.path.join(d, "assoc"), epochs=3, patience=3, batch=4,
                               virtual=0.5, d=32, layers=1, heads=2, dropout=0.0, lr=2e-3, wd=0.0, temperature=0.1,
                               w_else=0.5, w_overlap=0.5, seed=0)
        assoc.cmd_train(a)
        ev = json.load(open(os.path.join(d, "assoc", "eval_assoc.json")))
        assert ev["test_window"]["cross_camera"]["pairs"] > 0
        assert ev["baseline_plain_dinov2"]["track_top1"] is not None


def test_onboarding_steps():
    import system
    rng = np.random.default_rng(0)
    proto = rng.normal(size=(3, 16))
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    recs = []
    for m in range(10):                                   # 3 cows, cow 0 split over two tracks
        for tid, c in ((1, 0), (2, 1), (3, 2), (4, 0)):
            fp = proto[c] + rng.normal(scale=0.05, size=16)
            recs.append((m, tid, fp / np.linalg.norm(fp), 1.0, 0.5, 0.1))
    P, ids, ta = system.enrol(recs, thr=0.8, min_minutes=5)
    assert len(set(ids.tolist())) == 3 and ta[1] == ta[4]   # the two tracks of cow 0: one album cow
    later = [(m + 10, tid, fp, q1, q2, a) for m, tid, fp, q1, q2, a in recs]
    names = system.identify(later, P, ids, None, 0.5)
    assert names[(2, 10)] == ta[2] and names[(3, 19)] == ta[3]
    assert sum(1 for k, v in names.items() if len(k) == 2 and k[1] == 12 and v == ta[1]) == 1   # one cow, one track

    class T:
        rows = {7: {"sightings": [(0, "cam_1", [0.1, 0.1, 0.3, 0.3], None, 0.04, 1.0)], "last": 0}}
    out = system.iou_assign([{"cam": "cam_1", "box": [0.11, 0.1, 0.31, 0.3]}, {"cam": "cam_2", "box": [0.1, 0.1, 0.3, 0.3]}],
                            [7], T)
    assert out == [7, None]


def test_onboarding_assoc_step_runs():
    import assoc
    import system
    am = assoc.build(48, 16, d=32, layers=1, heads=2).eval()
    tracks = system.Tracks(20)
    rng = np.random.default_rng(1)
    cur = [{"cam": c, "box": [0.1 + 0.1 * i, 0.1, 0.2 + 0.1 * i, 0.2], "conf": 0.9, "feat": rng.normal(size=48).astype(np.float32)}
           for i, c in enumerate(["cam_1", "cam_2", "cam_1"])]
    import collections as co
    mem = co.deque(maxlen=16)
    scenes = {"cam_1": rng.normal(size=(64, 16)).astype(np.float32), "cam_2": rng.normal(size=(64, 16)).astype(np.float32)}
    a = argparse.Namespace(history=2, memory_threshold=0.85)
    out = system.assoc_assign(am, cur, [], tracks, mem, scenes, {"cam_1": 0, "cam_2": 1}, 100.0, -1.0, "cpu", a)
    assert len(out) == 3
    assert out[0] is None or out[0] != out[2]            # two boxes on one camera are never one cow


def test_scoring_and_results_page():
    import idnet
    import summarize
    import system
    with tempfile.TemporaryDirectory() as d:
        t0 = 1690304400.0                                   # 12:00 local on 25 July
        with open(os.path.join(d, "labels.jsonl"), "w") as fh:
            for k in range(0, 600, 15):                     # 10 minutes, 2 cows on cam_1
                fh.write(json.dumps({"cam": "cam_1", "ts": t0 + k, "boxes": [[1, [0.1, 0.1, 0.3, 0.3]],
                                                                             [2, [0.6, 0.6, 0.8, 0.8]]]}) + "\n")
        det_log = [(t0 + k, "cam_1", [0.1, 0.1, 0.3, 0.3], 11, 0.9) for k in range(0, 600, 2)]
        det_log += [(t0 + k, "cam_1", [0.6, 0.6, 0.8, 0.8], 12, 0.9) for k in range(0, 600, 2)]
        names = {(11, m): "A01" for m in range(10)}
        names.update({(12, m): "A01" for m in range(5, 10)})   # cow 2 named wrong after minute 5
        a = argparse.Namespace(data=d, start=12.0, end=12.17, step=2, match_iou=0.3)
        res = system.score(a, det_log, {11: "A01", 12: "A02"}, names, t0, t0 + 300, {"cam_1"}, metrics)
        assert res["detector"]["recall"] == 1.0 and res["tracking"]["idf1"] == 1.0
        assert res["album"]["album_cows"] == 2 and res["album"]["purity"] == 1.0
        idr = res["identification"]["with_not_sure"]
        assert idr["recall"] == pytest.approx(0.5) and idr["precision"] == pytest.approx(0.5)
        w = os.path.join(d, "work")
        os.makedirs(os.path.join(w, "system"))
        res["settings"] = {"setup_min": 5, "start": 12.0, "end": 12.17}
        json.dump(res, open(os.path.join(w, "system", "system_assoc.json"), "w"))
        fake_mcc(os.path.join(d, "mcc"))
        idnet.cmd_train(argparse.Namespace(feats=os.path.join(d, "mcc"), out=os.path.join(w, "idnet"), max_error=0.05,
                                           epochs=1, patience=1, P=8, min_len=3, max_len=8, drop=0.1, noise=0.01, d=32,
                                           layers=1, heads=2, dropout=0.0, lr=2e-3, wd=0.0, temperature=0.1, seed=0))
        summarize.main(["--work", w, "--out", os.path.join(d, "res")])
        page = open(os.path.join(d, "res", "RESULTS.md")).read()
        assert "ID F1" in page and "IDF1" in page
