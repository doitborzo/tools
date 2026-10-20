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
                               dropout=0.0, lr=2e-3, wd=0.0, temperature=0.1, seed=0,
                               split="days", val_cows=3, test_cows=5, split_seed=0, time_aug=1.0)
        idnet.cmd_train(a)
        ev = json.load(open(os.path.join(d, "idnet", "eval_id.json")))
        assert ev["split"]["test"] == "2023Aug17" and ev["test"]["queries"] == 8 * 3
        assert ev["test"]["top1"] >= 0.8                              # easy made-up cows
        assert ev["test_id_with_not_sure"]["f1"] is not None
        ck = torch.load(os.path.join(d, "idnet", "idnet.pt"), weights_only=False)
        assert "nan" in ck and "enrol_threshold" in ck


def fake_mmcows(folder, cows=6, dim=48, sdim=16, keys=40, seed=0, train_every=1):
    """Two cameras over overlapping halves of one pen, a labelled frame every 15 s."""
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(cows, dim))
    pos = rng.uniform(0.1, 0.9, size=(cows, 2))
    frames, rows, feats, scenes = [], [], [], []
    for k in range(keys):
        pos = np.clip(pos + rng.normal(scale=0.02, size=pos.shape), 0.05, 0.95)
        split = "window" if k >= keys - 10 else "train"
        if split == "train" and k % train_every:
            continue                                          # training frames sparser, like MmCows' one a minute
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


def test_history_found_with_sparse_training_frames():
    import assoc
    import random
    with tempfile.TemporaryDirectory() as d:
        fake_mmcows(os.path.join(d, "mm"), keys=80, train_every=4)
        data = assoc.Data(os.path.join(d, "mm"))
        assert data.stride == {"train": 4, "window": 1}
        for make in (data.real, data.virtual):
            e = make("train", data.keys["train"][10], random.Random(0))
            hist = e["dt"][e["kind"] == assoc.HIST]
            assert len(hist) and set(np.round(hist).tolist()) <= {-60.0, -120.0}
            assert (e["kind"] == assoc.MEM).any()
        e = data.real("window", data.keys["window"][5], random.Random(0), train=False)
        assert set(np.round(e["dt"][e["kind"] == assoc.HIST]).tolist()) == {-15.0, -30.0}
        assert assoc.evaluate(assoc.build(48, 16, d=32, layers=1, heads=2).eval(), data, "train",
                              data.keys["train"][5:12], "cpu")["track_cases"] > 0


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
        res = system.score(a, det_log, {11: "A01", 12: "A02"}, names, t0, t0 + 300, {"cam_1"}, metrics,
                           {"unseen": [2], "seen": [1]})
        g = res["identification"]["by_cow_group"]
        assert g["seen"]["with_not_sure"]["recall"] == 1.0 and g["unseen"]["with_not_sure"]["recall"] == 0.0
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
                                           layers=1, heads=2, dropout=0.0, lr=2e-3, wd=0.0, temperature=0.1, seed=0,
                               split="days", val_cows=3, test_cows=5, split_seed=0, time_aug=1.0))
        summarize.main(["--work", w, "--out", os.path.join(d, "res")])
        page = open(os.path.join(d, "res", "RESULTS.md")).read()
        assert "ID F1" in page and "IDF1" in page


def test_one_barn_split_by_cow():
    """MmCows-like boxes over a day -> idcrops -> ID transformer trained on some cows, tested on others."""
    import data_mmcows
    import idnet
    rng = np.random.default_rng(3)
    cows, dim = 16, 48
    proto = rng.normal(size=(cows, dim))
    frames, rows, feats = [], [], []
    t0 = 1690261200.0                                            # 00:00 local on 25 July
    for k in range(0, 24 * 60, 20):                              # a frame every 20 min, 2 cameras
        ts = t0 + 60 * k
        win = 12 <= C.local_hour(ts) < 15
        for cam in ("cam_1", "cam_2"):
            for c in range(cows):
                rows.append({"frame": len(frames), "cow": c, "box": [0.1, 0.1, 0.2, 0.2]})
                feats.append(proto[c] + rng.normal(scale=0.5, size=dim))
            frames.append({"cam": cam, "ts": ts, "split": "window" if win else "train", "boxes": [], "path": ""})
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "assoc.npz"), box_feats=np.array(feats, np.float16), scene=np.zeros((1, 144, 4)))
        for name, data in (("frames.jsonl", frames), ("assoc_rows.jsonl", rows)):
            with open(os.path.join(d, name), "w") as fh:
                for r in data:
                    fh.write(json.dumps(r) + "\n")
        json.dump({"encoder": "stub", "dim": dim}, open(os.path.join(d, "assoc_meta.json"), "w"))
        data_mmcows.main(["idcrops", "--out", d, "--train-every", "1200", "--seq-len", "6"])
        ic = os.path.join(d, "idcrops")
        crops = [json.loads(l) for l in open(os.path.join(ic, "crops.jsonl"))]
        assert {r["day"] for r in crops} == {"A", "B"}
        n_window = sum(1 for f in frames if f["split"] == "window") * cows
        assert len(crops) == len(rows) - n_window                # nothing from 12:00-15:00
        assert max(len([r for r in crops if (r["day"], r["cow"], r["seq"]) == k])
                   for k in {(r["day"], r["cow"], r["seq"]) for r in crops}) <= 6
        a = argparse.Namespace(feats=ic, out=os.path.join(d, "idnet"), max_error=0.05, epochs=4, patience=4, P=8,
                               min_len=2, max_len=6, drop=0.1, noise=0.01, d=32, layers=1, heads=2, dropout=0.0,
                               lr=2e-3, wd=0.0, temperature=0.1, seed=0, split="cows", val_cows=3, test_cows=5,
                               split_seed=0, time_aug=0.02)
        idnet.cmd_train(a)
        ev = json.load(open(os.path.join(d, "idnet", "eval_id.json")))
        sp = ev["split"]
        assert sp["by"] == "cows" and len(sp["train"]) == 8 and len(sp["val"]) == 3 and len(sp["test"]) == 5
        assert not set(sp["test"]) & set(sp["train"]) and not set(sp["val"]) & set(sp["train"])
        assert ev["test"]["queries"] > 0 and ev["test"]["top1"] >= 0.6     # new cows, easy made-up ones
        ck = torch.load(os.path.join(d, "idnet", "idnet.pt"), weights_only=False)
        assert ck["split"]["test"] == sp["test"]


def test_video_from_a_made_up_system_run():
    import cv2
    import video
    t0 = 1690304400.0                                         # 12:00 local
    with tempfile.TemporaryDirectory() as d:
        data, sysd = os.path.join(d, "mm"), os.path.join(d, "sys")
        os.makedirs(os.path.join(data, "live"), exist_ok=True)
        os.makedirs(sysd)
        live, labels, tracks = [], [], []
        for k in range(0, 50):                                # 100 s from 12:59:20, 2 cameras
            ts = t0 + 3560 + 2 * k
            for cam in ("cam_1", "cam_2"):
                p = os.path.join(data, "live", f"{cam}_{k}.jpg")
                cv2.imwrite(p, np.full((200, 320, 3), 90, np.uint8))
                live.append({"cam": cam, "ts": ts, "path": p})
                b = [0.1 + 0.002 * k, 0.2, 0.3 + 0.002 * k, 0.5]
                tracks.append({"t": ts, "cam": cam, "box": b, "track": 1 if cam == "cam_1" else 2, "conf": 0.9})
                if k % 7 == 0:
                    labels.append({"cam": cam, "ts": ts, "boxes": [[5 if cam == "cam_1" else 9, b]]})
        for name, rows in (("live.jsonl", live), ("labels.jsonl", labels)):
            with open(os.path.join(data, name), "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        with open(os.path.join(sysd, "tracks_assoc.jsonl"), "w") as fh:
            for r in tracks:
                fh.write(json.dumps(r) + "\n")
        np.savez(os.path.join(data, "masks.npz"), cam_1=np.ones((35, 56), bool))
        json.dump({"t_start": t0, "t_setup": t0 + 3600, "step": 2, "setup_min": 60,
                   "track_album": {"1": "A01", "2": "A02"},
                   "names": [[1, 60, "A01", "A01"], [2, 60, "A01", "A01"], [1, 61, None, "A01"]],
                   "album_to_cow": {"A01": 5, "A02": 9}}, open(os.path.join(sysd, "names_assoc.json"), "w"))
        out = os.path.join(d, "v.mp4")
        tally = video.render(argparse.Namespace(data=data, system=sysd, tag="assoc", time_from="12:59",
                                                time_to="13:01", fps=5.0, width=640, match_iou=0.3, out=out,
                                                realtime=False, title=""))
        cap = cv2.VideoCapture(out)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        assert ok and n >= 45 and frame.shape[1] == 640
        # after 13:00: cam_1's cow 5 named A01 = cow 5 (right), cam_2's cow 9 named A01 too (wrong)
        assert tally["right"] >= 1 and tally["wrong"] >= 1


def test_v14c_taking_out_the_average_cow_splits_the_album():
    import system
    rng = np.random.default_rng(5)
    dim, cows = 32, 6
    common = rng.normal(size=dim)
    common /= np.linalg.norm(common)
    own = rng.normal(size=(cows, dim)) * 0.12                     # small differences between cows
    t0 = 1690304400.0
    recs, det_log = [], []
    for m in range(10):
        for c in range(cows):
            tid = 6 if (c == 0 and m >= 5) else c                  # cow 0's track breaks at minute 5
            fp = common * 3 + own[c] + rng.normal(scale=0.03, size=dim)
            recs.append((m, tid, (fp / np.linalg.norm(fp)).astype(np.float32), 1.0, 0.5, 0.1))
            for k in range(0, 60, 2):
                det_log.append((t0 + 60 * m + k, "cam_1", [0.1, 0.1, 0.2, 0.2], tid, 0.9))
    sims = [float(a[2] @ b[2]) for a in recs[:6] for b in recs[:6] if a is not b]
    assert min(sims) > 0.87                                       # way A: everyone looks alike
    _, ids_a, _ = system.enrol(recs, 0.8735, 5)
    assert len(set(ids_a.tolist())) == 1                          # ... one album cow: the A01 collapse
    c = system.centering(recs)
    rb = [(m, tid, system.apply_centering(c, fp), *rest) for m, tid, fp, *rest in recs]
    thr, cal = system.self_threshold(rb, det_log, t0)
    assert thr is not None and cal["different_pairs"] > 0 and cal["same_p10"] > cal["different_p99"]
    _, ids_b, ta = system.enrol(rb, thr, 5)
    assert len(set(ids_b.tolist())) == cows                       # way C: one album cow per cow
    assert ta[0] == ta[6]                                         # cow 0's two tracks: one album cow


def test_v14d_wildlifereid_plan_combine_and_training():
    import data_wr10k as W
    import idnet
    rng = np.random.default_rng(7)
    meta = []
    for ds, n_cows in (("Cows2021", 12), ("OpenCows2020", 6), ("AerialCattle2017", 3)):
        for c in range(n_cows):
            for i in range(10):
                meta.append({"identity": f"{ds}_{c}", "path": f"images/{ds}/{c}/{i}.jpg", "date": f"2020-02-{i + 1:02d}",
                             "dataset": ds, "split": "train"})
    meta.append({"identity": "Cows2021_tiny", "path": "images/Cows2021/tiny/0.jpg", "date": "", "dataset": "Cows2021", "split": "train"})
    keep = W.plan(meta, "Cows2021,OpenCows2020", min_images=4, val_share=0.2, seq_len=4, spacing=4.0)
    assert {r["dataset"] for r in keep} == {"Cows2021", "OpenCows2020"}         # AerialCattle2017 left out
    assert "Cows2021_tiny" not in {r["cow"] for r in keep}                       # too few images
    val = {r["cow"] for r in keep if r["day"] != "train"}
    assert len(val) == 2 + 1 and not val & {r["cow"] for r in keep if r["day"] == "train"}
    a = [r for r in keep if r["cow"] == sorted(val)[0]]
    assert max(r["name"] for r in a if r["day"] == "valA") < min(r["name"] for r in a if r["day"] == "valB")
    assert max(r["t"] for r in keep) == 12.0                                     # 4 per sequence, 4 s apart
    with tempfile.TemporaryDirectory() as d:
        wr, mm, out = (os.path.join(d, x) for x in ("wr", "mm", "idd"))
        dim = 48
        proto = {c: rng.normal(size=dim) for c in {r["cow"] for r in keep}}
        os.makedirs(wr)
        np.save(os.path.join(wr, "feats.npy"), np.array([proto[r["cow"]] + rng.normal(scale=0.5, size=dim) for r in keep], np.float16))
        with open(os.path.join(wr, "crops.jsonl"), "w") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
        json.dump({"encoder": "stub", "grid": 2, "crop": 224, "dim": dim}, open(os.path.join(wr, "meta.json"), "w"))
        fake_mcc(mm, days=2, cows=6, seqs=3, frames=6, dim=dim, seed=3)           # stands in for MmCows' idcrops
        rows = [json.loads(l) for l in open(os.path.join(mm, "crops.jsonl"))]
        with open(os.path.join(mm, "crops.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "day": "A" if r["day"].endswith("14") else "B"}) + "\n")
        W.main(["combine", "--wr", wr, "--mm", mm, "--out", out])
        days = collections.Counter(json.loads(l)["day"] for l in open(os.path.join(out, "crops.jsonl")))
        assert set(days) == {"train", "valA", "valB", "testA", "testB"}
        a = argparse.Namespace(feats=out, out=os.path.join(d, "idnet"), max_error=0.05, epochs=3, patience=3, P=8,
                               min_len=2, max_len=4, drop=0.1, noise=0.01, d=32, layers=1, heads=2, dropout=0.0,
                               lr=2e-3, wd=0.0, temperature=0.1, seed=0, split="groups", val_cows=3, test_cows=5,
                               split_seed=0, time_aug=0.25, time_aug_max=15.0, pure_batches=1)
        idnet.cmd_train(a)
        ev = json.load(open(os.path.join(d, "idnet", "eval_id.json")))
        assert ev["split"]["by"] == "groups" and len(ev["split"]["test"]) == 6 and ev["test"]["queries"] > 0
        assert not set(map(str, ev["split"]["test"])) & set(map(str, ev["split"]["train"]))


def test_rescore_a_saved_run_on_chosen_cows_and_eval_on_some_cows():
    import rescore
    t0 = 1690304400.0
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "labels.jsonl"), "w") as fh:
            for k in range(0, 600, 15):
                fh.write(json.dumps({"cam": "cam_1", "ts": t0 + k, "boxes": [[1, [0.1, 0.1, 0.3, 0.3]],
                                                                             [2, [0.6, 0.6, 0.8, 0.8]]]}) + "\n")
        with open(os.path.join(d, "tracks_assoc.jsonl"), "w") as fh:
            for k in range(0, 600, 2):
                fh.write(json.dumps({"t": t0 + k, "cam": "cam_1", "box": [0.1, 0.1, 0.3, 0.3], "track": 11, "conf": 0.9}) + "\n")
                fh.write(json.dumps({"t": t0 + k, "cam": "cam_1", "box": [0.6, 0.6, 0.8, 0.8], "track": 12, "conf": 0.9}) + "\n")
        names = [[11, m, "A01", "A01"] for m in range(10)] + [[12, m, None, "A01"] for m in range(5, 10)]
        json.dump({"t_start": t0, "t_setup": t0 + 300, "step": 2, "setup_min": 5, "track_album": {"11": "A01", "12": "A02"},
                   "names": names, "album_to_cow": {}}, open(os.path.join(d, "names_assoc.json"), "w"))
        res = rescore.rescore(d, d, "assoc", {"one": [1], "two": [2]}, start=12.0, end=12.17)
        g = res["identification"]["by_cow_group"]
        assert g["one"]["with_not_sure"]["recall"] == 1.0
        assert g["two"]["with_not_sure"]["recall"] == 0.0 and g["two"]["always_answer"]["recall"] == 0.0
    import idnet
    with tempfile.TemporaryDirectory() as d:
        fake_mcc(os.path.join(d, "mcc"))
        a = argparse.Namespace(feats=os.path.join(d, "mcc"), out=os.path.join(d, "idnet"), max_error=0.05, epochs=2,
                               patience=2, P=8, min_len=3, max_len=8, drop=0.1, noise=0.01, d=32, layers=1, heads=2,
                               dropout=0.0, lr=2e-3, wd=0.0, temperature=0.1, seed=0,
                               split="days", val_cows=3, test_cows=5, split_seed=0, time_aug=1.0)
        idnet.cmd_train(a)
        before = os.path.getmtime(os.path.join(d, "idnet", "idnet.pt"))
        idnet.main(["eval", "--feats", os.path.join(d, "mcc"), "--out", os.path.join(d, "idnet"),
                    "--only-cows", "1,2", "--eval-name", "some.json", "--keep-model"])
        ev = json.load(open(os.path.join(d, "idnet", "some.json")))
        assert ev["test"]["queries"] == 2 * 3 and ev["test_cows_only"] == ["1", "2"]
        assert os.path.getmtime(os.path.join(d, "idnet", "idnet.pt")) == before


def test_v14e_album_merging_scoring_and_video():
    import system_e as E
    rng = np.random.default_rng(11)
    t0 = 1690304400.0
    cows, dim = 6, 32
    common = rng.normal(size=dim)
    common /= np.linalg.norm(common)
    own = rng.normal(size=(cows, dim)) * 0.15
    # 12 chains in the setup hour: cow c has chains c and c+6 (its track broke at minute 5),
    # chains of different cows are seen together on cam_1 at every moment
    det_log, recs = [], []
    for m in range(10):
        for c in range(cows):
            cid = c if m < 5 else c + 6
            fp = common * 3 + own[c] + rng.normal(scale=0.03, size=dim)
            recs.append((m, cid, (fp / np.linalg.norm(fp)).astype(np.float32), 1.0, 0.5, 0.1))
            for k in range(0, 60, 2):
                det_log.append((t0 + 60 * m + k, "cam_1", [0.1 * c, 0.1, 0.1 * c + 0.08, 0.2], cid, 0.9))
    pairs = E.seen_together(det_log)
    assert (0, 1) in pairs and (0, 6) not in pairs              # cows 0 and 1 share moments; 0 and its later self never
    cen, P, ids, chain_album, thr, calib = E.build_album(recs, det_log, min_minutes=3)
    assert len(set(ids.tolist())) == cows                         # one album cow per cow ...
    assert all(chain_album[c] == chain_album[c + 6] for c in range(cows))   # ... with both chains of a cow in it
    assert len({chain_album[c] for c in range(cows)}) == cows      # and never two cows seen together merged
    # the setup labels: chain c is cow c+1 (ear tags 1..6), every 15 s
    labels = [{"cam": "cam_1", "ts": t0 + 60 * m + s, "boxes": [[c + 1, [0.1 * c, 0.1, 0.1 * c + 0.08, 0.2]] for c in range(cows)]}
              for m in range(10) for s in (0, 15, 30, 45)]
    truth = E.labelled_chains(det_log, labels, t0 + 600, 0.3)
    assert truth[(0, int(t0 // 60))].most_common(1)[0][0] == 1 and truth[(7, int(t0 // 60) + 6)].most_common(1)[0][0] == 2
    # scoring after setup: chain 0 named right every step, chain 1 never, chain 2 "not sure" (right guess)
    a = argparse.Namespace(step=2, match_iou=0.3)
    after = [{"cam": "cam_1", "ts": t0 + 600 + s, "boxes": [[1, [0.0, 0.1, 0.08, 0.2]], [2, [0.1, 0.1, 0.18, 0.2]], [3, [0.2, 0.1, 0.28, 0.2]]]}
             for s in (0, 15, 30)]
    log2 = [(t0 + 600 + k, "cam_1", [0.1 * c, 0.1, 0.1 * c + 0.08, 0.2], c, 0.9) for k in range(0, 32, 2) for c in range(3)]
    said = {}
    for k in range(300, 316):
        said[(0, k)] = ("cow1", "cow1")
        said[(1, k)] = ("cow3", "cow3")
        said[(2, k)] = (None, "cow3")
    import metrics
    r = E.score_e(a, log2, after, said, {"cow1": 1, "cow2": 2, "cow3": 3}, t0, t0 + 600, metrics)
    assert r["with_not_sure"]["right"] == 3 and r["with_not_sure"]["answered"] == 6
    assert r["always_answer"]["right"] == 6                      # chain 2's guess was right too
    # a per-step names file plays in video.py
    import cv2
    import video
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "live"))
        live = []
        for k in range(300, 310):
            p = os.path.join(d, "live", f"{k}.jpg")
            cv2.imwrite(p, np.full((100, 160, 3), 80, np.uint8))
            live.append({"cam": "cam_1", "ts": t0 + 2 * k, "path": p})
        for name, rows in (("live.jsonl", live), ("labels.jsonl", after)):
            with open(os.path.join(d, name), "w") as fh:
                for r_ in rows:
                    fh.write(json.dumps(r_) + "\n")
        with open(os.path.join(d, "tracks_e_confirmed.jsonl"), "w") as fh:
            for t, cam, box, cid, conf in log2:
                fh.write(json.dumps({"t": t, "cam": cam, "box": box, "track": cid, "conf": conf}) + "\n")
        json.dump({"t_start": t0, "t_setup": t0 + 600, "step": 2, "setup_min": 10, "per_step": True,
                   "track_album": {}, "names": [[c, k, n, al] for (c, k), (n, al) in said.items()],
                   "album_to_cow": {"cow1": 1, "cow2": 2, "cow3": 3}}, open(os.path.join(d, "names_e_confirmed.json"), "w"))
        tally = video.render(argparse.Namespace(data=d, system=d, tag="e_confirmed", time_from="12:10", time_to="12:10:20",
                                                fps=2.0, width=320, match_iou=0.3, out=os.path.join(d, "v.mp4"),
                                                realtime=False, title=""))
        assert tally["right"] >= 1 and tally["wrong"] >= 1 and tally.get("guess right", 0) >= 1



def test_freshness_finds_a_drifting_look():
    """Made-up day: each cow's look is a little of herself and a lot of her pose (lying here, standing there,
    shared by every cow in it), and she changes pose and place now and then - so an album fits the minutes
    around it best, before and after, and a cow in the same spot best of all."""
    import freshness as Fr
    import idnet
    rng = np.random.default_rng(5)
    cows, dim, t0, n_pose = 8, 32, 1690275600.0, 4              # 04:00 local
    base = rng.normal(size=(cows, dim)) * 0.6
    poses = rng.normal(size=(n_pose, dim)) * 2.0
    pose = rng.integers(0, n_pose, cows)
    frames, rows, feats = [], [], []
    for m in range(600):
        pose = np.where(rng.random(cows) < 0.01, rng.integers(0, n_pose, cows), pose)   # a change every ~100 min
        boxes = []
        for c in range(cows):
            x = 0.05 + 0.1 * c + 0.05 * pose[c]                    # a new pose is a new spot
            b = [min(x, 0.9), 0.4, min(x + 0.04, 0.98), 0.5]
            boxes.append([c, b])
            rows.append({"frame": len(frames), "cow": c, "box": b})
            feats.append(base[c] + poses[pose[c]] + rng.normal(scale=0.3, size=dim))
        frames.append({"cam": "cam_1", "ts": t0 + 60 * m + 1, "split": "train", "boxes": boxes})
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "assoc.npz"), box_feats=np.array(feats, np.float16))
        for name, data in (("frames.jsonl", frames), ("assoc_rows.jsonl", rows)):
            with open(os.path.join(d, name), "w") as fh:
                for r in data:
                    fh.write(json.dumps(r) + "\n")
        Fr.main(["--data", d, "--model", "none", "--out", os.path.join(d, "fr"), "--min-cows", "8", "--every-min", "60"])
        out = json.load(open(os.path.join(d, "fr", "freshness.json")))
        g = out["by_gap"]
        # queries are timed from the middle of a 10-minute album: the nearest are 5 minutes off
        assert "+0..+5" not in g and "-5..+0" not in g
        assert g["+5..+15"]["top1"] > 0.9 and g[">= +180"]["top1"] < g["+5..+15"]["top1"] - 0.2
        assert g["-15..-5"]["top1"] > g["-180..-120"]["top1"] + 0.2    # stale in both directions
        pl = out["by_gap_and_place"]
        assert pl["+30..+60 | same spot"]["top1"] > pl["+30..+60 | near"]["top1"] + 0.2   # same gap, moved: worse
        assert all(out["by_window_near"][h]["top1"] > out["by_window_far"][h]["top1"] for h in out["by_window_far"])
        assert out["chance"] == pytest.approx(1 / 8)
        assert any("| moved" in k for k in out["by_gap_and_place"]) and any("| same spot" in k for k in out["by_gap_and_place"])
        # with an ID model (untrained here): runs on one-frame sequences
        m = idnet.build(dim, d=16, layers=1, heads=2)
        torch.save({"kw": m.kw, "state": m.state_dict(), "epoch": 0, "split": {}, "feats": {}}, os.path.join(d, "idnet.pt"))
        Fr.main(["--data", d, "--idnet", os.path.join(d, "idnet.pt"), "--out", os.path.join(d, "fr2"), "--min-cows", "8",
                 "--every-min", "120"])
        assert os.path.exists(os.path.join(d, "fr2", "FRESHNESS.md"))



def test_split_zip_and_live_frames_from_two_windows():
    """A zip cut in two pieces reads as one (MmCows' whole-day 1-second zips), and a second window of live
    frames is added to live.jsonl next to the first (a 3-hour onboarding before the 12:00-15:00 stream)."""
    import io as _io
    import zipfile
    import data_mmcows as M
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cowbench"))
    import extra_det as X
    from PIL import Image
    t9 = 1690293600                                                   # 09:00 local on 25 July
    buf = _io.BytesIO()
    jpg = _io.BytesIO()
    Image.fromarray(np.full((20, 30, 3), 90, np.uint8)).save(jpg, "JPEG")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for s in range(0, 4 * 3600, 1):                              # 09:00-13:00, every second
            if s % 30 == 0:                                           # (every 30 s, to keep the test small)
                hh, mm, ss = 9 + s // 3600, s // 60 % 60, s % 60
                z.writestr(f"{t9 + s}_{hh:02d}-{mm:02d}-{ss:02d}.jpg", jpg.getvalue())
    data = buf.getvalue()
    cut = len(data) // 2 + 7

    class Piece:
        def __init__(self, b):
            self.b, self.size = b, len(b)

        def get(self, a, e):
            return self.b[a:e + 1]

    sz = object.__new__(X.SplitRemoteZip)
    sz.parts, sz.starts, sz.size = [Piece(data[:cut]), Piece(data[cut:])], [0, cut, len(data)], len(data)
    sz.zip = zipfile.ZipFile(_io.BufferedReader(X._RangeFile(sz), buffer_size=1 << 12))
    names = sz.names()
    assert len(names) == 480 and all(sz.read(n) == jpg.getvalue() for n in names[::37])
    straddle = [i for i in sz.zip.infolist() if i.header_offset < cut < i.header_offset + i.compress_size + 60]
    assert straddle and sz.read(straddle[0].filename) == jpg.getvalue()        # a frame cut in two
    # which zip a window reads from
    a = argparse.Namespace(start=12.0, end=15.0, live=M.LIVE)
    assert M.LIVE_HOURS[0] <= a.start and a.end <= M.LIVE_HOURS[1]
    picked = []
    real = X.SplitRemoteZip
    X.SplitRemoteZip = lambda urls: picked.append(urls) or sz
    try:
        assert M.live_zip(argparse.Namespace(start=9.0, end=13.0, live=M.LIVE), 1) is sz
        assert picked[0] == [M.LIVE_DAY.format(k=1, p=p) for p in M.LIVE_DAY_PARTS]
        real_remote = M.remote
        M.remote = lambda url: ("3h", url)
        assert M.live_zip(a, 2) == ("3h", M.LIVE.format(k=2))
        M.remote = real_remote
        # two windows into one live.jsonl
        with tempfile.TemporaryDirectory() as d:
            M.live_zip = lambda args, k: sz
            for s_, e_ in ((12.0, 13.0), (9.0, 12.0)):
                M.cmd_live(argparse.Namespace(out=d, cams="1", start=s_, end=e_, live_step=60.0, workers=2, max_side=64))
            rows = [json.loads(l) for l in open(os.path.join(d, "live.jsonl"))]
            hours = sorted({int(C.local_hour(r["ts"])) for r in rows})
            assert hours == [9, 10, 11, 12] and len(rows) == 4 * 60 and all(os.path.exists(r["path"]) for r in rows)
    finally:
        X.SplitRemoteZip = real
        import importlib
        importlib.reload(M)



def test_v14g_results_page_and_comparison_with_the_control():
    import system_e as E

    def run(start, setup_min, f1s):
        prf = lambda f: {"f1": f, "precision": f, "recall": f, "answered_share": 1.0}
        return {"settings": {"start": start, "setup_min": setup_min, "end": 13.0},
                "identification": {w: {"with_not_sure": prf(f / 2), "always_answer": prf(f)} for w, f in f1s.items()},
                "album": {"built": {"album_cows": 30, "true_cows": 16, "purity": 0.9, "coverage": 1.0},
                          "confirmed_cows": 16, "oracle_cows": 16, "threshold": 0.7, "self_calibration": {}},
                "detector": {"recall": 0.95, "precision": 0.96}, "chains": 900}

    g = run(9.0, 180, {"built": 0.2, "confirmed": 0.45, "oracle": 0.5})
    c = run(11.0, 60, {"built": 0.1, "confirmed": 0.3, "oracle": 0.3})
    assert E.window_text(g["settings"]) == "onboarding 09:00-12:00 (180 min), scored 12:00-13:00"
    with tempfile.TemporaryDirectory() as d:
        page = os.path.join(d, "RESULTS.md")
        E.report_e(g, page, "G")
        text = open(page).read()
        assert text.startswith("# Version 14 (G):") and "onboarding 09:00-12:00 (180 min), scored 12:00-13:00" in text
        for name, obj in (("g.json", g), ("c.json", c)):
            json.dump(obj, open(os.path.join(d, name), "w"))
        E.compare_main(["--main", os.path.join(d, "g.json"), "--control", os.path.join(d, "c.json"), "--append", page])
        text = open(page).read()
        assert "control: onboarding 11:00-12:00 (60 min), scored 12:00-13:00" in text
        assert "| confirmed | **45.0%** | 30.0% | 22.5% | 15.0% |" in text
        assert "| oracle | **50.0%** | 30.0% |" in text
