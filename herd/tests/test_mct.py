"""Who-was-where-when in identification: cows seen at the same moment are different cows."""
import _paths  # noqa: F401
import copy
import os
import tempfile

import numpy as np

from common import DEFAULTS


def _cfg(**identity):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["identity"].update(identity)
    return cfg


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def _gallery(d, cows, **identity):
    import gallery as g
    gal = g.Gallery(d, _cfg(**identity))
    for cid, v in cows.items():
        gal.cows[cid] = {"prototypes": _unit(v)[None], "created": 0.0, "last_seen": 0.0}
    return gal


def _near(v, rng, noise=0.05):
    return _unit(np.asarray(v, np.float32) + rng.normal(0, noise, len(v)))


def test_match_batch_gives_no_cow_twice():
    rng = np.random.default_rng(0)
    a, b = np.eye(16)[0], np.eye(16)[0] * 0.8 + np.eye(16)[1] * 0.6
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {"A": a, "B": b})
        items = [{"fingerprint": _near(a, rng, 0.01)}, {"fingerprint": _near(a, rng, 0.2)}]
        ds = gal.match_batch(items)
        cows = [x["cow"] for x in ds if x["cow"]]
        assert len(cows) == len(set(cows))                         # never one cow for two bursts at once
        assert ds[0]["cow"] == "A"
        assert gal.match(items[1]["fingerprint"], exclude={"A"})["cow"] in (None, "B")


def test_claims_hold_a_cow_against_a_track_seen_at_the_same_time():
    import pipeline
    from store import Store
    a, b = np.eye(16)[0], np.eye(16)[2]
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        cfg = _cfg()
        shared = pipeline.SharedGallery(_gallery(d, {"A": a, "B": b}), store, cfg, 1000.0)
        out = lambda v: {"fingerprint": v, "quality_max": 0.5, "quality_mean": 0.5}
        d1 = shared.decide(out(_unit(a)), 0.05, 1000.0, "cam1-1")
        assert d1["state"] == "confirmed" and d1["cow"] == "A"
        shared.together(["cam1-1", "cam1-2"], 1001.0)               # both tracks in one frame: two cows
        # a less sure burst of the other track: A is held, so not A
        d2 = shared.decide(out(_unit(a * 0.9 + b * 0.25)), 0.05, 1010.0, "cam1-2")
        assert d2["cow"] != "A"
        # a track never seen with cam1-1 may still be A (it may be the same cow, later, elsewhere)
        d3 = shared.decide(out(_unit(a)), 0.05, 1020.0, "cam2-7")
        assert d3["cow"] == "A"
        assert store.query("SELECT COUNT(*) FROM events WHERE kind='claim-kept'")[0][0] >= 1


def test_enroll_never_merges_cows_seen_together():
    rng = np.random.default_rng(1)
    x = _unit(rng.normal(size=32))
    y = _unit(x + 0.55 * _unit(rng.normal(size=32)))               # a look-alike, cos ~0.85
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {}, new_cow_min_bursts=4)
        for k in range(6):
            gal.pool.append({"ts": k, "track": "t-x1", "fp": _near(x, rng, 0.02)})
            gal.pool.append({"ts": 100 + k, "track": "t-x2", "fp": _near(x, rng, 0.02)})
            gal.pool.append({"ts": k, "track": "t-y", "fp": _near(y, rng, 0.02)})
        made = gal.enroll(200.0, cannot_link={("t-x1", "t-y")})
        assert len(made) == 2                                      # x's two tracks one cow, y another
        sizes = sorted(len(gal.history[c]) for c in made)
        assert sizes == [6, 12]


def test_enroll_holds_back_when_the_herd_is_full():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {"A": np.eye(8)[0]}, new_cow_min_bursts=3, max_cows=1)
        for k in range(5):
            gal.pool.append({"ts": k, "track": "t1", "fp": _near(np.eye(8)[3], rng, 0.02)})
        assert gal.enroll(10.0) == [] and gal.held_back == 1 and len(gal.pool) == 5


def test_report_follows_cow_aliases():
    import report
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        for k, cow in enumerate(["new-1", "new-1", "A"]):
            store.burst(("cam1", "trk", 100.0 + k, "confirmed", cow, 0.9, 0.2, 0.9, 0.1, "lying", "none",
                         None, 0.5, 100))
        assert report.track_map(store) == {"trk": "new-1"}
        store.alias_cow("new-1", "A", 200.0)
        assert report.track_map(store) == {"trk": "A"}
        store.unalias_cow("new-1")
        assert report.track_map(store) == {"trk": "new-1"}


# ------------------------------------------------------------------ stage B: one track across cameras

def _mct_cfg(cams, **mct):
    cfg = _cfg()
    cfg["cameras"] = [dict({"id": c, "mask": []}, **extra) for c, extra in cams]
    cfg["mct"].update(mct)
    return cfg


def _box(cx, cy=0.5, s=0.1):
    return [cx - s, cy - s, cx + s, cy + s]


def test_overlap_hands_a_cow_on_and_keeps_two_apart():
    import mct
    rng = np.random.default_rng(3)
    A, B = _unit(rng.normal(size=64)), _unit(rng.normal(size=64))
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {}), ("cam2", {})], neighbours=[["cam1", "cam2"]]))
    g = None
    for k in range(5):                                       # A walks right in cam1
        g1 = gt.update("cam1", float(k), [{"box": _box(0.5 + 0.08 * k), "emb": _near(A, rng)}])[0]
        assert g is None or g1 == g
        g = g1
    # the overlap: cam2 sees her at its left edge while cam1 still has her, and B beside her
    out = gt.update("cam2", 4.0, [{"box": _box(0.12), "emb": _near(A, rng)}, {"box": _box(0.5), "emb": _near(B, rng)}])
    assert out[0] == g and out[1] != g
    for k in range(5, 8):                                    # cam1 has lost her; cam2 follows by IoU
        out = gt.update("cam2", float(k), [{"box": _box(0.12 + 0.05 * (k - 4)), "emb": _near(A, rng)}])
        assert out[0] == g
    assert gt.stats["overlap"] == 1


def test_handover_refuses_a_look_alike_pair():
    import mct
    rng = np.random.default_rng(4)
    A = _unit(rng.normal(size=64))
    A2 = _unit(A + 0.02 * rng.normal(size=64))             # two cows the frame id cannot tell apart
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {}), ("cam2", {})], neighbours=[["cam1", "cam2"]]))
    ga, gb = gt.update("cam1", 0.0, [{"box": _box(0.8, 0.3), "emb": A}, {"box": _box(0.8, 0.7), "emb": A2}])
    out = gt.update("cam2", 1.0, [{"box": _box(0.1, 0.5), "emb": _near(A, rng, 0.01)}])
    assert out[0] not in (ga, gb) and gt.stats["overlap_ambiguous"] == 1


def test_portal_bridges_a_blind_passage_in_its_time_window():
    import mct
    rng = np.random.default_rng(5)
    A = _unit(rng.normal(size=64))
    cfg = _mct_cfg([("cam1", {}), ("cam3", {})],
                   portals=[{"from": "cam1", "to": "cam3", "min_s": 5, "max_s": 30, "exit": [[0.7, 0], [1, 0], [1, 1], [0.7, 1]]}])
    gt = mct.GlobalTracker(cfg)
    g = gt.update("cam1", 0.0, [{"box": _box(0.85), "emb": A}])[0]
    assert gt.update("cam3", 2.0, [{"box": _box(0.2), "emb": _near(A, rng)}])[0] != g      # too soon
    gt2 = mct.GlobalTracker(cfg)
    g = gt2.update("cam1", 0.0, [{"box": _box(0.85), "emb": A}])[0]
    assert gt2.update("cam3", 12.0, [{"box": _box(0.2), "emb": _near(A, rng)}])[0] == g
    gt3 = mct.GlobalTracker(cfg)
    g = gt3.update("cam1", 0.0, [{"box": _box(0.85), "emb": A}])[0]
    assert gt3.update("cam3", 45.0, [{"box": _box(0.2), "emb": _near(A, rng)}])[0] != g    # too late


def test_reacquire_after_a_short_hide():
    import mct
    rng = np.random.default_rng(6)
    A = _unit(rng.normal(size=64))
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {})]))
    g = gt.update("cam1", 0.0, [{"box": _box(0.5), "emb": A}])[0]
    gt.update("cam1", 1.0, [])
    # back 7 s later (past max_age_s), a little further on: the same cow by her frame id
    assert gt.update("cam1", 8.0, [{"box": _box(0.58), "emb": _near(A, rng, 0.02)}])[0] == g
    assert gt.stats["reacquired"] == 1


def test_disjoint_cameras_add_cannot_link():
    import mct
    rng = np.random.default_rng(7)
    got = []
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {}), ("cam4", {})], disjoint=[["cam1", "cam4"]]),
                           apart=lambda pairs, ts: got.extend(pairs))
    a = gt.update("cam1", 10.0, [{"box": _box(0.5), "emb": _unit(rng.normal(size=8))}])[0]
    b = gt.update("cam4", 10.2, [{"box": _box(0.5), "emb": _unit(rng.normal(size=8))}])[0]
    assert (b, a) in got or (a, b) in got


class _FakeModels:
    """What Camera._second_global asks of the models: crops, encoder and frame heads, ids set by the test."""
    crop_size, margin = 16, 0.1

    def __init__(self):
        self.ids = {}

    def detect(self, img, **kw):
        return self.boxes

    def encode(self, crops, **kw):
        return None

    def frame_heads(self, feats, boxes, with_id=False, **kw):
        n = len(boxes)
        pp, pa = np.tile([0.9, 0.1], (n, 1)), np.tile([0.1, 0.1, 0.8], (n, 1))
        return pp, pa, [self.ids[round((b[0] + b[2]) / 2, 2)] for b in boxes]


def test_overlap_counts_every_second_once():
    """Two cameras over one floor, each owning a half (mask) and tracking all of it (tracking_mask):
    one cow walking across is one track and one row a second, whichever camera owns her."""
    import mct
    import pipeline
    from store import Store
    rng = np.random.default_rng(8)
    A = _unit(rng.normal(size=64))
    left, right = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]
    whole = [[0, 0], [1, 0], [1, 1], [0, 1]]
    # both cameras see the same floor in the same frame coordinates (the simplest overlap)
    cams = [("cam1", {"mask": left, "tracking_mask": whole}), ("cam2", {"mask": right, "tracking_mask": whole})]
    cfg = _mct_cfg(cams, enabled=True, neighbours=[["cam1", "cam2"]])
    cfg["cameras"] = [dict(c, min_box_pct=0.0, min_box_side_px=0) for c in cfg["cameras"]]
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        shared = pipeline.SharedGallery(_gallery(d, {}), store, cfg, 0.0)
        models = _FakeModels()
        glob = mct.GlobalTracker(cfg, store, shared.separate)
        cams_ = [pipeline.Camera(c, i, 2, cfg, models, shared, store, 0.0) for i, c in enumerate(cfg["cameras"])]
        for c in cams_:
            c.glob = glob
        img = np.zeros((64, 64, 3), np.uint8)
        for k in range(10):
            cx = round(0.15 + 0.08 * k, 2)
            models.boxes = [_box(cx)]
            models.ids = {cx: _near(A, rng, 0.03)}
            for c in cams_:
                c.second(float(k), img)
        rows = store.query("SELECT cam, track, ts FROM seconds ORDER BY ts")
        assert [r[2] for r in rows] == [float(k) for k in range(10)]            # one row a second
        assert len({r[1] for r in rows}) == 1                                   # one track
        assert {r[0] for r in rows} == {"cam1", "cam2"}                          # owned by each in turn


def test_frame_id_training_points_frames_at_their_burst():
    import json as js
    import model as mdl
    import train
    rng = np.random.default_rng(9)
    dim, T = 24, 30
    with tempfile.TemporaryDirectory() as d:
        feat, run = os.path.join(d, "features"), os.path.join(d, "run")
        tr = os.path.join(feat, "train")
        os.makedirs(tr), os.makedirs(run)
        js.dump({"dim": dim, "encoder": "x", "grid": 2, "crop": 224, "margin": 0.1}, open(os.path.join(tr, "meta.json"), "w"))
        rows = []
        for clip in range(10):
            cows = rng.normal(size=(4, dim)) * 2
            bf = np.stack([c + rng.normal(size=(T, dim)) * 0.7 for c in cows]).astype(np.float16)
            bt = np.tile(np.arange(T, dtype=np.float32) / 25, (4, 1))
            np.savez(os.path.join(tr, f"{clip}.npz"), burst_feats=bf, burst_times=bt, key_feats=np.zeros((0, dim), np.float16))
            rows += [{"kind": "burst", "clip": str(clip), "i": i, "posture": 0, "activity": 2, "rumination": 0,
                      "area": 0.05, "uids": []} for i in range(4)]
            rows.append({"kind": "clip_done", "clip": str(clip)})
        with open(os.path.join(tr, "index.jsonl"), "w") as fh:
            fh.writelines(js.dumps(r) + "\n" for r in rows)
        import torch
        torch.manual_seed(0)
        m = mdl.HerdModel(dim, d=32, layers=1, heads=2)
        m.save(os.path.join(run, "model.pt"), {"crop_s": 0.5, "epoch": 1, "features": {"dim": dim},
                                                 "dev_clips": ["0", "1"], "rum_threshold": 0.5})
        train.main(["frame-id", "--features", feat, "--out", run, "--epochs", "8", "--P", "8", "--seed", "0"])
        ev = js.load(open(os.path.join(run, "eval_frame_id.json")))
        assert ev["frame_to_burst_top1_clip"] >= 0.8, ev
        m2, ck = mdl.HerdModel.load(os.path.join(run, "model.pt"))
        assert m2.frame_id and ck["dev_clips"] == ["0", "1"] and "frame_id" in ck
        assert os.path.exists(os.path.join(run, "model.pre_frame_id.pt"))
        old, _ = mdl.HerdModel.load(os.path.join(run, "model.pre_frame_id.pt"))
        assert not old.frame_id


def test_mct_eval_scores_one_track_across_cameras():
    """Four cows under two cameras (the second sees the pen mirrored), labelled every 15 s: the per-camera
    IoU tracker cannot join the cameras, mct must; cows hidden for a few seconds come back by frame id."""
    import json as js
    import mct_eval
    rng = np.random.default_rng(10)
    cows = {c: _unit(rng.normal(size=32)) for c in (1, 2, 3, 4)}
    with tempfile.TemporaryDirectory() as d:
        js.dump({"cams": ["cam_1", "cam_2"], "start_ts": 0, "end_ts": 300, "step": 1}, open(os.path.join(d, "meta.json"), "w"))
        os.makedirs(os.path.join(d, "emb"))
        for cam in ("cam_1", "cam_2"):
            os.makedirs(os.path.join(d, "labels", cam))
            T, N, B, F = [], [], [], []
            for ts in range(300):
                rows = []
                for c, v in cows.items():
                    x = 0.15 + 0.2 * (c - 1) + 0.03 * np.sin(ts / 40 + c)
                    y = 0.5 + 0.2 * np.cos(ts / 50 + c)
                    if cam == "cam_2":
                        x = 1 - x
                    box = [x - 0.06, y - 0.08, x + 0.06, y + 0.08]
                    rows.append((c, box))
                    hidden = cam == "cam_1" and c == 2 and 100 <= ts < 107
                    if not hidden:
                        B.append(box), F.append(_near(v, rng, 0.05))
                T.append(ts), N.append(sum(1 for c, _ in rows if not (cam == "cam_1" and c == 2 and 100 <= ts < 107)))
                if ts % 15 == 0:
                    with open(os.path.join(d, "labels", cam, f"{1690304400 + ts}_12-00-00.txt"), "w") as fh:
                        for c, b in rows:
                            fh.write(f"{c} {(b[0] + b[2]) / 2} {(b[1] + b[3]) / 2} {b[2] - b[0]} {b[3] - b[1]}\n")
            np.savez(os.path.join(d, "emb", f"{cam}.npz"), ts=np.array(T) + 1690304400, n=np.array(N),
                     boxes=np.array(B, np.float32), cls=np.array(F, np.float16), fid=np.array(F, np.float16))
        mct_eval.main(["score", "--data", d, "--min-bursts", "3", "--sweep", "0.6"])
        r = js.load(open(os.path.join(d, "mct_eval.json")))
        assert r["iou"]["cross_cam_same_track"] == 0.0 and r["iou"]["idf1_global"] < 0.7
        m = r["fid/mct"]
        assert m["cross_cam_same_track"] > 0.95 and m["idf1_global"] > 0.95, m
        assert m["false_merges"] == 0 and m["false_cannot_link"] == 0
        assert r["fid/cam"]["tracks_per_cow"] < r["iou"]["tracks_per_cow"]      # the hidden cow comes back
        e = m["enroll"]["with_cannot_link"]
        assert e["cows_made"] == 4 and e["purity"] == 1.0
