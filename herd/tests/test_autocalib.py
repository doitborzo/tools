"""Cameras calibrated by the cows themselves: two views of one floor, cows walking and lying."""
import _paths  # noqa: F401
import collections
import json
import os
import tempfile

import numpy as np


def _distort(q, k, ar=16 / 9):
    """The inverse of common.undistort: where a lens with k shows the point q."""
    from common import undistort
    p = np.array(q, float)
    for _ in range(50):
        u = np.array(undistort(p[0], p[1], k, ar))
        p = p + (np.array(q) - u)
    return p


def _rows(lag=7, kB=-0.2, seconds=600, seed=0):
    """Two views of one floor, cows walking and lying: (cam, track, ts, x1, y1, x2, y2) every second."""
    from common import homography
    rng = np.random.default_rng(seed)
    floor = [(0, 0), (1000, 0), (1000, 800), (0, 800)]
    HA = homography(floor, [(0.1, 0.35), (0.9, 0.35), (0.98, 0.95), (0.02, 0.95)])       # from one side
    HB = homography(floor, [(0.85, 0.3), (0.95, 0.9), (0.1, 0.92), (0.2, 0.32)])         # from another corner
    cows = []
    for i in range(14):
        p = rng.uniform([100, 100], [900, 700])
        path = []
        moving = i < 8
        for t in range(seconds):
            if moving and (t // 40 + i) % 3 == 0:            # walks a while, then stands, then walks again
                p = np.clip(p + rng.normal(0, 25, 2), 50, 950)
            path.append(p.copy())
        cows.append(path)
    rows = []
    for cam, H, k, off in (("cam1", HA, 0.0, 0), ("cam2", HB, kB, lag)):
        for i, path in enumerate(cows):
            for t, p in enumerate(path):
                q = H @ np.array([p[0], p[1], 1.0])
                u, v = _distort((q[0] / q[2], q[1] / q[2]), k)
                if not (0.02 < u < 0.98 and 0.05 < v < 0.99):
                    continue
                u, v = u + rng.normal(0, 0.004), v + rng.normal(0, 0.004)
                h, w = 0.12, 0.14
                rows.append((cam, f"{cam}-{i}", 1000.0 + t + off, u - w / 2, v - h, u + w / 2, v))
    return rows


def _scene(d, **kw):
    from store import Store
    store = Store(os.path.join(d, "s.sqlite"))
    store.seconds([r + ("lying", "none", 0.5) for r in _rows(**kw)])
    return os.path.join(d, "s.sqlite")


CONFIG = """
[farm]
timezone_offset_hours = 2
[mct]
enabled = false
# mode = "link"
disjoint = []
[[cameras]]
id = "cam1"
url = "rtsp://a"
# a comment
[[cameras]]
id = "cam2"
url = "rtsp://b"
"""


def test_two_cameras_calibrated_by_moving_cows():
    import autocalib
    from common import floor_mapper, load_config
    with tempfile.TemporaryDirectory() as d:
        st = _scene(d)
        cfg = os.path.join(d, "barn.toml")
        open(cfg, "w").write(CONFIG)
        out = os.path.join(d, "calib")
        autocalib.main(["--config", cfg, "--store", st, "--out", out, "--apply"])
        rep = json.load(open(os.path.join(out, "calib.json")))
        pair = rep["pairs"]["cam1:cam2"]
        assert pair["lag"] == 7 and pair["share"] > 0.8, pair                      # the clocks' difference found
        assert pair["kB"] in (-0.1, -0.2, -0.3)                                      # the bent lens seen
        c = load_config(cfg)
        assert c["mct"]["enabled"] and c["mct"]["mode"] == "link" and 80 <= c["mct"]["link_dist"] <= 300
        cams = {x["id"]: x for x in c["cameras"]}
        fa, fb = floor_mapper(cams["cam1"], c["mct"]), floor_mapper(cams["cam2"], c["mct"])
        # one cow, both cameras, the same second: about the same place on the plan
        from store import Store
        s = Store(st)
        ra = {r[0].split("-")[1] + f"@{int(r[1])}": r[2:] for r in s.query("SELECT track, ts, x1, y1, x2, y2 FROM seconds WHERE cam='cam1'")}
        rb = {r[0].split("-")[1] + f"@{int(r[1]) - 7}": r[2:] for r in s.query("SELECT track, ts, x1, y1, x2, y2 FROM seconds WHERE cam='cam2'")}
        both = sorted(set(ra) & set(rb))[::25]
        d_same = [np.hypot(*np.subtract(fa(ra[k]), fb(rb[k]))) for k in both]
        assert np.median(d_same) < 0.5 * c["mct"]["link_dist"], (np.median(d_same), c["mct"]["link_dist"])
        assert "a comment" in open(cfg).read() and os.path.exists(cfg + ".bak")


def test_no_shared_floor_is_said_so():
    import autocalib
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        s = Store(os.path.join(d, "s.sqlite"))
        rng = np.random.default_rng(1)
        rows = []
        for cam in ("cam1", "cam2"):                      # two pens: their cows move independently
            for i in range(6):
                p = rng.uniform(0.2, 0.8, 2)
                for t in range(300):
                    p = np.clip(p + rng.normal(0, 0.01, 2), 0.1, 0.9)
                    rows.append((cam, f"{cam}-{i}", 1000.0 + t, p[0] - 0.05, p[1] - 0.1, p[0] + 0.05, p[1], "standing", "none", 0.1))
        s.store = None
        s.seconds(rows)
        cfg = os.path.join(d, "barn.toml")
        open(cfg, "w").write(CONFIG)
        try:
            autocalib.main(["--config", cfg, "--store", os.path.join(d, "s.sqlite"), "--out", os.path.join(d, "c")])
            raise AssertionError("calibrated two pens that share no floor")
        except SystemExit as e:
            assert "no camera pair" in str(e)


BITS = 11                                                     # the second, written into the picture's top row


def _videos(d, lag=7, seconds=600, fps=2, size=(320, 180)):
    """The scene as two recordings with the same start time, cam2's clock lag s behind: each frame carries
    its second in white blocks, so the stand-in detector knows which boxes it shows."""
    import cv2
    boxes = collections.defaultdict(list)
    for cam, _, ts, *b in _rows(lag=lag, seconds=seconds):
        boxes[(cam, int(ts) - 1000)].append(b)
    rec = os.path.join(d, "rec")
    os.makedirs(rec)
    for n, cam in enumerate(("cam1", "cam2")):
        w = cv2.VideoWriter(os.path.join(rec, f"{cam}_2026-10-20_06-00-00.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, size)
        for i in range(seconds * fps):
            img = np.full((size[1], size[0], 3), 40, np.uint8)
            code = i // fps + (n << (BITS - 1))
            for bit in range(BITS):
                if code >> bit & 1:
                    img[0:20, bit * 24 + 2:bit * 24 + 22] = 255
            w.write(img)
        w.release()
    return rec, boxes


class _CodeLive:
    """The detector: reads the second (and the camera) off the picture, gives that second's cows."""

    def __init__(self, boxes):
        self.boxes, self.calls = boxes, 0

    def many(self, imgs):
        out = []
        for img in imgs:
            a = np.asarray(img)
            code = sum(1 << bit for bit in range(BITS) if a[2:18, bit * 24 + 5:bit * 24 + 19].mean() > 128)
            cam = "cam2" if code >> (BITS - 1) else "cam1"
            out.append([{"bbox": b, "det_score": 0.9} for b in self.boxes.get((cam, code & ((1 << (BITS - 1)) - 1)), [])])
        self.calls += len(imgs)
        return out


def test_recordings_looked_at_whole_despite_the_masks():
    """The masks cut the overlap away from cam1 (a run would store none of its cows there): the detector
    looks at the recordings again, on the whole picture, and a second run reads its boxes back."""
    import autocalib
    with tempfile.TemporaryDirectory() as d:
        rec, boxes = _videos(d)
        cfg = os.path.join(d, "barn.toml")
        open(cfg, "w").write(CONFIG.replace('url = "rtsp://a"', 'url = "rtsp://a"\nmask = [[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]]'))
        out = os.path.join(d, "calib")
        live = _CodeLive(boxes)
        autocalib.main(["--config", cfg, "--videos", rec, "--out", out], live=live)
        pair = json.load(open(os.path.join(out, "calib.json")))["pairs"]["cam1:cam2"]
        assert pair["lag"] in (6, 7, 8) and pair["share"] > 0.7, pair
        assert os.path.exists(os.path.join(out, "cam1_cam2.jpg")) and os.path.exists(os.path.join(out, "cam1_cam2_moving.jpg"))
        n = live.calls
        autocalib.main(["--config", cfg, "--videos", rec, "--out", out], live=live)
        assert live.calls == n                                              # the boxes read back


def test_a_full_pen_of_lying_cows_is_no_calibration():
    """Two pens full of lying cows whose boxes all twitch together on both cameras (the picture's exposure
    changing, a person walking past): no cow moved, so no floor is claimed - and a dense pen, where any
    point is near some cow, does not pass the check by chance."""
    import autocalib
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        s = Store(os.path.join(d, "s.sqlite"))
        rng = np.random.default_rng(3)
        twitch = rng.normal(0, 1, 500) * (rng.random(500) < 0.15)     # the same seconds on both cameras
        rows = []
        for cam in ("cam1", "cam2"):
            for i in range(22):
                p = rng.uniform(0.1, 0.9, 2)
                for t in range(500):
                    j = rng.normal(0, 0.002, 2) + 0.01 * twitch[t] * rng.normal(1, 0.3, 2)
                    rows.append((cam, f"{cam}-{i}", 1000.0 + t, p[0] - 0.06 + j[0], p[1] - 0.1 + j[1],
                                 p[0] + 0.06 + j[0], p[1] + j[1], "lying", "none", 0.9))
        s.seconds(rows)
        _, mv = autocalib.moving_pairs(autocalib.load_tracks(s, "cam1"), autocalib.load_tracks(s, "cam2"), 90)
        assert len(mv) < 4, len(mv)                                    # the twitch is not a cow moving
        cfg = os.path.join(d, "barn.toml")
        open(cfg, "w").write(CONFIG)
        try:
            autocalib.main(["--config", cfg, "--store", os.path.join(d, "s.sqlite"), "--out", os.path.join(d, "c")])
            raise AssertionError("calibrated two pens of lying cows")
        except SystemExit as e:
            assert "no camera pair" in str(e)
