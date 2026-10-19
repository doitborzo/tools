"""Datasets from the barn's recordings: frames + boxes for the detector, the cache for identity."""
import _paths  # noqa: F401
import json
import os
import sys
import tarfile
import tempfile
import types

import numpy as np

CONFIG = """
[farm]
timezone_offset_hours = 2
[sampling]
frame_width = 320
[detector]
weights = "{det}"
[store]
path = "{d}/barn/herd.sqlite"
[[cameras]]
id = "cam1"
url = "rtsp://x"
exclude = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.2], [0.0, 0.2]]
[[cameras]]
id = "cam2"
url = "rtsp://y"
"""


def _video(path, seconds, fps=5, size=(640, 360)):
    """A cow (a white block) walking across a dark pen, so frames differ."""
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        x = int((i / (seconds * fps)) * (size[0] - 120))
        img[150:300, x:x + 120] = 230
        w.write(img)
    w.release()


def _setup(d):
    det = os.path.join(d, "det")
    os.makedirs(det)
    json.dump({"threshold": 0.5, "base_model": "stub"}, open(os.path.join(det, "det_train_meta.json"), "w"))
    cfg = os.path.join(d, "barn.toml")
    open(cfg, "w").write(CONFIG.format(det=det, d=d))
    rec = os.path.join(d, "rec")
    _video(os.path.join(rec, "cam1_2026-10-20_06-00-00.mp4"), 60)
    _video(os.path.join(rec, "cam2", "20261020T060010.mp4"), 60)
    _video(os.path.join(rec, "cam1_2026-10-20_09-00-00.mp4"), 60)
    open(os.path.join(rec, "notes.mp4"), "w").close()                  # no camera, no time: skipped
    return cfg, rec


class _StubLive:
    """detector.Live as box_dataset uses it: one sure cow and one unsure on every frame."""
    name, device = "stub", "cpu"

    def __init__(self, best, threshold=None, tiles=None, calib=True):
        self.threshold = threshold

    def many(self, imgs):
        return [[{"bbox": [0.1, 0.4, 0.3, 0.8], "det_score": 0.9}, {"bbox": [0.6, 0.4, 0.8, 0.8], "det_score": 0.3}]
                for _ in imgs]


def test_recordings_found_by_camera_and_time():
    import common
    with tempfile.TemporaryDirectory() as d:
        for f in ("cam1_2026-10-20_06-00-00.mp4", "cam10/20261020T060130.mkv", "cam2-2026.10.20-06.02.00.avi",
                  "x.mp4", "cam1_2026-10-20_07-00-00.mp4", "cam2_readme.txt"):
            p = os.path.join(d, f)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
        recs, bad = common.recordings(d, ["cam1", "cam2", "cam10"], 2)
        assert [(c, os.path.basename(p)) for c, p, _ in recs] == [
            ("cam1", "cam1_2026-10-20_06-00-00.mp4"), ("cam10", "20261020T060130.mkv"),
            ("cam2", "cam2-2026.10.20-06.02.00.avi"), ("cam1", "cam1_2026-10-20_07-00-00.mp4")]
        assert recs[0][2] == 1792468800.0                     # 06:00 at UTC+2
        assert [os.path.basename(b) for b in bad] == ["x.mp4"]
        assert [[c for c, _, _ in g] for g in common.sessions(recs)] == [["cam1", "cam10", "cam2"], ["cam1"]]


def test_frames_page_and_export_for_the_detector():
    import box_dataset
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        out = os.path.join(d, "boxes")
        sys.modules["detector"] = types.SimpleNamespace(Live=_StubLive)
        try:
            box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10"])
        finally:
            del sys.modules["detector"]
        index = json.load(open(os.path.join(out, "frames.json")))
        fr = index["frames"]
        assert {f["cam"] for f in fr} == {"cam1", "cam2"} and len(fr) == 18          # 6 a minute-long recording
        assert fr[0]["time"] == "2026-10-20 06:00:05" and fr[0]["w"] == 320           # cut to frame_width
        from PIL import Image
        im = np.asarray(Image.open(os.path.join(out, fr[0]["file"])))
        assert abs(int(im[10, 160, 0]) - 114) < 8                                   # cam1's exclude zone grey
        assert len(fr[0]["boxes"]) == 2 and index["threshold"] == 0.5                # the sure one and the offer
        html = open(os.path.join(out, "boxes.html"), encoding="utf-8").read()
        data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
        assert len(data["frames"]) == 18 and "Save labels.json" in html
        # again: nothing new is cut or detected
        box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10", "--detector", "none"])
        assert len(json.load(open(os.path.join(out, "frames.json")))["frames"]) == 18
        # labels: done frames are exported, one skipped is not; held out by whole hours of a camera
        labels = {f["file"]: {"status": "done", "boxes": [[0.1, 0.4, 0.3, 0.8]]} for f in fr}
        labels[fr[1]["file"]] = {"status": "skip", "boxes": []}
        labels[fr[2]["file"]] = {"status": "", "boxes": [[0.1, 0.1, 0.2, 0.2]]}       # edited, not marked done
        lp = os.path.join(d, "labels.json")
        json.dump({"frames": labels}, open(lp, "w"))
        box_dataset.main(["page", "--data", out, "--labels", lp])
        tgz = os.path.join(d, "herd_det_barn_test.tgz")
        box_dataset.main(["export", "--data", out, "--labels", lp, "--out", tgz])
        work = os.path.join(d, "work")
        with tarfile.open(tgz) as t:
            t.extractall(work)
        root = os.path.join(work, "extra_det", "barn")
        tr = json.load(open(os.path.join(root, "train", "_annotations.coco.json")))
        va = json.load(open(os.path.join(root, "valid", "_annotations.coco.json")))
        assert len(tr["images"]) + len(va["images"]) == 16 and va["images"] and tr["images"]
        hours = lambda c: {(im["file_name"][:4], im["file_name"][14:16]) for im in c["images"]}
        assert not hours(tr) & hours(va)                                              # whole hours held out
        a = tr["annotations"][0]
        assert a["bbox"] == [32.0, 72.0, 64.0, 72.0] and os.path.exists(os.path.join(root, "train", tr["images"][0]["file_name"]))
        import extra_det
        assert extra_det._sources("barn", os.path.join(work, "extra_det")) == ["barn"]   # eval takes it


def test_barn_video_plays_sessions_into_a_cache_of_its_own():
    import barn_dataset
    import pipeline
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        calls = []

        class _M:
            def __init__(self, c):
                calls.append(("models",))

        def _run(c, start, models=None):
            calls.append(("run", start, [(x["id"], os.path.basename(x["url"]), x["start_ts"]) for x in c["cameras"]],
                          c["store"]["path"], dict(c["training_cache"]), c["sampling"]["burst_every_s"]))
            return {x["id"]: {"id_confirmed": 1} for x in c["cameras"]}

        old = pipeline.Models, pipeline.run
        pipeline.Models, pipeline.run = _M, _run
        try:
            out = os.path.join(d, "vid")
            barn_dataset.main(["video", "--config", cfg, "--videos", rec, "--out", out])
            runs = [c for c in calls if c[0] == "run"]
            assert len(runs) == 2 and sum(1 for c in calls if c[0] == "models") == 1     # the models loaded once
            first = runs[0]
            assert [c[0] for c in first[2]] == ["cam1", "cam2"] and first[2][1][2] - first[2][0][2] == 10
            assert first[1] == first[2][0][2]                                         # the session's first start
            assert first[3] == os.path.join(out, "herd.sqlite")                       # not the barn's database
            assert first[4]["enabled"] and first[4]["every_n"] == 1 and first[4]["folder"].startswith(out)
            assert first[5] == 20.0
            calls.clear()
            barn_dataset.main(["video", "--config", cfg, "--videos", rec, "--out", out])   # all played: nothing
            assert not [c for c in calls if c[0] == "run"]
        finally:
            pipeline.Models, pipeline.run = old
