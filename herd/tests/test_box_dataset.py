"""Datasets from the barn's recordings: frames + boxes for the detector, the cache for identity."""
import _paths  # noqa: F401
import collections
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


def test_trial_two_passes_and_a_night_between():
    import gallery as gallery_mod
    import pipeline
    import show_ids
    import trial
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        passes, drawn = [], []

        class _M:
            id_abstain, id_thresholds = None, {}

            def __init__(self, c):
                pass

        def _run(c, start, models=None, status=None):
            store = Store(c["store"]["path"])
            folder = os.path.join(os.path.dirname(c["store"]["path"]), "gallery")
            g = gallery_mod.Gallery(folder, c)
            first = not g.cows
            passes.append((c["sampling"]["burst_every_s"], [x["id"] for x in c["cameras"]], first))
            for x in c["cameras"]:
                tr = f"{x['id']}-1"
                store.seconds([(x["id"], tr, start + s, 0.1, 0.4, 0.3, 0.8, "lying", "none", 0.9) for s in range(10)])
                if first:            # the first pass: her bursts wait in the pool
                    v = np.zeros(8, np.float32)
                    v[0] = 1
                    g.pool += [{"ts": start + s, "track": tr, "fp": v} for s in range(8)]
                else:                # the second: she is known
                    cow = next(iter(g.cows))
                    for s in range(3):
                        store.burst((x["id"], tr, start + s, "confirmed", cow, 0.99, 0.3, 0.9))
            g.save()
            return {}
        old = pipeline.Models, pipeline.run, show_ids.main
        pipeline.Models, pipeline.run = _M, _run
        show_ids.main = lambda argv: drawn.append(argv[argv.index("--cam") + 1])
        try:
            out = os.path.join(d, "trial")
            trial.main(["--config", cfg, "--videos", rec, "--out", out, "--min-bursts", "6"])
        finally:
            pipeline.Models, pipeline.run, show_ids.main = old
        assert [p[0] for p in passes] == [20.0, 20.0, 60, 60]          # 2 sessions a pass; the second as configured
        assert [p[2] for p in passes] == [True, True, False, False]
        rep = json.load(open(os.path.join(out, "trial.json")))
        assert len(rep["cows_enrolled"]) == 1 and rep["cows_in_gallery"] == 1
        assert rep["cameras"]["cam1"]["with_a_cow"] == 1.0 and rep["cameras"]["cam1"]["posture"] == {"lying": 20}
        assert sorted(drawn) == ["cam1", "cam2"]


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

        def _run(c, start, models=None, status=None):
            if interrupt and status is not None:
                status["interrupted"] = True
            calls.append(("run", start, [(x["id"], os.path.basename(x["url"]), x["start_ts"]) for x in c["cameras"]],
                          c["store"]["path"], dict(c["training_cache"]), c["sampling"]["burst_every_s"]))
            return {x["id"]: {"id_confirmed": 1} for x in c["cameras"]}

        old = pipeline.Models, pipeline.run
        pipeline.Models, pipeline.run = _M, _run
        try:
            out = os.path.join(d, "vid")
            interrupt = True                              # Ctrl+C: the session is not marked done
            barn_dataset.main(["video", "--config", cfg, "--videos", rec, "--out", out + "\\"])
            assert not os.path.exists(os.path.join(out, "videos_done.txt")) and os.path.isdir(out)
            calls.clear()
            interrupt = False
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


def test_record_names_files_as_the_tools_read_them():
    import common
    import record
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.mp4")
        _video(src, 4)
        cfg = os.path.join(d, "barn.toml")
        open(cfg, "w").write(f'[farm]\ntimezone_offset_hours = 2\n[[cameras]]\nid = "cam1"\nurl = "{src}"\n'
                             f'[[cameras]]\nid = "cam2"\nurl = "{src}"\n')
        import shutil
        if not shutil.which("ffmpeg"):
            assert record.command("rtsp://x", 60, "o.mp4")[5:7] == ["-rtsp_transport", "tcp"]
            return
        out = os.path.join(d, "rec")
        assert record.main(["--config", cfg, "--out", out, "--minutes", "0.05", "--cameras", "cam2"]) == 0
        recs, bad = common.recordings(out, ["cam1", "cam2"], 2)
        assert [c for c, _, _ in recs] == ["cam2"] and not bad


def test_zones_page_and_apply():
    import common
    import zone_page
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        sys.modules["detector"] = types.SimpleNamespace(Live=lambda *a, **k: (lambda img: [{"bbox": [0.1, 0.4, 0.3, 0.8], "det_score": 0.9}]))
        try:
            zone_page.main(["page", "--config", cfg, "--videos", rec, "--out", os.path.join(d, "zones")])
        finally:
            del sys.modules["detector"]
        html = open(os.path.join(d, "zones", "zones.html"), encoding="utf-8").read()
        data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
        cams = {c["id"]: c for c in data["cams"]}
        assert set(cams) == {"cam1", "cam2"} and cams["cam1"]["w"] == 320 and cams["cam1"]["boxes"]
        assert cams["cam1"]["exclude"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 0.2], [0.0, 0.2]]      # from the config
        assert os.path.exists(os.path.join(d, "zones", cams["cam2"]["still"]))
        # a camera not in the config yet (cam3's recording), named on the command line
        import shutil as sh
        sh.copy(os.path.join(rec, "cam1_2026-10-20_06-00-00.mp4"), os.path.join(rec, "cam3_2026-10-20_06-00-00.mp4"))
        zone_page.main(["page", "--config", cfg, "--videos", rec, "--cameras", "cam1", "cam2", "cam3",
                        "--detector", "none", "--out", os.path.join(d, "zones3")])
        html = open(os.path.join(d, "zones3", "zones.html"), encoding="utf-8").read()
        data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
        assert [c["id"] for c in data["cams"]] == ["cam1", "cam2", "cam3"] and data["cams"][2]["still"]
        # the page's Save -> barn.toml
        ct = os.path.join(d, "cameras.toml")
        open(ct, "w").write('[[cameras]]\nid = "cam1"\nurl = "rtsp://a"\nmask = [[0, 0], [1, 0], [1, 1]]\n'
                            'exclude = [[[0, 0], [1, 0], [1, 0.1]], [[0, 0.9], [1, 0.9], [1, 1]]]\n\n'
                            '[[cameras]]\nid = "cam3"\nurl = "rtsp://c"\nfloor_points = [[0.1, 0.9, 0, 0]]\n')
        zone_page.main(["apply", "--config", cfg, "--cameras", ct])
        got = common.load_config(cfg)
        assert [c["id"] for c in got["cameras"]] == ["cam1", "cam3"] and len(got["cameras"][0]["exclude"]) == 2
        assert got["detector"]["weights"].endswith("det") and os.path.exists(cfg + ".bak")


def test_file_frames_on_the_recordings_own_clock():
    import pipeline
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v.mp4")
        _video(path, 3, fps=5)
        ts = [t for t, _ in pipeline.frames_from(path, 1000.0)]
        assert len(ts) == 15 and ts[0] == 1000.0 and abs(ts[-1] - 1002.8) < 0.05
        assert all(b > a for a, b in zip(ts, ts[1:]))


def test_fix_videos_command():
    import fix_videos
    enc = fix_videos.command("in.mkv", "out.mp4", 25.0)
    assert enc[:2] == ["ffmpeg", "-nostdin"] and "discardcorrupt" in " ".join(enc) and "setpts=PTS-STARTPTS,fps=25.0" in enc
    assert enc[-1] == "out.mp4" and "libx264" in enc
    cp = fix_videos.command("in.mkv", "out.mp4", 25.0, copy=True)
    assert "copy" in cp and "libx264" not in cp


def test_one_camera_cut_again_after_its_zone_changed():
    import box_dataset
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        out = os.path.join(d, "boxes")
        box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10", "--detector", "none"])
        fr = json.load(open(os.path.join(out, "frames.json")))["frames"]
        lp = os.path.join(d, "labels.json")
        # labelled: one box in the top band, one below it
        json.dump({"frames": {f["file"]: {"status": "done", "boxes": [[0.1, 0.4, 0.3, 0.8], [0.4, 0.3, 0.5, 0.36]]}
                              for f in fr}}, open(lp, "w"))
        # cam1's exclude zone grows to y 0.4: its frames cut again, the detector looks again
        text = open(cfg).read()
        open(cfg, "w").write(text.replace("[1.0, 0.2], [0.0, 0.2]", "[1.0, 0.4], [0.0, 0.4]"))
        sys.modules["detector"] = types.SimpleNamespace(Live=_StubLive)
        try:
            box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10",
                              "--cameras", "cam1", "--again"])
        finally:
            del sys.modules["detector"]
        fr2 = json.load(open(os.path.join(out, "frames.json")))["frames"]
        c1 = [f for f in fr2 if f["cam"] == "cam1"]
        assert len(fr2) == len(fr) and all(f.get("cut") and f["exclude"][2] == [1.0, 0.4] and f["boxes"] for f in c1)
        assert not any(f.get("cut") for f in fr2 if f["cam"] == "cam2")
        from PIL import Image
        im = np.asarray(Image.open(os.path.join(out, c1[0]["file"])))
        assert abs(int(im[int(0.3 * im.shape[0]), 160, 0]) - 114) < 8                # the new grey
        # export: cam1's labels are from before the new cut - left out until checked; cam2's go
        tgz = os.path.join(d, "x.tgz")
        box_dataset.main(["export", "--data", out, "--labels", lp, "--out", tgz, "--block-min", "1"])
        with tarfile.open(tgz) as t:
            names = [m.name for m in t.getmembers() if m.name.endswith(".jpg")]
        assert names and all("cam2_" in n for n in names)
        # checked on the page (the label carries the cut): back in, the box now in the grey left out
        labels = json.load(open(lp))["frames"]
        for f in c1:
            labels[f["file"]]["cut"] = f["cut"]
        json.dump({"frames": labels}, open(lp, "w"))
        box_dataset.main(["export", "--data", out, "--labels", lp, "--out", tgz, "--block-min", "1"])
        work = os.path.join(d, "w")
        with tarfile.open(tgz) as t:
            t.extractall(work)
        coco = [json.load(open(os.path.join(work, "extra_det", "barn", s, "_annotations.coco.json"))) for s in ("train", "valid")]
        per = collections.Counter(a["image_id"] for c in coco for a in c["annotations"])
        ims = {(s, im["id"]): im["file_name"] for s, c in zip(("t", "v"), coco) for im in c["images"]}
        assert any(n.startswith("cam1_") for n in ims.values())
        cam1_boxes = [len([a for a in c["annotations"] if a["image_id"] == im["id"]]) for c in coco for im in c["images"]
                      if im["file_name"].startswith("cam1_")]
        assert cam1_boxes and set(cam1_boxes) == {1}                                 # the one in the grey dropped
        cam2_boxes = [len([a for a in c["annotations"] if a["image_id"] == im["id"]]) for c in coco for im in c["images"]
                      if im["file_name"].startswith("cam2_")]
        assert set(cam2_boxes) == {2}


def test_barn_video_one_camera_again():
    import barn_dataset
    import pipeline
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        out = os.path.join(d, "vid")
        played = []

        def _run(c, start, models=None, status=None):
            played.append([x["id"] for x in c["cameras"]])
            return {}
        old = pipeline.Models, pipeline.run
        pipeline.Models, pipeline.run = (lambda c: object()), _run
        try:
            barn_dataset.main(["video", "--config", cfg, "--videos", rec, "--out", out])
            # an earlier burst of each camera in the cache
            day = os.path.join(out, "training_cache", "2026-10-20")
            for cam in ("cam1", "cam2"):
                os.makedirs(os.path.join(day, cam), exist_ok=True)
                open(os.path.join(day, cam, "x.npz"), "w").close()
            with open(os.path.join(day, "index.jsonl"), "w") as fh:
                for cam in ("cam1", "cam2"):
                    fh.write(json.dumps({"file": f"2026-10-20/{cam}/x.npz", "cam": cam, "track": cam + "-1", "day": "2026-10-20"}) + "\n")
            played.clear()
            barn_dataset.main(["video", "--config", cfg, "--videos", rec, "--out", out, "--cameras", "cam1"])
            assert played == [["cam1"], ["cam1"]]                 # both cam1 recordings, cam2 not
            rows = barn_dataset.read_cache(os.path.join(out, "training_cache"))
            assert [r["cam"] for r in rows] == ["cam2"] and not os.path.exists(os.path.join(day, "cam1"))
        finally:
            pipeline.Models, pipeline.run = old


def test_boxes_named_by_cow_go_to_the_identity_model():
    import box_dataset
    import reid
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        out = os.path.join(d, "boxes")
        box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10", "--detector", "none"])
        fr = json.load(open(os.path.join(out, "frames.json")))["frames"]
        tags = ["UA01", "UA02", "UA03", "UA04", "UA05", "UA06"]
        labels = {}
        for k, f in enumerate(fr):            # two named cows a frame, one box without an id
            a, b = tags[k % 6], tags[(k + 1) % 6]
            labels[f["file"]] = {"status": "done", "boxes": [[0.1, 0.4, 0.3, 0.8, a], [0.5, 0.4, 0.7, 0.8, b],
                                                             [0.75, 0.5, 0.9, 0.7]]}
        labels[fr[0]["file"]]["status"] = ""                                             # not done: only --done-only drops it
        lp = os.path.join(d, "labels.json")
        json.dump({"frames": labels}, open(lp, "w"))
        tgz = os.path.join(d, "ids.tgz")
        box_dataset.main(["ids", "--data", out, "--labels", lp, "--out", tgz, "--min-crops", "2", "--done-only"])
        work = os.path.join(d, "work")
        with tarfile.open(tgz) as t:
            t.extractall(work)
        R = os.path.join(work, "reid")
        rows = [json.loads(x) for x in open(os.path.join(R, "raw", "barn_ids", "index.jsonl"))]
        assert len(rows) == 2 * (len(fr) - 1) and {r["cow"] for r in rows} == set(tags)
        assert {r["cam"] for r in rows} == {"cam1", "cam2"} and all(r["ready"] for r in rows)
        val = {r["cow"] for r in rows if r["split"] == "val"}
        assert len(val) == 1 and not val & {r["cow"] for r in rows if r["split"] == "train"}
        from PIL import Image
        im = Image.open(os.path.join(R, "raw", "barn_ids", rows[0]["file"]))
        assert im.width == im.height
        # reid: no fetch; extract makes bursts of the crops with the held-out split kept
        reid.main(["fetch", "--data", R, "--sources", "barn_ids"])
        real = reid.encode_all
        reid.encode_all = lambda paths, meta, batch=256, ready=None: np.random.default_rng(0).normal(
            size=(len(paths), 16)).astype(np.float16)
        try:
            import torch
            import model as mdl
            m = mdl.HerdModel(16, d=32, layers=1, heads=2)
            mp = os.path.join(d, "model.pt")
            m.save(mp, {"features": {"encoder": "x", "grid": 2, "crop": 224, "margin": 0.1}, "crop_s": 3.5, "dev_clips": []})
            reid.main(["extract", "--data", R, "--sources", "barn_ids", "--model", mp])
        finally:
            reid.encode_all = real
        bursts = [json.loads(x) for x in open(os.path.join(R, "feat", "barn_ids", "bursts.jsonl"))]
        assert bursts and {b["identity"].split(":")[0] for b in bursts} == {"barn_ids"}
        assert {b["cow"] for b in bursts if b["split"] == "val"} == val


def test_ids_filled_in_between_frames_from_the_video():
    import box_dataset
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        out = os.path.join(d, "boxes")
        box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10", "--detector", "none",
                          "--cameras", "cam2"])
        fr = sorted(json.load(open(os.path.join(out, "frames.json")))["frames"], key=lambda f: f["ts"])
        assert len(fr) == 6
        # cow UA01 lies still (the same box), cow UA02 moved far between frames 2 and 3
        labels = {f["file"]: {"status": "", "boxes": [[0.1, 0.4, 0.3, 0.8, "UA01"],
                                                      [0.5 + (0.3 if k >= 3 else 0), 0.1, 0.6 + (0.3 if k >= 3 else 0), 0.3, "UA02"]]}
                  for k, f in enumerate(fr)}
        lp = os.path.join(d, "labels.json")
        json.dump({"frames": labels}, open(lp, "w"))
        tgz = os.path.join(d, "ids.tgz")
        box_dataset.main(["ids", "--data", out, "--labels", lp, "--out", tgz, "--min-crops", "2", "--fill", "2",
                          "--config", cfg, "--videos", rec])
        with tarfile.open(tgz) as t:
            t.extractall(os.path.join(d, "w"))
        rows = [json.loads(x) for x in open(os.path.join(d, "w", "reid", "raw", "barn_ids", "index.jsonl"))]
        fill = collections.Counter(r["cow"] for r in rows if r["by"] == "fill")
        # 5 gaps of 10 s, a crop every 2 s inside each: 4 a gap; UA02 not across the gap where she moved
        assert fill["UA01"] == 5 * 4 and fill["UA02"] == 4 * 4
        assert len({r["order"] for r in rows if r["cow"] == "UA01"}) == 6 + 20


def test_check_the_models_on_the_barns_labels():
    import box_dataset
    import check
    import detector as det_mod
    import model as mdl
    import reid
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        out = os.path.join(d, "boxes")
        box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10", "--detector", "none"])
        fr = json.load(open(os.path.join(out, "frames.json")))["frames"]
        tags = ["UA01", "UA02", "UA03"]
        labels = {f["file"]: {"status": "done", "boxes": [[0.1, 0.4, 0.3, 0.8, tags[k % 3]], [0.5, 0.4, 0.7, 0.8, tags[(k + 1) % 3]]]}
                  for k, f in enumerate(fr)}
        lp = os.path.join(d, "labels.json")
        json.dump({"frames": labels}, open(lp, "w"))
        box_dataset.main(["export", "--data", out, "--labels", lp, "--out", os.path.join(d, "herd_det_barn_1.tgz"),
                          "--block-min", "1"])
        box_dataset.main(["ids", "--data", out, "--labels", lp, "--out", os.path.join(d, "herd_barn_ids_1.tgz"),
                          "--min-crops", "2"])
        mp = os.path.join(d, "model.pt")
        mdl.HerdModel(16, d=32, layers=1, heads=2).save(mp, {"features": {"encoder": "x", "grid": 2, "crop": 224, "margin": 0.1},
                                                             "crop_s": 3.5, "dev_clips": []})
        text = open(cfg).read()
        open(cfg, "w").write(text.replace("[detector]", f'[model]\ncheckpoint = "{mp}"\n\n[detector]'))
        real_live, real_enc = det_mod.Live, reid.encode_all
        det_mod.Live = _StubLive
        reid.encode_all = lambda paths, meta, batch=256, ready=None: np.random.default_rng(0).normal(size=(len(paths), 16)).astype(np.float16)
        try:
            check.main(["--config", cfg])
        finally:
            det_mod.Live, reid.encode_all = real_live, real_enc
        rep = json.load(open(os.path.join(d, "check", "check.json")))
        r = rep["detector"]["det"]
        assert r["at_own"]["recall"] == 0.5 and r["at_best"]["precision"] > 0              # stub: 1 of the 2 cows
        ident = rep["identity"]["herd model (no identity model)"]
        assert "other_camera_top1" in ident["metrics"] and "thresholds" in ident


def test_ids_gallery_rows_by_camera():
    import box_dataset
    with tempfile.TemporaryDirectory() as d:
        cfg, rec = _setup(d)
        out = os.path.join(d, "boxes")
        box_dataset.main(["frames", "--config", cfg, "--videos", rec, "--out", out, "--every", "10", "--detector", "none"])
        fr = json.load(open(os.path.join(out, "frames.json")))["frames"]
        labels = {f["file"]: {"status": "done", "boxes": [[0.1, 0.4, 0.3, 0.8, "4"], [0.5, 0.4, 0.7, 0.8, "7", 1]]} for f in fr}
        lp = os.path.join(d, "labels.json")
        json.dump({"frames": labels}, open(lp, "w"))
        box_dataset.main(["gallery", "--data", out, "--labels", lp, "--thumbs", "3"])
        html = open(os.path.join(out, "ids.html"), encoding="utf-8").read()
        data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
        rows = {(r["cam"], r["id"]): r for r in data["rows"]}
        assert set(rows) == {("cam1", "4"), ("cam2", "4"), ("cam1", "7"), ("cam2", "7")}
        assert rows[("cam1", "7")]["auto"] == rows[("cam1", "7")]["n"] == 12 and len(rows[("cam1", "4")]["thumbs"]) == 3
        assert all(os.path.exists(os.path.join(out, t["src"])) for r in data["rows"] for t in r["thumbs"])
