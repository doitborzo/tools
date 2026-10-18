"""Identity on sets with cow ids: pseudo-bursts, training a copy of the temporal model, its NaN model."""
import _paths  # noqa: F401
import json
import os
import random
import tempfile

import numpy as np


def test_plan_reads_each_sets_layout():
    import reid
    names = ["x/MultiCamCows2024Root/2023Aug14/018/00026_2.jpg",
             "x/MultiCamCows2024Root/videos/camera1/2023Aug14/a.mp4",
             "y/Sub-levels/Identification/Test/068/image_0001818_2020-02-11_14-2-17_roi_003.jpg",
             "y/Sub-levels/Identification/Train/RGBDCows2020/Identification/RGB/263/0/r_1.jpg",
             "z/identification/images/train/007/abc.jpg"]
    assert reid.plan("multicam", names) == [("018", "2023Aug14", "2", 26, names[0])]
    assert reid.plan("cows2021", names) == [("068", "2020-02-11", "0", 1818, names[2])]   # the id-less part is left out
    assert reid.plan("opencows", names)[0][:3] == ("007", "all", "0")


def test_bursts_by_camera_and_day_and_the_stretch():
    import reid
    rows = [{"cow": c, "day": d, "cam": cam, "order": k, "k": i}
            for i, (c, d, cam, k) in enumerate((c, d, cam, k) for c in "AB" for d in ("d1", "d2")
                                               for cam in ("1", "2") for k in range(40))]
    bs = reid.make_bursts(rows, random.Random(0), {"B"}, tracklets=True)
    assert len(bs) == 2 * 2 * 2                      # 40 crops: one window of 35, the tail of 5 joins it
    assert {b["split"] for b in bs if b["cow"] == "B"} == {"val"}
    assert all(len(b["frames"]) == 40 for b in bs)
    s = reid.stretch(40)
    assert len(s) == reid.SLOTS and s[0] == 0 and s[-1] == 39 and np.all(np.diff(s) >= 0)


def test_identity_training_knows_a_cow_on_another_day():
    import torch
    import model as mdl
    import reid
    rng = np.random.default_rng(0)
    dim, cows, days, cams = 24, [f"{i:03d}" for i in range(20)], ["d1", "d2", "d3"], ["1", "2"]
    vec = {c: rng.normal(size=dim) * 1.5 for c in cows}
    rows, feats = [], []
    for c in cows:
        for d in days:
            for cam in cams:
                view = rng.normal(size=dim) * 1.5          # this day's and camera's look of her
                for k in range(12):
                    rows.append({"cow": c, "day": d, "cam": cam, "order": k, "k": len(feats)})
                    feats.append(vec[c] + view + rng.normal(size=dim) * 0.3)
    feats = np.array(feats, np.float16)
    with tempfile.TemporaryDirectory() as d:
        val = set(cows[:5])
        bursts = reid.make_bursts(rows, random.Random(0), val, tracklets=True)
        meta = {"encoder": "x", "grid": 2, "crop": 224, "margin": 0.1, "dim": dim}
        folder = os.path.join(d, "feat", "multicam")
        reid.write_bursts(folder, bursts, "multicam", meta)
        np.savez(os.path.join(folder, "crops.npz"), feats=feats)          # each crop's vector once
        tr = reid.Bursts(d, ["multicam"], "train")
        assert len(tr.bursts) == 15 * 3 * 2 and tr.whole(0).shape == (reid.SLOTS, dim)
        torch.manual_seed(0)
        m = mdl.HerdModel(dim, d=32, layers=1, heads=2)
        mp = os.path.join(d, "model.pt")
        m.save(mp, {"features": meta, "crop_s": 3.5, "dev_clips": []})
        reid.main(["train", "--data", d, "--sources", "multicam", "--model", mp, "--epochs", "6", "--P", "16",
                   "--patience", "6"])
        ev = json.load(open(os.path.join(d, "id_model", "eval.json")))
        assert ev["after"]["other_day_top1"] >= ev["before"]["other_day_top1"]
        assert ev["after"]["other_day_top1"] > 0.6, ev
        ab = json.load(open(os.path.join(d, "id_model", "abstain.json")))
        assert ab["features"] and 0 <= ab["threshold"] <= 1.01
        idm, abl = reid.load_identity(os.path.join(d, "id_model", "identity.pt"), meta, torch.device("cpu"))
        x = torch.randn(2, 5, dim)
        out = idm(x, torch.zeros(2, 5), torch.ones(2, 5, dtype=torch.bool))
        assert out["fingerprint"].shape == (2, mdl.ID_DIM) and abl["threshold"] == ab["threshold"]
        # fine-tuning that model (--init): it starts where the first training ended
        reid.main(["train", "--data", d, "--sources", "multicam", "--model", mp, "--epochs", "2", "--P", "16",
                   "--init", os.path.join(d, "id_model", "identity.pt"), "--out", os.path.join(d, "id_model_ft")])
        ev2 = json.load(open(os.path.join(d, "id_model_ft", "eval.json")))
        assert abs(ev2["before"]["other_day_top1"] - ev["after"]["other_day_top1"]) < 1e-6
        assert ev2["after"]["score"] >= ev2["before"]["score"]
        ck = torch.load(os.path.join(d, "id_model_ft", "identity.pt"), weights_only=False)
        assert ck["init"].endswith("identity.pt") and ck["sources"] == ["multicam"]


def test_plan_reads_cobra_and_sideview():
    import reid
    names = ["lying_right/0017/0002.png", "standing/0023/0101.jpg", "lying_right/0017/0002.json",
             "barn/images/594/594_0977.jpg", "barn/images/594/594_0980.jpg", "barn/masks/594/594_0980.png",
             "parlor/images/12/12_0020.jpg", "parlor/images/12/12_0025.jpg", "snapshots/images/7/7_0003.jpg"]
    assert reid.plan("cobra", names) == [("0017", "lying_right", "0", 2, names[0]),
                                         ("0023", "standing", "0", 101, names[1])]   # the posture is its "day"
    got = reid.plan("sideview", names)
    # video frames thinned (barn every 5th, parlor every 10th), the subset is the "day", masks left out
    assert [(g[0], g[1], g[3]) for g in got] == [("594", "barn", 980), ("12", "parlor", 20), ("7", "snapshots", 3)]


def test_mask_crop_cuts_the_masked_cow():
    import io
    from PIL import Image
    import reid
    img = np.zeros((300, 400, 3), np.uint8)
    img[100:200, 50:250] = 200
    mask = np.zeros((300, 400), np.uint8)
    mask[100:200, 50:250] = 255
    enc = lambda a, f: (lambda b: (Image.fromarray(a).save(b, f), b.getvalue())[1])(io.BytesIO())
    out = Image.open(io.BytesIO(reid.mask_crop(enc(img, "JPEG"), enc(mask, "PNG"))))
    assert 200 <= out.size[0] <= 220 and 100 <= out.size[1] <= 112
    assert reid.mask_crop(enc(img, "JPEG"), enc(np.zeros_like(mask), "PNG")) is None    # no cow, no crop
    assert reid.as_jpeg(enc(img, "PNG"))[:3] == b"\xff\xd8\xff"


def test_tag_tracks_labels_the_tagged_cows_only():
    """Three cows lying still, two with a neck tag ~80 cm along the stall from their body's centre, one
    without; a camera looking straight down (world cm -> pixels of MmCows' 4480 x 2800 frame)."""
    import reid
    from common import DEFAULTS, floor_mapper
    P = [2.0, 0, 0, 2000, 0, 2.0, 0, 1400, 0, 0, 0, 1]
    floor = floor_mapper({"proj_mat": P, "proj_size": reid.MM_PROJ}, DEFAULTS["mct"])
    P = np.asarray(P).reshape(3, 4)
    body = {1: (-300.0, 0.0), 2: (0.0, 0.0), 13: (300.0, 0.0)}           # cow 13 wears no tag

    def box(x, y):
        u, v = (x + 1000) * 2 / 4480, (y + 700) * 2 / 2800
        return [u - 0.04, v - 0.05, u + 0.04, v + 0.05]
    rng = np.random.default_rng(0)
    frames, tags, truth = [], {}, {}
    for k in range(40):
        ts = 1690000011 + 15 * k
        order = [1, 2, 13] if k % 2 else [13, 2, 1]                          # the detector's order varies
        frames.append((ts, [box(*body[c]) for c in order]))
        truth.update({(ts, j): c for j, c in enumerate(order)})
        tags[ts] = [(t, [body[t][0] + rng.normal(0, 30), body[t][1] + 80 + rng.normal(0, 30), 100.0]) for t in (1, 2)]
    got = reid.tag_tracks(frames, tags, floor, P)
    assert got and all(truth[key] == t for key, t in got.items())
    assert {t for t in got.values()} == {1, 2} and 13 not in got.values()
    assert sum(1 for key in got if truth[key] == 1) == 40                    # every second of her track
    assert reid._every(1690000011, 30) != reid._every(1690000026, 30) and reid._every(1690000011, 15)


def test_preset_split_keeps_mmcows_held_out_cows():
    import reid
    rows = [{"cow": c, "day": "0725", "cam": cam, "order": k, "k": i, "split": "val" if c == "13" else "train"}
            for i, (c, cam, k) in enumerate((c, cam, k) for c in ("01", "13") for cam in ("cam_1", "cam_2")
                                            for k in range(16))]
    bs = reid.make_bursts(rows, random.Random(0), set(), tracklets=False)
    assert {b["split"] for b in bs if b["cow"] == "13"} == {"val"}
    assert {b["split"] for b in bs if b["cow"] == "01"} == {"train"}
    assert all(len(b["frames"]) == reid.PICS for b in bs)


def test_label_command_human_and_tag_days():
    """reid.py label end to end with a stand-in detector: 7/25 by the human labels (cow 13, untagged,
    held out), another day by the neck tags, and 7/25's tag labels checked against the human ones."""
    from PIL import Image
    import pipeline
    import reid
    body = {1: (-300.0, 0.0), 2: (0.0, 0.0), 13: (300.0, 0.0)}

    def box(x, y):
        u, v = (x + 1000) * 2 / 4480, (y + 700) * 2 / 2800
        return [u - 0.04, v - 0.05, u + 0.04, v + 0.05]
    xml = ("<?xml version=\"1.0\" ?>\n<opencv_storage>\n<projection_matrix type_id=\"opencv-matrix\">\n<rows>3</rows>"
           "<cols>4</cols><dt>d</dt>\n<data>2.0 0 0 2000 0 2.0 0 1400 0 0 0 1</data></projection_matrix>\n</opencv_storage>\n")

    class Stub:
        crop_size, margin = 32, 0.1

        def __init__(self, cfg):
            pass

        def detect_many(self, imgs, batch=8):
            return [[box(*body[c]) for c in (13, 1, 2)] for _ in imgs]
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "mmcows")
        for day in ("0725", "0722"):
            os.makedirs(os.path.join(root, "proj_mat", day))
            for c in range(1, 5):
                open(os.path.join(root, "proj_mat", day, f"proj_mat_cam{c}.xml"), "w").write(xml)
            for t in (1, 2):
                os.makedirs(os.path.join(root, "uwb", day), exist_ok=True)
                with open(os.path.join(root, "uwb", day, f"T{t:02d}.csv"), "w") as fh:
                    fh.write("timestamp,coord_x_cm,coord_y_cm,coord_z_cm\n")
                    for k in range(12):
                        fh.write(f"{1690000011 + 30 * k}.0,{body[t][0]},{body[t][1] + 70},100.0\n")
            for cam in ("cam_1", "cam_2"):
                os.makedirs(os.path.join(root, "frames", day, cam))
                os.makedirs(os.path.join(root, "labels", cam), exist_ok=True)
                for k in range(12):
                    ts = 1690000011 + 30 * k
                    Image.new("RGB", (448, 280), (90, 90, 90)).save(os.path.join(root, "frames", day, cam, f"{ts}.jpg"))
                    if day == "0725":
                        with open(os.path.join(root, "labels", cam, f"{ts}.txt"), "w") as fh:
                            for c, (x, y) in body.items():
                                b = box(x, y)
                                fh.write(f"{c} {(b[0] + b[2]) / 2} {(b[1] + b[3]) / 2} {b[2] - b[0]} {b[3] - b[1]}\n")
        real = pipeline.Models
        pipeline.Models = Stub
        try:
            reid.main(["label", "--data", d, "--model", "unused.pt", "--detector", "unused"])
        finally:
            pipeline.Models = real
        rows = [json.loads(x) for x in open(os.path.join(d, "raw", "mmcows", "index.jsonl"))]
        human = [r for r in rows if r["by"] == "human"]
        assert {(r["cow"], r["split"]) for r in human} == {("01", "train"), ("02", "train"), ("13", "val")}
        assert len(human) == 3 * 2 * 12
        tagged = [r for r in rows if r["by"] == "uwb"]
        assert tagged and {r["cow"] for r in tagged} == {"01", "02"} and {r["day"] for r in tagged} == {"0722"}
        assert all(os.path.exists(os.path.join(d, "raw", "mmcows", r["file"])) for r in rows)
        check = json.load(open(os.path.join(root, "label.json")))["tag_check"]
        assert check["tag_labels"] > 0 and check["tag_right"] == check["tag_labels"]
        # one day again (the pod labels a day while the next downloads): the index keeps every day
        pipeline.Models = Stub
        try:
            reid.main(["label", "--data", d, "--model", "unused.pt", "--detector", "unused", "--mm-days", "0722",
                       "--force"])
        finally:
            pipeline.Models = real
        again = [json.loads(x) for x in open(os.path.join(d, "raw", "mmcows", "index.jsonl"))]
        assert len(again) == len(rows) and json.load(open(os.path.join(root, "label.json")))["tag_check"] == check


def test_moments_loss_and_the_models_own_thresholds():
    """MmCows-like seconds (each cow seen by 3 cameras, each camera its own look of her): training with
    the frame-against-frame loss, the moment metric on held-out cows, and thresholds calibrated on them
    - stored with the model and applied where the config keeps the default."""
    import copy as _copy
    import torch
    import model as mdl
    import reid
    from common import DEFAULTS
    rng = np.random.default_rng(1)
    dim, cows, cams = 24, [f"{i:02d}" for i in range(1, 15)], ["cam_1", "cam_2", "cam_3"]
    vec = {c: rng.normal(size=dim) * 1.5 for c in cows}
    look = {(c, cam): rng.normal(size=dim) * 0.8 for c in cows for cam in cams}
    rows, feats = [], []
    for day in ("0722", "0725"):
        for s in range(30):
            ts = 1690000011 + 60 * s
            for c in cows:
                held = int(c) > 10
                if held and day != "0725":
                    continue
                for cam in cams:
                    rows.append({"cow": c, "day": day, "cam": cam, "order": ts, "file": f"{c}/{day}_{cam}_{ts}.jpg",
                                 "ready": True, "split": "val" if held else "train", "k": len(feats)})
                    feats.append(vec[c] + look[(c, cam)] + rng.normal(size=dim) * 0.4)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "raw", "mmcows"))
        with open(os.path.join(d, "raw", "mmcows", "index.jsonl"), "w") as fh:
            fh.writelines(json.dumps({k: v for k, v in r.items() if k != "k"}) + "\n" for r in rows)
        meta = {"encoder": "x", "grid": 2, "crop": 224, "margin": 0.1, "dim": dim}
        folder = os.path.join(d, "feat", "mmcows")
        reid.write_bursts(folder, reid.make_bursts(rows, random.Random(0), set(), tracklets=False), "mmcows", meta)
        np.savez(os.path.join(folder, "crops.npz"), feats=np.array(feats, np.float16))
        mo = reid.Moments(d, "val")
        assert len(mo) == 30 and set(mo.ids) == {"11", "12", "13", "14"}
        torch.manual_seed(0)
        m = mdl.HerdModel(dim, d=32, layers=1, heads=2)
        mp = os.path.join(d, "model.pt")
        m.save(mp, {"features": meta, "crop_s": 3.5, "dev_clips": []})
        reid.main(["train", "--data", d, "--sources", "mmcows", "--model", mp, "--epochs", "4", "--P", "16",
                   "--patience", "4", "--moments", "4"])
        ev = json.load(open(os.path.join(d, "id_model", "eval.json")))
        assert "moment_cross_cam_top1" in ev["after"] and ev["after"]["moment_queries"] > 0
        assert ev["after"]["moment_cross_cam_top1"] >= ev["before"]["moment_cross_cam_top1"]
        th = reid.identity_thresholds(os.path.join(d, "id_model", "identity.pt"))
        for k in ("reacquire_sim", "handover_min_sim", "late_sim", "new_cow_similarity", "merge_similarity"):
            assert -1 <= th[k] <= 1 and 0 <= th["recall"][k] <= 1, (k, th)
        assert th["merge_similarity"] >= th["new_cow_similarity"]
        cfg = _copy.deepcopy(DEFAULTS)
        cfg["mct"]["handover_min_sim"] = 0.42                     # set in barn.toml: kept
        changed = reid.apply_thresholds(cfg, th, DEFAULTS, say=None)
        assert cfg["mct"]["reacquire_sim"] == th["reacquire_sim"] and cfg["mct"]["late_sim"] == 0.0   # late: off
        assert cfg["mct"]["handover_min_sim"] == 0.42 and cfg["identity"]["new_cow_similarity"] == th["new_cow_similarity"]
        assert not any(c.startswith("mct.handover_min_sim") for c in changed)
        # thresholds again for the trained model (a change to calibrate, no training)
        os.remove(os.path.join(d, "id_model", "thresholds.json"))
        reid.main(["calibrate", "--data", d, "--identity", os.path.join(d, "id_model", "identity.pt")])
        again = reid.identity_thresholds(os.path.join(d, "id_model", "identity.pt"))
        assert abs(again["reacquire_sim"] - th["reacquire_sim"]) < 1e-3
