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
        for part in ("train", "val"):
            reid.write_split(os.path.join(d, "feat", "multicam", part), [b for b in bursts if b["split"] == part],
                             feats, "multicam", meta)
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
