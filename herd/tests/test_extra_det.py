"""The public detector sets: label formats, splits, zip members read by range, the merge into RF-DETR's COCO."""
import _paths  # noqa: F401
import io
import json
import os
import tempfile
import zipfile

import extra_det as x


def test_yolo_boxes_clipped():
    b = x.yolo_boxes("0 0.5 0.5 0.2 0.4\r\n1 0.05 0.5 0.2 0.2\n\n0 0.5 0.5 0 0\nbad line\n")
    assert b == [[0.4, 0.3, 0.6, 0.7], [0.0, 0.4, 0.15, 0.6]]


def test_read_member_by_range():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a/stored.txt", b"cow " * 50, compress_type=zipfile.ZIP_STORED)
        z.writestr("a/deflated.txt", b"lying " * 500, compress_type=zipfile.ZIP_DEFLATED)
    blob = buf.getvalue()
    get = lambda s, e: blob[s:min(e, len(blob) - 1) + 1]
    z = zipfile.ZipFile(io.BytesIO(blob))
    assert x.read_member(get, z.getinfo("a/stored.txt")) == b"cow " * 50
    assert x.read_member(get, z.getinfo("a/deflated.txt")) == b"lying " * 500


def test_splits_keep_neighbours_together():
    names = [f"D/train/images/BarnCam-01-{v:02d}_frame_{i}_jpg.rf.x.jpg" for v in range(1, 12) for i in range(3)]
    names += [n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt" for n in names]
    plan = x.plan_zenodo(names)
    held = {p[1].split("_frame")[0] for p in plan if p[0] == "valid"}
    assert held == {"BarnCam-01-05", "BarnCam-01-10"} and len(plan) == 33

    names = [f"visual_data/images/0725/cam_{c}/{1690243200 + 15 * k}_00-00-00.jpg" for c in (1, 2) for k in range(2400)]
    names += [n.replace("/images/", "/labels/combined/").rsplit(".", 1)[0] + ".txt" for n in names]
    plan = x.plan_mmcows(names, every=8)
    assert len(plan) == 600 and plan[0][1].startswith("cam_1_")
    hours = {}
    for split, fname, _, _ in plan:
        hours.setdefault(int(fname.split("_")[2]) // 3600, set()).add(split)
    assert all(len(s) == 1 for s in hours.values())          # an hour is train or valid, never both
    assert {h for h, s in hours.items() if "valid" in s} == {h for h in hours if h % 6 == 5}

    base = "z/detection_and_localisation/"
    names = {base + f"images/{i:06d}.jpg" for i in (1, 2, 3707, 3708)}
    names |= {n.replace("/images/", "/labels-darknet/")[:-4] + ".txt" for n in names}
    plan = x.plan_opencows(names, {"0": {"test": ["000002.jpg"]}})
    assert [(p[0], p[1]) for p in plan] == [("train", "000001.jpg"), ("valid", "000002.jpg"), ("train", "003707.jpg")]


def test_extra_merged_into_rfdetr_coco():
    import rfdetr_det
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "zenodo", "train")
        os.makedirs(src)
        items = []
        for i in range(3):
            Image.new("RGB", (200, 100)).save(os.path.join(src, f"f{i}.jpg"))
            items.append((f"f{i}.jpg", (200, 100), [[0.1, 0.2, 0.5, 0.6]] * (i + 1)))
        assert x.write_coco(src, items) == (3, 6)
        assert x.load_coco(src)[1]["boxes"][0] == [0.1, 0.2, 0.5, 0.6]
        out = os.path.join(d, "ds", "train")
        n, a = rfdetr_det.write_coco(None, [], out, [("zenodo", src, 2)])
        assert n == 2
        coco = json.load(open(os.path.join(out, "_annotations.coco.json")))
        assert all(im["file_name"].startswith("x_zenodo_") for im in coco["images"])
        assert len(coco["annotations"]) == a and {c["image_id"] for c in coco["annotations"]} <= {1, 2}
        assert os.path.exists(os.path.join(out, coco["images"][0]["file_name"]))
        # a small set (the barn's) repeated: its images k times, under names of their own
        n3, a3 = rfdetr_det.write_coco(None, [], os.path.join(d, "ds", "rep"), [("zenodo", src, 0)], {"zenodo": 2})
        assert (n3, a3) == (6, 12)
        names = [im["file_name"] for im in json.load(open(os.path.join(d, "ds", "rep", "_annotations.coco.json")))["images"]]
        assert len(set(names)) == 6


def test_rfdetr_init_from_own_detector():
    import argparse
    import torch
    import rfdetr_det
    with tempfile.TemporaryDirectory() as d:
        best = os.path.join(d, "best")
        os.makedirs(best)
        torch.save({"model": {"class_embed.bias": torch.zeros(2)}, "args": argparse.Namespace(class_names=["cow"])},
                   os.path.join(best, "checkpoint.pth"))
        json.dump({"kind": "rfdetr", "base_model": "rfdetr-large", "checkpoint": "checkpoint.pth", "size": 704},
                  open(os.path.join(best, "det_train_meta.json"), "w"))
        a = argparse.Namespace(init=best, size="large", resolution=None, out=os.path.join(d, "new"))
        slim = torch.load(rfdetr_det.init_checkpoint(a), weights_only=True)   # what rfdetr itself can read
        assert a.resolution == 704 and slim["args"]["class_names"] == ["cow"] and "class_embed.bias" in slim["model"]
        a.size = "small"
        try:
            rfdetr_det.init_checkpoint(a)
            assert False, "another size must be refused"
        except SystemExit:
            pass
