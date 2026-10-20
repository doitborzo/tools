#!/usr/bin/env python3
"""Cow boxes on the barn's own cameras: a detector training set made from their recordings.

The detector was trained on CBVD-5 (side view) and public top-down sets; the barn's cameras, light
and pens are its own. A few hundred frames of them, boxes corrected by hand, teach it the barn.

  1. frames  recordings -> one frame every --every s per camera (frames that barely changed are
             skipped; at most --max a camera, spread over all the recordings), cut and greyed as the
             pipeline gives them to the detector (frame_width, the camera's `exclude` zone grey), the
             detector's boxes on each as a first guess -> DATA/frames/<cam>/*.jpg, DATA/frames.json
             and the labelling page DATA/boxes.html
  2. label   (a person, in a browser) boxes.html: move, resize, delete, draw boxes; a frame marked
             done (Enter) is exported, one not marked is not. "Save labels.json" writes the result.
  3. export  -> herd_det_barn_<date>.tgz: COCO train/ and valid/ (valid: whole hours of a camera, a
             fifth, never next to a training frame of the same minutes) as extra_det/barn - what
             run_pod.sh's detector step (DET_ARCH=rfdetr) takes in beside CBVD-5 and the public sets.

    python herd/herd.py boxes frames --config /srv/herd/barn.toml --videos /srv/herd/rec --out /srv/herd/boxes
    python herd/herd.py boxes page   --data /srv/herd/boxes --labels labels.json   # the page again, names kept
    python herd/herd.py boxes export --data /srv/herd/boxes --labels labels.json
    python herd/herd.py boxes ids    --data /srv/herd/boxes --labels labels.json   # boxes named with a cow id

Recordings are named by camera and local start time (cam1_2026-10-20_06-00-00.mp4, cam1/20261020T060000.mkv
...): see common.recordings. One file: --video cam1=file.mp4 --start 2026-10-20T06:00:00.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import itertools
import json
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402,F401  (puts cowbench on the path)
from common import blank_excluded, load_config, recordings  # noqa: E402


def _tz(cfg):
    return dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def plan_times(recs, every, max_n):
    """[(path, start ts, fps, n frames)] of one camera -> [(path, frame index, ts)]: a frame every `every`
    s, thinned evenly to at most max_n over all its recordings."""
    out = []
    for path, ts, fps, n in recs:
        step = max(1, int(round(every * fps)))
        out += [(path, i, ts + i / fps) for i in range(step // 2, n, step)]
    if max_n and len(out) > max_n:
        keep = np.unique(np.round(np.linspace(0, len(out) - 1, max_n)).astype(int))
        out = [out[i] for i in keep]
    return out


def changed(a, b, min_change):
    """Two small grey thumbnails differ by more than min_change (0-255, mean) - a frame worth labelling."""
    return b is None or float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) >= min_change


def cut_frame(cap, i, c, width):
    """Frame i of an open recording as the detector sees it in the barn (frame_width, the exclude zone grey)."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    if not ok:
        return None
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if w > width:                                        # as Camera.push
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    if c.get("blank_excluded", True):                   # as Camera.blank
        img = blank_excluded(img, c.get("exclude"))
    return img


def recut(index, out, by_cam, cams, width, again):
    """The frames already cut of these cameras, cut again from their recordings at the same moments with
    the camera's zone as the config has it now (an exclude zone changed); the detector looks again, and
    the page asks to check each of them again (a label made before keeps its boxes, minus those now in
    the grey)."""
    import cv2
    from PIL import Image
    n, missing = collections.Counter(), collections.Counter()
    stamp = int(dt.datetime.now().timestamp())
    for f in index["frames"]:
        if f["cam"] not in again:
            continue
        src = [r for r in by_cam.get(f["cam"], []) if r[1] - 1 <= f["ts"] <= r[1] + r[3] / r[2] + 1]
        if not src:
            missing[f["cam"]] += 1
            continue
        path, start, fps, _ = src[0]
        cap = cv2.VideoCapture(path)
        img = cut_frame(cap, f.get("frame", int(round((f["ts"] - start) * fps))), cams[f["cam"]], width)
        cap.release()
        if img is None:
            missing[f["cam"]] += 1
            continue
        Image.fromarray(img).save(os.path.join(out, f["file"]), quality=92)
        f.update(w=int(img.shape[1]), h=int(img.shape[0]), cut=stamp, exclude=cams[f["cam"]].get("exclude") or [])
        f.pop("boxes", None)
        n[f["cam"]] += 1
    for cam in sorted(again):
        print(f"[boxes] {cam}: {n[cam]} frames cut again with its zone as now"
              + (f"; {missing[cam]} not found in the recordings (left as they were)" if missing[cam] else ""), flush=True)


def cmd_frames(args):
    import cv2
    from PIL import Image
    cfg = load_config(args.config)
    tz = _tz(cfg)
    cams = {c["id"]: c for c in cfg["cameras"]}
    if args.video:
        if not args.start:
            sys.exit("--video needs --start (when the recording began, local time)")
        t0 = dt.datetime.fromisoformat(args.start)
        t0 = (t0 if t0.tzinfo else t0.replace(tzinfo=tz)).timestamp()
        recs = []
        for v in args.video:
            cam, _, path = v.partition("=")
            if cam not in cams or not os.path.exists(path):
                sys.exit(f"--video {v}: want <camera id>=<file>, a camera of {args.config} ({', '.join(cams)})")
            recs.append((cam, path, t0))
    else:
        recs, bad = recordings(args.videos, list(cams), cfg["farm"]["timezone_offset_hours"])
        if bad:
            print(f"[boxes] {len(bad)} files without a camera id or a time in their name - skipped, e.g. {bad[0]}")
        if not recs:
            sys.exit(f"no recordings under {args.videos} named <camera>..<YYYY-MM-DD_HH-MM-SS>")
    if args.cameras:
        unknown = sorted(set(args.cameras) - set(cams))
        if unknown:
            sys.exit(f"--cameras {' '.join(unknown)}: not in {args.config} ({', '.join(cams)})")
        recs = [r for r in recs if r[0] in args.cameras]
    out = os.path.abspath(args.out.rstrip("\\"))
    index_path = os.path.join(out, "frames.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else {"frames": []}
    have = {f["file"] for f in index["frames"]}
    width = cfg["sampling"]["frame_width"]
    by_cam = collections.defaultdict(list)
    for cam, path, ts in recs:
        info = video_info(path)
        if info is None:
            print(f"[boxes] cannot open {path} - skipped")
            continue
        by_cam[cam].append((path, ts) + info)
    if args.cameras and args.again:
        recut(index, out, by_cam, cams, width, set(args.cameras))
    for cam, rs in sorted(by_cam.items()):
        c = cams[cam]
        plan = plan_times(rs, args.every, args.max)
        folder = os.path.join(out, "frames", cam)
        os.makedirs(folder, exist_ok=True)
        kept, last, cap, cap_path = 0, None, None, None
        for path, i, ts in plan:
            name = f"{cam}_{dt.datetime.fromtimestamp(ts, tz):%Y%m%d_%H%M%S}.jpg"
            rel = f"frames/{cam}/{name}"
            if rel in have:
                kept += 1
                continue
            if path != cap_path:
                if cap is not None:
                    cap.release()
                cap, cap_path = cv2.VideoCapture(path), path
            img = cut_frame(cap, i, c, width)
            if img is None:
                continue
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
            if not changed(small, last, args.min_change):
                continue
            last = small
            Image.fromarray(img).save(os.path.join(out, rel), quality=92)
            index["frames"].append({"file": rel, "cam": cam, "ts": ts, "w": img.shape[1], "h": img.shape[0],
                                    "time": dt.datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M:%S"),
                                    "frame": i, "exclude": c.get("exclude") or []})
            have.add(rel)
            kept += 1
        if cap is not None:
            cap.release()
        print(f"[boxes] {cam}: {kept} frames from {len(rs)} recordings ({len(plan)} looked at)", flush=True)
    index["frames"].sort(key=lambda f: (f["cam"], f["ts"]))
    if args.detector != "none":
        detect(index, out, args.detector or cfg["detector"]["weights"], cfg)
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    write_page(out, index, {})
    print(f"[boxes] {len(index['frames'])} frames -> {os.path.join(out, 'boxes.html')} (open it in a browser)")


def detect(index, out, weights, cfg):
    """The detector's boxes on every frame not yet looked at: those above its threshold drawn, those down
    to half of it offered (dashed) - a cow it nearly missed is one click."""
    from PIL import Image
    import detector as det_mod
    meta = json.load(open(os.path.join(weights, "det_train_meta.json"), encoding="utf-8"))
    thr = cfg["detector"]["threshold"] or meta["threshold"]
    todo = [f for f in index["frames"] if "boxes" not in f]
    index.update(threshold=thr, detector=os.path.abspath(weights))
    if not todo:
        return
    live = det_mod.Live(weights, threshold=thr / 2, tiles=cfg["detector"].get("tiles"))
    print(f"[boxes] the detector ({live.name}, threshold {thr}) on {len(todo)} frames ({live.device})", flush=True)
    for k in range(0, len(todo), 4):
        chunk = todo[k:k + 4]
        imgs = [Image.open(os.path.join(out, f["file"])).convert("RGB") for f in chunk]
        for f, found in zip(chunk, live.many(imgs)):
            f["boxes"] = [[round(v, 4) for v in d["bbox"]] + [round(float(d["det_score"]), 3)] for d in found]
        print(f"\r  {min(k + 4, len(todo))}/{len(todo)}", end="", flush=True)
    print(flush=True)


def in_grey(box, exclude):
    """A box whose centre lies in one of the exclude polygons (greyed for the detector)."""
    from common import _polys, in_polygon
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(in_polygon(cx, cy, p) for p in _polys(exclude))


def read_labels(path):
    """labels.json (the page's Save) -> {frame file: {"status": "done" | "skip", "boxes": [[x1, y1, x2, y2] 0-1]}}."""
    if not path:
        return {}
    d = json.load(open(path, encoding="utf-8"))
    return d.get("frames", d)


def write_page(out, index, labels):
    data = {"key": out, "threshold": index.get("threshold", 0.5), "frames": index["frames"], "labels": labels}
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    with open(os.path.join(out, "boxes.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


def cmd_page(args):
    out = os.path.abspath(args.data)
    index = json.load(open(os.path.join(out, "frames.json"), encoding="utf-8"))
    labels = read_labels(args.labels)
    write_page(out, index, labels)
    done = sum(1 for v in labels.values() if v.get("status") == "done")
    print(f"[boxes] {len(index['frames'])} frames ({done} done in {args.labels or 'no labels file'}) -> "
          f"{os.path.join(out, 'boxes.html')}")


def split_blocks(frames, val_share, block_s, seed):
    """Done frames -> (train, valid): whole blocks (a camera's hour) go to valid, so no valid frame has a
    training frame of the same minutes beside it; at least one block each when there are two."""
    blocks = collections.defaultdict(list)
    for f in frames:
        blocks[(f["cam"], int(f["ts"] // block_s))].append(f)
    keys = sorted(blocks)
    random.Random(seed).shuffle(keys)
    val, n_val = set(), 0
    for k in keys:
        if n_val >= val_share * len(frames) or len(val) + 1 >= len(keys):
            break
        val.add(k)
        n_val += len(blocks[k])
    tr = [f for k in keys if k not in val for f in blocks[k]]
    va = [f for k in keys if k in val for f in blocks[k]]
    return tr, va


def write_coco(folder, frames, src):
    import shutil
    os.makedirs(folder, exist_ok=True)
    images, anns = [], []
    for i, f in enumerate(frames, 1):
        name = os.path.basename(f["file"])
        shutil.copy(os.path.join(src, f["file"]), os.path.join(folder, name))
        w, h = f["w"], f["h"]
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for b in f["labelled"]:
            x, y, bw, bh = b[0] * w, b[1] * h, (b[2] - b[0]) * w, (b[3] - b[1]) * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1,
                         "bbox": [round(x, 1), round(y, 1), round(bw, 1), round(bh, 1)], "area": bw * bh, "iscrowd": 0})
    with open(os.path.join(folder, "_annotations.coco.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "cow", "supercategory": "none"}]}, fh)
    return len(anns)


def cmd_export(args):
    import shutil
    import tarfile
    import tempfile
    src = os.path.abspath(args.data)
    index = json.load(open(os.path.join(src, "frames.json"), encoding="utf-8"))
    labels = read_labels(args.labels)
    done, stale, grey = [], 0, 0
    for f in index["frames"]:
        lab = labels.get(f["file"])
        if lab and lab.get("status") == "done":
            if f.get("cut") and lab.get("cut") != f["cut"]:    # cut again (its zone changed) since it was labelled
                stale += 1
                continue
            boxes = [b[:4] for b in lab.get("boxes", []) if b[2] - b[0] > 0.002 and b[3] - b[1] > 0.002]
            keep = [b for b in boxes if not in_grey(b, f.get("exclude"))]
            grey += len(boxes) - len(keep)
            done.append(dict(f, labelled=keep))
    if stale:
        print(f"[boxes] {stale} frames were cut again after they were labelled - left out until checked on the page "
              f"(boxes page, then Enter on each; Save labels.json)")
    if grey:
        print(f"[boxes] {grey} boxes in a camera's excluded (grey) zone left out")
    if len(done) < 2:
        sys.exit(f"{len(done)} frames marked done in {args.labels}: label more (Enter on the page marks a frame done)")
    tr, va = split_blocks(done, args.val_share, args.block_min * 60, args.seed)
    if not va:
        sys.exit("all done frames are in one hour of one camera: label frames of other hours too (they are "
                 "the held-out check)")
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "extra_det", "barn")
    n_tr = write_coco(os.path.join(root, "train"), tr, src)
    n_va = write_coco(os.path.join(root, "valid"), va, src)
    shutil.copy(args.labels, os.path.join(root, "labels.json"))
    out = args.out or f"herd_det_barn_{dt.datetime.now():%Y%m%d_%H%M}.tgz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(os.path.join(tmp, "extra_det"), arcname="extra_det")
    shutil.rmtree(tmp)
    empty = sum(1 for f in done if not f["labelled"])
    cams = collections.Counter(f["cam"] for f in done)
    print(f"[boxes] {len(done)} frames ({', '.join(f'{c} {n}' for c, n in sorted(cams.items()))}; {empty} with no "
          f"cow): train {len(tr)} frames / {n_tr} cows, valid {len(va)} frames / {n_va} cows -> {out} "
          f"({os.path.getsize(out) / 2 ** 20:.0f} MB)")


def box_iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def in_between(items, frames, every, min_iou=0.5, max_gap=30.0):
    """More crops of a named cow from the video between two cut frames: where she has a box on two
    neighbouring frames of a camera (at most max_gap s apart) that overlap (IoU >= min_iou: she stayed about
    where she was), a box every `every` s between them, moved linearly -> [(cow, cam, ts, box)]."""
    order = {}
    for cam in {f["cam"] for f in frames}:
        fs = sorted((f for f in frames if f["cam"] == cam), key=lambda f: f["ts"])
        order.update({f["file"]: (cam, k) for k, f in enumerate(fs)})
    by = collections.defaultdict(dict)
    for cow, f, box in items:
        cam, k = order[f["file"]]
        by[(cow, cam)][k] = (f["ts"], box)
    out = []
    for (cow, cam), ks in by.items():
        for k in sorted(ks):
            if k + 1 not in ks:
                continue
            (t1, b1), (t2, b2) = ks[k], ks[k + 1]
            if t2 - t1 > max_gap or box_iou(b1, b2) < min_iou:
                continue
            t = t1 + every
            while t < t2 - every / 2:
                a = (t - t1) / (t2 - t1)
                out.append((cow, cam, t, [b1[i] + (b2[i] - b1[i]) * a for i in range(4)]))
                t += every
    return out


def video_frames(recs, wanted, width):
    """{(cam, ts): rgb frame} for the wanted moments, read through each recording once (no seeking)."""
    import cv2
    got = {}
    for cam, items in itertools.groupby(sorted(wanted), key=lambda x: x[0]):
        tss = [ts for _, ts in items]
        for path, start, fps, n in recs.get(cam, []):
            need = sorted({int(round((ts - start) * fps)): ts for ts in tss if start <= ts <= start + n / fps}.items())
            if not need:
                continue
            cap = cv2.VideoCapture(path)
            i, k = 0, 0
            while k < len(need):
                if not cap.grab():
                    break
                if i == need[k][0]:
                    ok, f = cap.retrieve()
                    if ok:
                        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        h, w = img.shape[:2]
                        if w > width:
                            img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
                        got[(cam, need[k][1])] = img
                    k += 1
                    while k < len(need) and need[k][0] <= i:
                        k += 1
                i += 1
            cap.release()
    return got


def cmd_ids(args):
    """Boxes named with a cow's id -> the identity model's set "barn_ids" (reid.py): each such box cut as the
    barn cuts a cow (square, with the model's margin), its cow's id, camera and time; with --fill, more of
    her from the video between two cut frames where she stayed about in place; a sixth of the cows held out
    whole (never trained on: they score the model and set its thresholds). reid_pod.sh takes
    herd_barn_ids_<date>.tgz in and runs DINOv2 on the crops there."""
    import random
    import shutil
    import tarfile
    import tempfile
    from PIL import Image
    from common import crop, load_config, recordings, square_crop_box
    src = os.path.abspath(args.data)
    index = json.load(open(os.path.join(src, "frames.json"), encoding="utf-8"))
    labels = read_labels(args.labels)
    items, stale = [], 0
    for f in index["frames"]:
        lab = labels.get(f["file"])
        if not lab or (args.done_only and lab.get("status") != "done") or lab.get("status") == "skip":
            continue
        if f.get("cut") and lab.get("cut") != f["cut"]:
            stale += 1
            continue
        for b in lab.get("boxes", []):
            if len(b) > 4 and str(b[4]).strip() and not in_grey(b, f.get("exclude")):
                items.append((str(b[4]).strip(), f, b[:4]))
    extra = []
    if args.fill:
        if not (args.config and args.videos):
            sys.exit("--fill needs --config and --videos (the recordings the frames were cut from)")
        cfg = load_config(args.config)
        recs, _ = recordings(args.videos, [c["id"] for c in cfg["cameras"]], cfg["farm"]["timezone_offset_hours"])
        by_cam = collections.defaultdict(list)
        for cam, path, ts in recs:
            info = video_info(path)
            if info:
                by_cam[cam].append((path, ts) + info)
        extra = in_between(items, index["frames"], args.fill, args.fill_iou, args.fill_gap)
        print(f"[ids] {len(extra)} more boxes between the cut frames (every {args.fill:g} s, where the cow stayed "
              f"in place); reading them from the recordings", flush=True)
        pics = video_frames(by_cam, {(cam, ts) for _, cam, ts, _ in extra}, cfg["sampling"]["frame_width"])
        tz = dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))
        extra = [(cow, {"cam": cam, "ts": ts, "pic": pics[(cam, ts)],
                        "time": dt.datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M:%S")}, box)
                 for cow, cam, ts, box in extra if (cam, ts) in pics]
    count = collections.Counter(cow for cow, _, _ in items)
    few = sorted(c for c, n in count.items() if n < args.min_crops)
    items = [it for it in items + extra if count[it[0]] >= args.min_crops]
    cows = sorted({cow for cow, _, _ in items})
    if len(cows) < 2:
        sys.exit(f"{len(cows)} cow(s) with {args.min_crops}+ named boxes: name more boxes "
                 f"(select a box, I, her tag, Enter; Save labels.json)")
    k = max(1, round(len(cows) * args.val_share)) if args.val_share > 0 else 0
    val = set(random.Random(args.seed).sample(cows, min(k, len(cows) - 1)))
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, "reid", "raw", "barn_ids")
    rows, images = [], {}
    for cow, f, box in sorted(items, key=lambda it: (it[0], it[1]["cam"], it[1]["ts"])):
        if "pic" in f:
            img = f["pic"]
        else:
            if f["file"] not in images:
                images = {f["file"]: np.asarray(Image.open(os.path.join(src, f["file"])).convert("RGB"))}
            img = images[f["file"]]
        x1, _, x2, _ = square_crop_box(box, img.shape[1], img.shape[0], args.margin)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in cow)
        name = f"{safe}/{f['cam']}_{int(f['ts'] * 10)}_{len(rows)}.jpg"
        os.makedirs(os.path.join(raw, safe), exist_ok=True)
        # as the barn cuts a cow: square, the margin round her, grey past the frame's edge
        Image.fromarray(crop(img, box, max(32, min(448, x2 - x1)), args.margin)).save(os.path.join(raw, name), quality=92)
        rows.append({"file": name, "cow": cow, "day": f["time"][:10].replace("-", ""), "cam": f["cam"],
                     "order": int(f["ts"] * 10), "ready": True, "split": "val" if cow in val else "train",
                     "by": "fill" if "pic" in f else "box"})
    with open(os.path.join(raw, "index.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(raw, "complete.json"), "w", encoding="utf-8") as fh:
        json.dump({"crops": len(rows), "cows": len(cows), "held_out": sorted(val), "margin": args.margin}, fh)
    out = args.out or f"herd_barn_ids_{dt.datetime.now():%Y%m%d_%H%M}.tgz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(os.path.join(tmp, "reid"), arcname="reid")
    shutil.rmtree(tmp)
    cams = collections.Counter(r["cam"] for r in rows)
    multi = sum(1 for c in cows if len({r["cam"] for r in rows if r["cow"] == c}) > 1)
    print(f"[ids] {len(rows)} crops ({sum(1 for r in rows if r['by'] == 'fill')} from between the frames) of "
          f"{len(cows)} cows ({', '.join(f'{c} {n}' for c, n in sorted(cams.items()))}); {multi} cows seen by 2+ "
          f"cameras; held out: {', '.join(sorted(val))} -> {out}")
    if few:
        print(f"[ids] fewer than {args.min_crops} named boxes, left out: {', '.join(few)}")
    if stale:
        print(f"[ids] {stale} frames cut again after they were labelled - left out until checked on the page")


def cmd_gallery(args):
    """Every cow id of every camera as a row of her crops, side by side - to see that one id is one cow on
    every camera (and fix it: each row gets the cow it really is; Save labels.json writes the new ids)."""
    from PIL import Image
    src = os.path.abspath(args.data)
    index = json.load(open(os.path.join(src, "frames.json"), encoding="utf-8"))
    labels = read_labels(args.labels)
    by = collections.defaultdict(list)
    for f in index["frames"]:
        for b in (labels.get(f["file"]) or {}).get("boxes", []):
            if len(b) > 4 and str(b[4]).strip():
                by[(f["cam"], str(b[4]).strip())].append((f, b))
    folder = os.path.join(src, "gallery")
    shutil.rmtree(folder, ignore_errors=True)
    rows = []
    for (cam, cow), items in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        items.sort(key=lambda it: it[0]["ts"])
        pick = sorted({int(round(i)) for i in np.linspace(0, len(items) - 1, min(args.thumbs, len(items)))})
        thumbs = []
        for k in pick:
            f, b = items[k]
            with Image.open(os.path.join(src, f["file"])) as im:
                w, h = im.size
                pad = 0.08
                c = im.crop((int(max(0, (b[0] - pad * (b[2] - b[0])) * w)), int(max(0, (b[1] - pad * (b[3] - b[1])) * h)),
                             int(min(w, (b[2] + pad * (b[2] - b[0])) * w)), int(min(h, (b[3] + pad * (b[3] - b[1])) * h))))
                c.thumbnail((160, 160))
                safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in cow)
                rel = f"gallery/{cam}/{safe}_{k}.jpg"
                os.makedirs(os.path.join(src, "gallery", cam), exist_ok=True)
                c.convert("RGB").save(os.path.join(src, rel), quality=85)
                thumbs.append({"src": rel, "time": f["time"][11:]})
        rows.append({"cam": cam, "id": cow, "n": len(items), "auto": sum(1 for _, b in items if len(b) > 5 and b[5]),
                     "thumbs": thumbs})
    data = {"key": src, "rows": rows, "labels": labels}
    out = os.path.join(src, "ids.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(GALLERY.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    cams = collections.defaultdict(set)
    for r in rows:
        cams[r["id"]].add(r["cam"])
    print(f"[ids] {len(rows)} rows (an id on a camera) of {len(cams)} ids; {sum(1 for v in cams.values() if len(v) > 1)} "
          f"ids on 2+ cameras -> {out} (open it in a browser)")


GALLERY = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cow ids by camera</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1da;--accent:#2f6f4f;--warn:#a33}
@media (prefers-color-scheme: dark){:root{--bg:#171816;--card:#20221f;--fg:#ecebe6;--mute:#9a9a93;--line:#34362f;--accent:#7fc29c;--warn:#e88}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:17px;margin:0 0 6px}.help{color:var(--mute);font-size:13px;margin:2px 0}.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button,select,input{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button.main{background:var(--accent);color:var(--bg);border-color:var(--accent)}main{padding:8px 16px 40px}
.group{border:1px solid var(--line);border-radius:8px;background:var(--card);margin:8px 0;padding:6px 8px}
.group h2{font-size:14px;margin:2px 0 6px}.one{color:var(--warn)}
.row{display:grid;grid-template-columns:200px 1fr;gap:8px;padding:4px 0;border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}.thumbs{display:flex;flex-wrap:wrap;gap:3px}.thumbs img{height:96px;border-radius:4px}
.mute{color:var(--mute)}.row input{width:100%;margin-top:4px}
.match{display:grid;grid-template-columns:1fr 1fr;gap:12px}.match .ref{position:sticky;top:120px;max-height:80vh;overflow:auto}
.cand{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:4px 6px;margin:4px 0;cursor:pointer}
.cand:hover{border-color:var(--accent)}.cand img{height:64px;border-radius:3px;margin-right:2px}
.row.cur{outline:2px solid var(--accent);border-radius:6px}.row.donep{opacity:.55}.warn{color:var(--warn)}
@media (max-width:640px){header{position:static}.row{grid-template-columns:1fr}.thumbs img{height:72px}.match{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Cow ids by camera: one id, one cow?</h1>
<p class="help">Each group is one id; each row in it, that id on one camera, with her crops spread over the recording. If the rows
of a group are not the same cow, or a cow has a different id on another camera, type in each row the cow it really is (her ear
tag): rows with the same name become one cow on every camera. <b>x</b> removes the id. <b>Save labels.json</b> writes the
labels with the new ids - then <code>boxes ids</code> again (and on the box page: Load labels.json).</p>
<div class="bar"><select id="cam"><option value="">all cameras</option></select>
<label><input type="checkbox" id="single"> only ids on one camera</label><input id="find" placeholder="find an id" size="10">
<button class="main" id="save">Save labels.json</button><span class="mute" id="stat"></span></div>
<div class="bar"><b>match cameras:</b> name the cows of <select id="ma"></select> by those of <select id="mb"></select>
<button id="mgo">start</button><button id="mstop">back to the list</button>
<span class="mute">click a cow on the right: the selected row on the left becomes her; "new cow" keeps it apart</span></div>
<div class="bar warn" id="dups"></div></header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const KEY = "herd-ids-rename:" + DATA.key;
let rename = {};
try { rename = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const $ = id => document.getElementById(id), k = r => r.cam + "\u0001" + r.id;
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
[...new Set(DATA.rows.map(r => r.cam))].sort().forEach(c => $("cam").insertAdjacentHTML("beforeend", `<option>${esc(c)}</option>`));
const name = r => (rename[k(r)] || r.id).trim();
function render() {
  const c = $("cam").value, single = $("single").checked, f = $("find").value.trim().toLowerCase();
  const groups = {};
  DATA.rows.forEach(r => (groups[name(r)] = groups[name(r)] || []).push(r));
  const list = $("list"); list.innerHTML = "";
  Object.keys(groups).sort((a, b) => a.localeCompare(b, undefined, {numeric: true})).forEach(g => {
    const rs = groups[g], cams = new Set(rs.map(r => r.cam));
    if (c && !cams.has(c)) return;
    if (single && cams.size > 1) return;
    if (f && !g.toLowerCase().includes(f)) return;
    const div = document.createElement("div"); div.className = "group";
    div.innerHTML = `<h2>${esc(g)} <span class="${cams.size > 1 ? "mute" : "one"}">- ${cams.size} camera${cams.size > 1 ? "s" : ""}</span></h2>`;
    rs.sort((a, b) => a.cam.localeCompare(b.cam)).forEach(r => {
      const row = document.createElement("div"); row.className = "row";
      row.innerHTML = `<div><b>${esc(r.cam)}</b> <span class="mute">id ${esc(r.id)}, ${r.n} boxes${r.auto ? ` (${r.auto} filled in)` : ""}</span>
        <input placeholder="the cow it is" value="${esc(rename[k(r)] || "")}"></div>
        <div class="thumbs">${r.thumbs.map(t => `<img loading="lazy" src="${esc(t.src)}" title="${esc(t.time)}">`).join("")}</div>`;
      row.querySelector("input").addEventListener("change", e => { setName(r, e.target.value); render(); });
      div.appendChild(row);
    });
    list.appendChild(div);
  });
  stat(); dups();
}
let mode = null, curRow = 0;
[...new Set(DATA.rows.map(r => r.cam))].sort().forEach((c, i) => {
  $("ma").insertAdjacentHTML("beforeend", `<option ${i === 1 ? "selected" : ""}>${esc(c)}</option>`);
  $("mb").insertAdjacentHTML("beforeend", `<option>${esc(c)}</option>`);
});
function setName(r, v) {
  v = (v || "").trim();
  if (v && v !== r.id) rename[k(r)] = v; else delete rename[k(r)];
  try { localStorage.setItem(KEY, JSON.stringify(rename)); } catch (e) {}
}
function dups() {                    // one camera, two rows named the same cow: two cows on one frame under one id
  const out = [];
  [...new Set(DATA.rows.map(r => r.cam))].forEach(c => {
    const n = {};
    DATA.rows.filter(r => r.cam === c).forEach(r => (n[name(r)] = n[name(r)] || []).push(r.id));
    Object.entries(n).filter(([, v]) => v.length > 1).forEach(([g, v]) => out.push(`${c}: ${v.join(" + ")} are all "${g}"`));
  });
  $("dups").textContent = out.length ? "One cow twice on one camera - check: " + out.join("; ") : "";
}
function renderMatch() {
  const A = mode.a, B = mode.b, rowsA = DATA.rows.filter(r => r.cam === A), list = $("list");
  const refs = {};
  DATA.rows.filter(r => r.cam === B).forEach(r => (refs[name(r)] = refs[name(r)] || []).push(r));
  list.innerHTML = `<div class="match"><div id="left"></div><div class="ref" id="right"></div></div>`;
  rowsA.forEach((r, i) => {
    const row = document.createElement("div");
    const linked = name(r) in refs;
    row.className = "row" + (i === curRow ? " cur" : "") + (linked ? " donep" : "");
    row.innerHTML = `<div><b>${esc(r.cam)} ${esc(r.id)}</b><br><span class="mute">now: ${esc(name(r))}${linked ? " (= " + esc(B) + ")" : ""}</span></div>
      <div class="thumbs">${r.thumbs.map(t => `<img loading="lazy" src="${esc(t.src)}" title="${esc(t.time)}">`).join("")}</div>`;
    row.onclick = () => { curRow = i; renderMatch(); };
    $("left").appendChild(row);
  });
  Object.keys(refs).sort((a, b) => a.localeCompare(b, undefined, {numeric: true})).forEach(g => {
    const c = document.createElement("div"); c.className = "cand";
    const thumbs = refs[g].flatMap(r => r.thumbs).slice(0, 5);
    const taken = rowsA.some((r, i) => i !== curRow && name(r) === g);
    c.innerHTML = `<b>${esc(g)}</b>${taken ? ' <span class="warn">(given to another row)</span>' : ""}<br>${thumbs.map(t => `<img loading="lazy" src="${esc(t.src)}">`).join("")}`;
    c.onclick = () => { setName(rowsA[curRow], g); curRow = Math.min(rowsA.length - 1, curRow + 1); renderMatch(); stat(); };
    $("right").appendChild(c);
  });
  const nc = document.createElement("div"); nc.className = "cand";
  nc.innerHTML = `<b>new cow</b> <span class="mute">- not on ${esc(B)}: keep her own id</span>`;
  nc.onclick = () => { setName(rowsA[curRow], ""); curRow = Math.min(rowsA.length - 1, curRow + 1); renderMatch(); stat(); };
  $("right").prepend(nc);
  const cur = $("left").querySelector(".cur"); if (cur) cur.scrollIntoView({block: "nearest"});
  dups();
}
function stat() {
  const ids = new Set(DATA.rows.map(name)), multi = [...ids].filter(g => new Set(DATA.rows.filter(r => name(r) === g).map(r => r.cam)).size > 1);
  $("stat").textContent = `${ids.size} cows, ${multi.length} of them on 2+ cameras; ${Object.keys(rename).length} rows renamed`;
}
$("mgo").onclick = () => { if ($("ma").value === $("mb").value) { alert("two different cameras"); return; } mode = {a: $("ma").value, b: $("mb").value}; curRow = 0; renderMatch(); };
$("mstop").onclick = () => { mode = null; render(); };
document.addEventListener("keydown", e => {
  if (!mode || e.target.tagName === "INPUT") return;
  const n = DATA.rows.filter(r => r.cam === mode.a).length;
  if (e.key === "ArrowDown" || e.key === "j") { curRow = Math.min(n - 1, curRow + 1); renderMatch(); e.preventDefault(); }
  if (e.key === "ArrowUp" || e.key === "k") { curRow = Math.max(0, curRow - 1); renderMatch(); e.preventDefault(); }
});
["cam", "single"].forEach(id => $(id).onchange = render); $("find").oninput = render;
$("save").onclick = () => {
  const cam = {};
  DATA.rows.forEach(r => cam[k(r)] = r);
  const out = {};
  // frames know their camera from the file name (frames/<cam>/...)
  Object.entries(DATA.labels).forEach(([file, lab]) => {
    const c = file.split("/")[1];
    out[file] = Object.assign({}, lab, {boxes: (lab.boxes || []).map(b => {
      if (b.length < 5 || !b[4]) return b;
      const v = rename[c + "\u0001" + String(b[4]).trim()];
      if (!v) return b;
      return v.toLowerCase() === "x" ? b.slice(0, 4) : b.slice(0, 4).concat([v]).concat(b.slice(5));
    })});
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({frames: out})], {type: "application/json"}));
  a.download = "labels.json"; a.click();
};
render();
</script></body></html>
"""


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Barn cow boxes</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1da;--accent:#2f6f4f;--done:#2f6f4f;--skip:#a33}
@media (prefers-color-scheme: dark){:root{--bg:#171816;--card:#20221f;--fg:#ecebe6;--mute:#9a9a93;--line:#34362f;--accent:#7fc29c;--done:#7fc29c;--skip:#e88}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line)}h1{font-size:17px;margin:0 0 6px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0}
button,select{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.main{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.mute{color:var(--mute)}.help{color:var(--mute);font-size:13px;margin:2px 0}
main{padding:8px 16px 30px}#wrap{position:relative;overflow:auto;max-height:82vh}canvas{display:block;max-width:100%;touch-action:none;cursor:crosshair;border-radius:6px;background:#000}
#status.done{color:var(--done);font-weight:600}#status.skip,#status.stale{color:var(--skip);font-weight:600}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
</style></head><body>
<header>
<h1>Barn cow boxes</h1>
<p class="help"><b>A cow the detector missed: drag over her</b> - from one corner to the other, anywhere on the picture (also
over other cows' boxes). Draw round the part of her you can see. A click selects a box: then drag it to move it, its corners
to resize; <kbd>Del</kbd> removes it. Dashed boxes are cows the detector was unsure of: click to keep, <kbd>A</kbd> keeps all.
<kbd>+</kbd> <kbd>&minus;</kbd> zoom (far cows). Grey is the camera's excluded zone: no boxes there. <kbd>Enter</kbd> marks the
frame done and opens the next; only frames marked done are exported. <kbd>S</kbd> skips a frame (spoilt, or unsure);
<kbd>&larr;</kbd> <kbd>&rarr;</kbd> move without marking; <kbd>Ctrl+Z</kbd> undoes. <b>Which cow</b> (for the identity
model): select a box, <kbd>I</kbd>, her ear tag, <kbd>Enter</kbd> - the next box without an id is selected; <kbd>P</kbd> takes
the ids of this camera's frame before for boxes in the same place; <kbd>F</kbd> carries every id typed by hand through all of
this camera's frames, forwards and backwards, as long as the cow's box clearly continues (blue ids: filled in - check them;
a typed id is never overwritten, and fixing a blue one makes it typed).</p>
<div class="bar">
<button id="prev">&larr;</button><span id="pos"></span><button id="next">&rarr;</button>
<select id="cam"></select><label><input type="checkbox" id="todo"> only not done</label>
<button id="zout" title="zoom out (-)">&minus;</button><span id="zoomv">1x</span><button id="zin" title="zoom in (+): far cows">+</button>
<button class="main" id="done">Done &#10003;</button><button id="skip">Skip</button><button id="reset">Detector's boxes</button>
<button class="main" id="save">Save labels.json</button><button id="loadbtn">Load labels.json</button><input type="file" id="load" accept=".json" hidden>
</div>
<div class="bar"><label>cow id <input id="cid" list="ids" size="12" placeholder="ear tag" disabled></label>
<button id="lastid" title="the id typed last">= last</button><button id="previd" title="ids from this camera's frame before (P)">ids from previous (P)</button>
<button id="fillid" title="the ids typed by hand carried through all of this camera's frames, by where the cows are (F)">fill ids through the recording (F)</button>
<label title="fill only the frames you have opened and edited"><input type="checkbox" id="onlyedited"> only frames I edited</label>
<span class="mute">select a box, <kbd>I</kbd> - type her tag - <kbd>Enter</kbd> (the next box without an id is selected)</span></div>
<datalist id="ids"></datalist>
<div class="bar"><span id="info"></span><span id="status"></span><span class="mute" id="stat"></span></div>
</header>
<main><div id="wrap"><canvas id="cv"></canvas></div></main>
<script>
const DATA = __DATA__;
const KEY = "herd-boxes:" + DATA.key;
let labels = Object.assign({}, DATA.labels);
try { Object.assign(labels, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let order = [], at = 0, img = new Image(), boxes = [], cands = [], sel = -1, undo = [], drag = null, scale = 1;
function store() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
const iou = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i || 1); };
function detBoxes(f) {
  const det = f.boxes || [];
  return [det.filter(b => b[4] >= DATA.threshold).map(b => b.slice(0, 4)), det.filter(b => b[4] < DATA.threshold)];
}
const cams = [...new Set(DATA.frames.map(f => f.cam))];
$("cam").innerHTML = '<option value="">all cameras</option>' + cams.map(c => `<option>${c}</option>`).join("");
function filter(keep) {
  const c = $("cam").value, todo = $("todo").checked;
  order = DATA.frames.map((f, i) => i).filter(i => (!c || DATA.frames[i].cam === c) &&
    (!todo || !labels[DATA.frames[i].file] || !labels[DATA.frames[i].file].status || stale(DATA.frames[i]) || i === keep));
  at = Math.max(0, order.indexOf(keep));
}
function frame() { return DATA.frames[order[at]]; }
function inPoly(x, y, p) { let ins = false; for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
  const [x1, y1] = p[i], [x2, y2] = p[j]; if ((y1 > y) !== (y2 > y) && x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1) ins = !ins; } return ins; }
function polys(m) { return !m || !m.length ? [] : (Array.isArray(m[0][0]) ? m : [m]); }
const grey = (f, b) => polys(f.exclude).some(p => inPoly((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, p));
// a label made before the frame was cut again (its camera's zone changed) is to be checked again
const stale = f => f && f.cut && labels[f.file] && labels[f.file].cut !== f.cut;
const isDone = f => labels[f.file] && labels[f.file].status === "done" && !stale(f);
function open() {
  const f = frame();
  if (!f) { ctx.clearRect(0, 0, cv.width, cv.height); $("pos").textContent = "0 / 0"; $("info").textContent = "no frames here"; return; }
  const lab = labels[f.file], [det, low] = detBoxes(f);
  boxes = lab ? lab.boxes.map(b => b.slice()).filter(b => !grey(f, b)) : det;
  cands = low.filter(c => !boxes.some(b => iou(b, c) > 0.5));
  sel = -1; undo = [];
  img = new Image();
  img.onload = () => { fit(); draw(); };
  img.src = f.file;
  $("pos").textContent = `${at + 1} / ${order.length}`;
  $("info").textContent = `${f.cam}  ${f.time}`;
  status();
}
let zoom = 1;
function fit() {
  const W = $("wrap").clientWidth, H = window.innerHeight * 0.78;
  scale = Math.min(W / img.naturalWidth, H / img.naturalHeight, 1.5) * zoom;
  cv.style.maxWidth = zoom > 1 ? "none" : "100%";
  $("zoomv").textContent = zoom + "x";
  cv.width = Math.round(img.naturalWidth * scale); cv.height = Math.round(img.naturalHeight * scale);
}
function status() {
  const f = frame(), s = f && labels[f.file] ? (stale(f) ? "stale" : labels[f.file].status || "") : "";
  $("status").textContent = s === "done" ? "done" : s === "skip" ? "skipped" : s === "stale" ? "cut again - check it, Enter" : "not done";
  $("status").className = s;
  const all = DATA.frames.filter(isDone).map(f => labels[f.file]), d = all;
  const named = d.reduce((n, l) => n + l.boxes.filter(b => b[4]).length, 0), cows = Object.keys(knownIds()).length;
  $("stat").textContent = `${d.length} of ${DATA.frames.length} frames done, ${d.reduce((n, l) => n + l.boxes.length, 0)} cows (${named} with an id; ${cows} ids), ${DATA.frames.filter(f => labels[f.file] && labels[f.file].status === "skip" && !stale(f)).length} skipped${DATA.frames.some(stale) ? `, ${DATA.frames.filter(stale).length} cut again - to check` : ""}`;
}
function draw() {
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  const X = v => v * cv.width, Y = v => v * cv.height;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]); ctx.strokeStyle = "#f0a020"; ctx.font = "12px sans-serif"; ctx.fillStyle = "#f0a020";
  cands.forEach(c => { ctx.strokeRect(X(c[0]), Y(c[1]), X(c[2] - c[0]), Y(c[3] - c[1])); ctx.fillText(c[4].toFixed(2), X(c[0]) + 3, Y(c[1]) + 13); });
  ctx.setLineDash([]);
  const anyId = boxes.some(b => b[4]);
  boxes.forEach((b, i) => {
    ctx.strokeStyle = i === sel ? "#ffe94a" : "#3ee07a";
    ctx.strokeRect(X(b[0]), Y(b[1]), X(b[2] - b[0]), Y(b[3] - b[1]));
    if (i === sel) { ctx.fillStyle = "#ffe94a"; corners(b).forEach(([x, y]) => ctx.fillRect(X(x) - 5, Y(y) - 5, 10, 10)); }
    if (b[4] || anyId) {                              // its cow id ("?" once the frame has some)
      const t = b[4] || "?";
      ctx.font = "bold 14px sans-serif";
      const w = ctx.measureText(t).width + 8;
      ctx.fillStyle = b[5] ? "rgba(30,90,180,0.85)" : b[4] ? "rgba(20,20,20,0.75)" : "rgba(160,40,40,0.8)";
      ctx.fillRect(X(b[0]), Y(b[1]) - 19, w, 19);
      ctx.fillStyle = "#fff"; ctx.fillText(t, X(b[0]) + 4, Y(b[1]) - 5);
    }
  });
  $("cid").disabled = sel < 0;
  $("cid").value = sel >= 0 ? (boxes[sel][4] || "") : "";
}
const corners = b => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
// a box is [x1, y1, x2, y2] or [x1, y1, x2, y2, "cow id"]
// [.., "cow id", 1]: an id filled in from a hand-set one on another frame (F), not typed
const withId = (xy, b) => b && b[4] ? xy.concat(b.slice(4)) : xy;
const rnd = b => withId(b.slice(0, 4).map(v => Math.round(v * 1e4) / 1e4), b);
function knownIds() {
  const n = {};
  Object.values(labels).forEach(l => (l.boxes || []).forEach(b => { if (b[4]) n[b[4]] = (n[b[4]] || 0) + 1; }));
  return n;
}
function idList() { $("ids").innerHTML = Object.keys(knownIds()).sort().map(k => `<option value="${k.replace(/"/g, "&quot;")}">`).join(""); }
function setId(v) {
  if (sel < 0) return;
  v = (v || "").trim().replace(/,/g, " ");
  push();
  boxes[sel] = v ? boxes[sel].slice(0, 4).concat([v]) : boxes[sel].slice(0, 4);
  if (v) lastId = v;
  edited(); idList();
}
function nextUnnamed() {          // left to right: the next box with no id, or none
  const order = boxes.map((b, i) => [i, b]).sort((p, q) => p[1][0] - q[1][0]);
  const k = order.find(([i, b]) => !b[4] && i !== sel);
  return k ? k[0] : -1;
}
function fillIds() {              // the ids typed by hand carried along this camera's frames, both ways, while
  const f = frame(); if (!f) return;  // the cow's box on the next frame is the one most like hers (and she its)
  const only = $("onlyedited").checked;
  const frs = DATA.frames.filter(g => g.cam === f.cam && (!only || labels[g.file])).sort((a, b) => a.ts - b.ts);
  const B = frs.map(g => (labels[g.file] ? labels[g.file].boxes : detBoxes(g)[0]).map(b => b[5] ? b.slice(0, 4) : b.slice()));
  const dist = B.map(bs => bs.map(b => b[4] ? 0 : Infinity));
  const best = (b, bs) => { let k = -1, v = 0; bs.forEach((q, j) => { const u = iou(b, q); if (u > v) { v = u; k = j; } }); return [k, v]; };
  B.forEach((bs, k0) => bs.forEach((b0, j0) => {
    if (!b0[4] || b0[5]) return;
    for (const dir of [1, -1]) {
      let k = k0, j = j0, d = 0;
      for (;;) {
        const k2 = k + dir;
        if (k2 < 0 || k2 >= B.length || Math.abs(frs[k2].ts - frs[k].ts) > 300) break;   // not across another recording
        const [j2, v] = best(B[k][j], B[k2]);
        if (j2 < 0 || v < 0.5 || best(B[k2][j2], B[k])[0] !== j) break;     // not clearly the same cow
        const t = B[k2][j2]; d++;
        if (t[4] && !t[5]) break;                                           // a hand-set id: hers to carry on
        if (B[k2].some((q, i) => i !== j2 && q[4] === b0[4] && !q[5])) break;  // she is elsewhere on that frame
        if (dist[k2][j2] <= d) break;                                       // a nearer hand-set id holds it
        B[k2][j2] = t.slice(0, 4).concat([b0[4], 1]); dist[k2][j2] = d;
        k = k2; j = j2;
      }
    }
  }));
  let n = 0, frames_ = 0;
  frs.forEach((g, k) => {
    const was = labels[g.file], auto = B[k].filter(b => b[5]).length;
    if (!was && !auto) return;
    labels[g.file] = {status: was ? was.status || "" : "", cut: was ? was.cut : g.cut, boxes: B[k].map(rnd)};
    n += auto; frames_ += auto ? 1 : 0;
  });
  store(); open(); idList();
  $("info").textContent += ` - ${n} ids filled in on ${frames_} frames of ${f.cam} (blue): check them`;
}
function idsFromPrevious() {      // each box without an id takes the id of the box most like it on this camera's
  const f = frame(), i = DATA.frames.indexOf(f);   // frame before (a lying cow keeps her place)
  const prev = DATA.frames.slice(0, i).reverse().find(g => g.cam === f.cam && labels[g.file] && (labels[g.file].boxes || []).some(b => b[4]));
  if (!prev) { alert("no earlier frame of this camera with ids"); return; }
  push();
  const pb = labels[prev.file].boxes.filter(b => b[4]), used = new Set(boxes.filter(b => b[4]).map(b => b[4]));
  let n = 0;
  boxes = boxes.map(b => {
    if (b[4]) return b;
    const best = pb.filter(q => !used.has(q[4])).map(q => [iou(b, q), q]).sort((x, y) => y[0] - x[0])[0];
    if (!best || best[0] < 0.3) return b;
    used.add(best[1][4]); n++;
    return b.slice(0, 4).concat([best[1][4]]);
  });
  edited();
  $("info").textContent = `${f.cam}  ${f.time}  - ${n} ids from ${prev.time}: check them`;
}
function edited() {
  const f = frame();
  const was = labels[f.file] || {};
  labels[f.file] = {status: was.status || "", cut: was.cut, boxes: boxes.map(rnd)};
  store(); status(); draw();
}
function push() { undo.push(JSON.stringify([boxes, cands])); if (undo.length > 50) undo.shift(); }
function pt(e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; }
cv.addEventListener("pointerdown", e => {
  if (!frame()) return;
  const [x, y] = pt(e), r = cv.getBoundingClientRect(), tol = 8 / r.width, toly = 8 / r.height;
  cv.setPointerCapture(e.pointerId);
  if (sel >= 0) {                                   // the selected box: its corners resize it, its inside moves it
    const k = corners(boxes[sel]).findIndex(([cx, cy]) => Math.abs(cx - x) < tol && Math.abs(cy - y) < toly);
    if (k >= 0) { push(); drag = {mode: "resize", k}; return; }
    const b = boxes[sel];
    if (!e.shiftKey && x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) { push(); drag = {mode: "move", x, y, b: b.slice()}; return; }
  }
  // anywhere else: a drag draws a new box (also over other cows' boxes - a herd lying close together);
  // a click without dragging selects the box under it, or keeps a dashed one
  const inside = boxes.map((b, i) => [i, b]).filter(([, b]) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
    .sort((p, q) => (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]) - (q[1][2] - q[1][0]) * (q[1][3] - q[1][1]));
  drag = {mode: "press", x, y, px: e.clientX, py: e.clientY, hit: inside.length ? inside[0][0] : -1,
          cand: cands.findIndex(b => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])};
});
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  const [x, y] = pt(e).map(v => Math.min(1, Math.max(0, v)));
  if (drag.mode === "press") {
    if (Math.abs(e.clientX - drag.px) + Math.abs(e.clientY - drag.py) < 6) return;
    push(); boxes.push([drag.x, drag.y, drag.x, drag.y]); sel = boxes.length - 1; drag = {mode: "new", x: drag.x, y: drag.y};
  }
  const b = boxes[sel];
  if (drag.mode === "move") {
    const dx = Math.min(1 - drag.b[2], Math.max(-drag.b[0], x - drag.x)), dy = Math.min(1 - drag.b[3], Math.max(-drag.b[1], y - drag.y));
    boxes[sel] = withId([drag.b[0] + dx, drag.b[1] + dy, drag.b[2] + dx, drag.b[3] + dy], drag.b);
  } else if (drag.mode === "new") {
    boxes[sel] = [Math.min(drag.x, x), Math.min(drag.y, y), Math.max(drag.x, x), Math.max(drag.y, y)];
  } else {
    const k = drag.k;
    if (k === 0 || k === 3) b[0] = x; else b[2] = x;
    if (k === 0 || k === 1) b[1] = y; else b[3] = y;
    boxes[sel] = withId([Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])], b);
    if (b[0] > b[2]) drag.k = [1, 0, 3, 2][drag.k];
    if (b[1] > b[3]) drag.k = [3, 2, 1, 0][drag.k];
  }
  draw();
});
cv.addEventListener("pointerup", () => {
  if (!drag) return;
  if (drag.mode === "press") {                      // a click
    if (drag.hit >= 0) sel = drag.hit;
    else if (drag.cand >= 0) { push(); boxes.push(cands[drag.cand].slice(0, 4)); cands.splice(drag.cand, 1); sel = boxes.length - 1; drag = null; edited(); return; }
    else sel = -1;
    drag = null; draw(); return;
  }
  const b = boxes[sel];
  if (drag.mode === "new" && ((b[2] - b[0]) * cv.width < 6 || (b[3] - b[1]) * cv.height < 6)) { boxes.splice(sel, 1); sel = -1; undo.pop(); draw(); }
  else edited();
  drag = null;
});
function mark(s) {
  const f = frame(); if (!f) return;
  labels[f.file] = {status: s, cut: f.cut, boxes: boxes.map(rnd)};
  store(); status();
  if (at + 1 < order.length) { at++; open(); } else draw();
}
function go(d) { if (!order.length) return; at = Math.min(order.length - 1, Math.max(0, at + d)); open(); }
let lastId = "";
$("cid").addEventListener("keydown", e => {
  if (e.key === "Enter") {
    setId($("cid").value);
    const k = nextUnnamed();
    if (k >= 0) { sel = k; draw(); $("cid").focus(); } else { sel = -1; draw(); $("cid").blur(); }
    e.preventDefault(); e.stopPropagation();
  } else if (e.key === "Escape") { $("cid").blur(); }
});
$("cid").addEventListener("change", () => { if (sel >= 0 && $("cid").value.trim() !== (boxes[sel][4] || "")) setId($("cid").value); });
$("lastid").onclick = () => { if (sel >= 0 && lastId) setId(lastId); };
$("previd").onclick = () => idsFromPrevious();
$("fillid").onclick = () => fillIds();
const ZOOMS = [1, 1.5, 2, 3, 4];
function setZoom(d) { zoom = ZOOMS[Math.min(ZOOMS.length - 1, Math.max(0, ZOOMS.indexOf(zoom) + d))]; if (img.naturalWidth) { fit(); draw(); } }
$("zin").onclick = () => setZoom(1); $("zout").onclick = () => setZoom(-1);
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
$("done").onclick = () => mark("done"); $("skip").onclick = () => mark("skip");
$("reset").onclick = () => { const f = frame(); if (!f) return; push(); [boxes, cands] = detBoxes(f); sel = -1; edited(); };
["cam", "todo"].forEach(id => $(id).onchange = () => { filter(order[at]); open(); });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if ((e.ctrlKey || e.metaKey) && e.key === "z") { if (undo.length) { [boxes, cands] = JSON.parse(undo.pop()); sel = -1; edited(); } e.preventDefault(); return; }
  if (e.key === "Enter") { mark("done"); e.preventDefault(); }
  else if (e.key === "s" || e.key === "S") mark("skip");
  else if (e.key === "ArrowRight") go(1);
  else if (e.key === "ArrowLeft") go(-1);
  else if ((e.key === "Delete" || e.key === "Backspace") && sel >= 0) { push(); boxes.splice(sel, 1); sel = -1; edited(); e.preventDefault(); }
  else if (e.key === "a" || e.key === "A") { push(); boxes.push(...cands.map(c => c.slice(0, 4))); cands = []; edited(); }
  else if (e.key === "Escape") { sel = -1; draw(); }
  else if ((e.key === "i" || e.key === "I") && sel >= 0) { $("cid").focus(); $("cid").select(); e.preventDefault(); }
  else if (e.key === "p" || e.key === "P") idsFromPrevious();
  else if (e.key === "f" || e.key === "F") fillIds();
  else if (e.key === "+" || e.key === "=") setZoom(1);
  else if (e.key === "-") setZoom(-1);
});
$("save").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({frames: labels})], {type: "application/json"}));
  a.download = "labels.json"; a.click();
};
$("loadbtn").onclick = () => $("load").click();
$("load").onchange = e => { const f = e.target.files[0]; if (!f) return;
  f.text().then(t => { const d = JSON.parse(t); Object.assign(labels, d.frames || d); store(); open(); }); };
idList();
window.addEventListener("resize", () => { if (img.naturalWidth) { fit(); draw(); } });
filter(0); open();
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames", help="recordings -> frames + the detector's boxes + the labelling page")
    f.add_argument("--config", required=True, help="the barn's config: cameras (ids, exclude zones), detector")
    f.add_argument("--videos", default=None, help="a folder of recordings named <camera>..<date and time>")
    f.add_argument("--video", action="append", default=[], help="cam1=file.mp4 (with --start), instead of --videos")
    f.add_argument("--start", default=None, help="--video: when the recording began, local time (ISO)")
    f.add_argument("--out", required=True)
    f.add_argument("--every", type=float, default=30.0, help="a frame every this many seconds")
    f.add_argument("--max", type=int, default=300, help="frames a camera at most, spread over all its recordings")
    f.add_argument("--min-change", type=float, default=3.0, help="a frame this little changed (0-255) is skipped")
    f.add_argument("--detector", default=None, help="a best/ folder (default: the config's); none: no first guess")
    f.add_argument("--cameras", nargs="*", default=None, help="only these cameras")
    f.add_argument("--again", action="store_true", help="with --cameras: their frames already cut are cut again "
                   "with the zone as the config has it now (an exclude zone changed) and looked at again")
    g = sub.add_parser("page", help="the labelling page again (after a code update, or with a labels.json)")
    g.add_argument("--data", required=True)
    g.add_argument("--labels", default=None)
    gl = sub.add_parser("gallery", help="every id of every camera as a row of her crops -> ids.html (check, rename)")
    gl.add_argument("--data", required=True)
    gl.add_argument("--labels", required=True)
    gl.add_argument("--thumbs", type=int, default=8, help="crops a row")
    x = sub.add_parser("export", help="labels.json -> herd_det_barn_<date>.tgz (COCO, for run_pod.sh detector)")
    x.add_argument("--data", required=True)
    x.add_argument("--labels", required=True)
    x.add_argument("--out", default=None)
    x.add_argument("--val-share", type=float, default=0.2, help="frames held out to check the detector")
    x.add_argument("--block-min", type=float, default=60.0, help="held out by whole blocks of a camera's minutes")
    x.add_argument("--seed", type=int, default=0)
    n = sub.add_parser("ids", help="boxes named with a cow id -> herd_barn_ids_<date>.tgz (the identity model)")
    n.add_argument("--data", required=True)
    n.add_argument("--labels", required=True)
    n.add_argument("--out", default=None)
    n.add_argument("--min-crops", type=int, default=6, help="cows with fewer named boxes are left out")
    n.add_argument("--val-share", type=float, default=1 / 6, help="cows held out (never trained on)")
    n.add_argument("--margin", type=float, default=0.1, help="context round the box (the herd model's margin)")
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--done-only", action="store_true", help="only frames marked done (default: every frame with "
                   "named boxes - ids filled in on the page included)")
    n.add_argument("--fill", type=float, default=0.0, help="seconds: more crops of each named cow from the video "
                   "between two cut frames where she stayed in place, one every this many s (needs --config, --videos)")
    n.add_argument("--fill-iou", type=float, default=0.5, help="her two boxes must overlap this much to fill between")
    n.add_argument("--fill-gap", type=float, default=30.0, help="at most this many s between the two frames")
    n.add_argument("--config", default=None)
    n.add_argument("--videos", default=None)
    args = p.parse_args(argv)
    if args.cmd == "frames" and not (args.videos or args.video):
        p.error("frames: --videos DIR or --video cam=file")
    {"frames": cmd_frames, "page": cmd_page, "export": cmd_export, "ids": cmd_ids, "gallery": cmd_gallery}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
