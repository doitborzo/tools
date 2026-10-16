#!/usr/bin/env python3
"""Identity on sets with real cow ids: the fingerprint taught to know a cow across days, postures and views.

The herd model's fingerprint was taught on CBVD-5, which has no cow ids: it learnt "the same cow within
7 s" and knows the view, not the cow (MmCows: a cow seen by two cameras got one track by look in
0.1-1.4% of cases). These sets name their cows:

    multicam   MultiCamCows2024 (Yu et al., University of Bristol, data.bris 2inu67jru7a6821kkgehxg3cv2,
               Non-Commercial Government Licence): 90 Holstein-Friesians, 3 ceiling cameras over a
               walkway, 7 days, 101,329 crops - one folder per cow per day, the same number every day,
               file <frame>_<camera>.jpg (tracklets sampled at 5 Hz). Crops only (3.6 GB of 36.5 GB).
    cows2021   Cows2021 (Gao et al., Bristol, data.bris 4vnrca7qw1642qlwxjadp87h7, NCGL): its
               identification test set - 181 cows, ~8.7k top-down crops over 4 weeks (the training part
               is tracklets without ids and is not used).
    opencows   OpenCows2020 (Andrew et al., Bristol, NCGL): its identification part, 46 cows, ~4.9k
               top-down crops.
    cobra      CoBRA youngstock re-identification (Zenodo 15018518, CC BY 4.0): 48 Holstein heifers
               from a roof camera at an angle, 11,438 segments (black background), standing / lying
               left / lying right - the posture is taken as the "day" (the same cow lying and standing).
    sideview   SideViewCows2026 (Moeller, Zenodo 21605650, CC BY 4.0): 110 cows from the side - a fixed
               parlor camera, handheld barn video, snapshots (some lying in cubicles); each frame cut to
               its cow's mask, video thinned (barn every 5th frame, parlor every 10th); the subset is
               the "day".
    mmcows     MmCows (Vu et al., Purdue, CC BY-NC-SA): the evaluation's own pen, 4 high cameras from
               the corners - the barn's view. 7/25 has human labels of all 16 cows: outside mct_eval's
               12:00-15:00, cows 1-10 are trained on and cows 11-16 held out (never trained on, never
               in mct_eval's hours). Other days have no labels, but cows 1-10 wear a UWB neck tag:
               its position every 15 s, against the detector's boxes on the floor, labels whole tracks
               (tag_tracks; 95% right when checked on 7/25 against the human labels).

    fetch    the crops, read in place from the zips (HTTP Range) -> DATA/raw/<source>/<cow>/...;
             mmcows: frames, labels, tags and projection matrices -> DATA/mmcows/
    label    (GPU, mmcows) the detector on its frames -> crops cut as the barn cuts them, with the cow's
             number from the human labels (7/25) or the neck tags (other days) -> DATA/raw/mmcows
    extract  (GPU) DINOv2 on every crop, as the barn does (square crop with the model's margin, its
             encoder and grid), then pseudo-bursts: a camera's 7 s of one cow (35 crops at 5 Hz) or a
             handful of one cow's pictures of one day, stretched to the barn's 175 frames, each with
             its cow's id -> DATA/feat/<source>/{crops.npz, bursts.jsonl} (each crop's vector once, a
             burst a list of crops). val: a sixth of each set's cows, never seen in training (the
             barn's cows are new cows too); mmcows: cows 11-16.
    train    a copy of the herd model's temporal transformer - or, with --init, an identity model
             already trained - taught to put one cow's bursts together (supervised contrastive over
             cow ids, different days, postures and cameras are positives), a quarter of the views cut
             to a single frame (mct.py tracks by one frame a second). Scored on the val cows: the same
             cow on another day, from another camera (MmCows' cows 11-16 too), and from one frame
             -> OUT/identity.pt, eval.json, and abstain.json (the NaN model refitted on these
             fingerprints). The herd model itself (posture, activity, rumination) is kept.

Then [model] identity = ".../identity.pt" in barn.toml: the gallery and mct.py use its fingerprint,
the gallery its abstain.json. Licences: all are non-commercial or CC BY (see each).

    python herd/reid.py fetch   --data /workspace/reid
    python herd/reid.py label   --data /workspace/reid --model /workspace/herd/run9/model.pt --detector <best/>
    python herd/reid.py extract --data /workspace/reid --model /workspace/herd/run9/model.pt
    python herd/reid.py train   --data /workspace/reid --model /workspace/herd/run9/model.pt \\
                                --init /workspace/reid/id_model/identity.pt --out /workspace/reid/id_model_ft
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import json
import math
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import read_jsonl, write_json  # noqa: E402

ZENODO = "https://zenodo.org/records"
MMCOWS = "https://huggingface.co/datasets/neis-lab/mmcows/resolve/main"
SOURCES = {
    "multicam": {"url": "https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip",
                 "licence": "Non-Commercial Government Licence", "cite": "Yu et al., MultiCamCows2024, arXiv 2410.12695"},
    "cows2021": {"url": "https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip",
                 "licence": "Non-Commercial Government Licence", "cite": "Gao et al., Cows2021, arXiv 2105.01938"},
    "opencows": {"url": None, "licence": "UK Non-Commercial Government Licence v2",
                 "cite": "Andrew et al., OpenCows2020"},          # the url is extra_det's (one zip for both)
    # Zenodo answers many Range requests with 429: its zips are downloaded whole (0.7 GB; 25 GB) and read locally
    "cobra": {"url": f"{ZENODO}/15018518/files/re_identification_youngstock.zip?download=1", "licence": "CC BY 4.0",
              "cite": "CoBRA re-identification youngstock, Zenodo 15018518", "whole": True},
    "sideview": {"url": {s: f"{ZENODO}/21605650/files/{s}.zip?download=1" for s in ("snapshots", "barn", "parlor")},
                 "licence": "CC BY 4.0", "cite": "Moeller, SideViewCows2026, Zenodo 21605650", "whole": True},
    "mmcows": {"url": MMCOWS, "licence": "CC BY-NC-SA 4.0", "cite": "Vu et al., MmCows, NeurIPS 2024"},
}
SIDEVIEW_STRIDE = {"snapshots": 1, "barn": 5, "parlor": 10}   # video frames: every k-th (neighbours are alike)
SLOTS, FPS = 175, 25.0                   # the barn's burst: 7 s at 25 fps
WIN = 35                                 # MultiCamCows2024: 7 s of a tracklet at 5 Hz
PICS = 8                                 # stills of one cow on one day: a burst of up to this many


# ------------------------------------------------------------------ fetch

def plan(source, names):
    """Zip member names -> [(cow, day, cam, order, member)] of one source."""
    out = []
    for n in names:
        low = n.lower()
        if not low.endswith((".jpg", ".jpeg", ".png")):
            continue
        if source == "multicam":
            m = re.search(r"MultiCamCows2024Root/(2023\w{3}\d\d)/(\d{3})/(\d+)_(\d+)\.jpg$", n)
            if m:
                out.append((m.group(2), m.group(1), m.group(4), int(m.group(3)), n))
        elif source == "cows2021":
            m = re.search(r"/Identification/Test/(\d+)/image_(\d+)_(\d{4}-\d\d-\d\d)_", n)
            if m:
                out.append((m.group(1), m.group(3), "0", int(m.group(2)), n))
        elif source == "opencows":
            m = re.search(r"/identification/images/(?:train|test)/(\d+)/([^/]+)$", n)
            if m:
                out.append((m.group(1), "all", "0", len(out), n))
        elif source == "cobra":
            # the posture is the "day": a cow standing and the same cow lying are two looks of her
            m = re.search(r"(?:^|/)(standing|lying_left|lying_right)/(\d+)/(\d+)\.(?:jpg|png)$", n)
            if m:
                out.append((m.group(2), m.group(1), "0", int(m.group(3)), n))
        elif source == "sideview":
            # the subset is the "day": the parlor's fixed camera, handheld barn video, snapshots
            m = re.search(r"(?:^|/)(parlor|barn|snapshots)/images/(\w+)/\w+?_(\d+)\.jpg$", n)
            if m and int(m.group(3)) % SIDEVIEW_STRIDE[m.group(1)] == 0:
                out.append((m.group(2), m.group(1), "0", int(m.group(3)), n))
    return out


def mask_crop(image, mask, pad=0.04, max_side=640):
    """SideViewCows: the frame cut to its cow's mask (other cows share the frame) -> JPEG bytes, or None."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(image)).convert("RGB")
    m = np.asarray(Image.open(io.BytesIO(mask)).convert("L")) > 127
    ys, xs = np.nonzero(m)
    if len(xs) < 400:
        return None
    w, h = im.size
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    im = im.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px) + 1), min(h, int(y2 + py) + 1)))
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def as_jpeg(data):
    """Any image bytes -> JPEG bytes (CoBRA's segments are partly PNG)."""
    if data[:3] == b"\xff\xd8\xff":
        return data
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


def cmd_fetch(args):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cowbench"))
    from extra_det import SOURCES as DET_SOURCES
    from extra_det import LocalZip, RemoteZip, download
    for src in args.sources:
        if src == "mmcows":
            fetch_mmcows(args)
            continue
        url = SOURCES[src]["url"] or DET_SOURCES["opencows"]["url"]
        folder = os.path.join(args.data, "raw", src)
        os.makedirs(folder, exist_ok=True)
        done_mark = os.path.join(folder, "complete.json")
        if os.path.exists(done_mark) and os.path.exists(os.path.join(folder, "index.jsonl")):
            print(f"[fetch] {src}: there already ({json.load(open(done_mark))['crops']} crops)", flush=True)
            continue
        rows, todo, zips = [], [], []
        urls = url if isinstance(url, dict) else {"": url}
        for part, url in urls.items():
            if SOURCES[src].get("whole"):
                path = os.path.join(args.data, "zips", f"{src}{'_' + part if part else ''}.zip")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                print(f"[fetch] {src}{' ' + part if part else ''}: downloading the zip whole", flush=True)
                z = LocalZip(download(url, path))
                zips.append(path)
            else:
                z = RemoteZip(url)
            names = z.names()
            items = plan(src, names)
            if args.max_per_source:
                items = items[:args.max_per_source]
            have = set(names)
            for cow, day, cam, order, member in items:
                name = f"{cow}/{day}_{cam}_{order:07d}.jpg"
                mask = member.replace("/images/", "/masks/")[:-4] + ".png" if src == "sideview" else None
                if mask is not None and mask not in have:
                    continue
                rows.append({"cow": cow, "day": day, "cam": cam, "order": order, "file": name})
                if not os.path.exists(os.path.join(folder, name)) and \
                        not os.path.exists(os.path.join(folder, name + ".empty")):
                    todo.append((z, member, mask, name))
        print(f"[fetch] {src}: {len(rows)} crops of {len({r['cow'] for r in rows})} cows, {len(todo)} to fetch "
              f"({SOURCES[src]['licence']})", flush=True)

        def one(job):
            z, member, mask, name = job
            path = os.path.join(folder, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = mask_crop(z.read(member), z.read(mask)) if mask else as_jpeg(z.read(member))
            if data is None:                               # an empty mask: no cow to cut
                open(path + ".empty", "w").close()
                return
            open(path + ".part", "wb").write(data)
            os.replace(path + ".part", path)

        t0, done = time.time(), 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for _ in ex.map(one, todo):
                done += 1
                if done % 2000 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"\r  {done}/{len(todo)}  {el / 60:.1f} min  ~{el / done * (len(todo) - done) / 60:.0f} min left",
                          end="", flush=True)
        print(flush=True)
        rows = [r for r in rows if os.path.exists(os.path.join(folder, r["file"]))]
        with open(os.path.join(folder, "index.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        if not args.max_per_source:
            write_json(done_mark, {"crops": len(rows), "cows": len({r["cow"] for r in rows})})
            for path in zips if not args.keep_zips else []:
                os.remove(path)                            # the crops are cut: the GBs of zip go


# ------------------------------------------------------------------ MmCows (the domain of mct_eval.py)

MM_CAMS = ("cam_1", "cam_2", "cam_3", "cam_4")
MM_DAY = "0725"                                   # the one day with human labels: every cow's number, 4 cameras
MM_EVAL = (1690304400 - 1800, 1690304400 + 3 * 3600 + 1800)   # mct_eval's 7/25 12:00-15:00 (+-30 min): never used
MM_TAGGED = set(range(1, 11))                     # UWB neck tags T01-T10: cows 11-16 are never trained on
MM_PROJ = (4480, 2800)                            # the pixels the projection matrices map to


def _mm_ts(name):
    m = re.search(r"(\d{10})(?:_[\d-]+)?\.(?:jpg|txt)$", name)
    return int(m.group(1)) if m else None


def _every(ts, every):
    """On MmCows' shared 15 s grid (:11, :26, ...): every every-th second, the same seconds on every camera."""
    return (ts // 15) % max(1, int(every) // 15) == 0


def _save(path, read):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path + ".part", "wb").write(read())
        os.replace(path + ".part", path)


def fetch_mmcows(args):
    """MmCows, read in place: 7/25's frames with the human labels (cows 1-10 train, 11-16 the held-out
    check), the --mm-uwb-days' frames, the neck tags' positions and the projection matrices."""
    from extra_det import RemoteZip, shrink
    root = os.path.join(args.data, "mmcows")
    pz = RemoteZip(f"{MMCOWS}/uwb_synced_images/proj_mat.zip")
    for n in pz.names():
        if n.endswith(".xml"):
            _save(os.path.join(root, n), lambda n=n: pz.read(n))          # proj_mat/<day>/proj_mat_camN.xml
    days = [d for d in args.mm_uwb_days if d != MM_DAY]
    sz = RemoteZip(f"{MMCOWS}/sensor_data.zip")
    have = set(sz.names())
    for d in days + [MM_DAY]:
        for t in sorted(MM_TAGGED):
            n = f"sensor_data/main_data/uwb/T{t:02d}/T{t:02d}_{d}.csv"
            if n in have:
                _save(os.path.join(root, "uwb", d, f"T{t:02d}.csv"), lambda n=n: sz.read(n))
    vz = RemoteZip(f"{MMCOWS}/visual_data.zip")
    names = vz.names()
    labels = [n for n in names if f"/labels/combined/{MM_DAY}/" in n and n.endswith(".txt")
              and n.split("/")[-2] in MM_CAMS and _every(_mm_ts(n), args.mm_label_every)
              and not MM_EVAL[0] <= _mm_ts(n) < MM_EVAL[1]]
    jobs = [(vz, n, os.path.join(root, "labels", n.split("/")[-2], f"{_mm_ts(n)}.txt"), False) for n in labels]
    marked = {(n.split("/")[-2], _mm_ts(n)) for n in labels}
    # 7/25's frames are visual_data's (the other pens blacked out, as the labels were drawn on them)
    jobs += [(vz, n, os.path.join(root, "frames", MM_DAY, n.split("/")[-2], f"{_mm_ts(n)}.jpg"), True)
             for n in names if f"/images/{MM_DAY}/" in n and n.endswith(".jpg")
             and (n.split("/")[-2], _mm_ts(n)) in marked]
    for d in days:
        z = RemoteZip(f"{MMCOWS}/uwb_synced_images/{d}.zip")
        jobs += [(z, n, os.path.join(root, "frames", d, n.split("/")[-2], f"{_mm_ts(n)}.jpg"), True)
                 for n in z.names() if n.endswith(".jpg") and n.split("/")[-2] in MM_CAMS
                 and _every(_mm_ts(n), args.mm_uwb_every)]
    todo = [j for j in jobs if not os.path.exists(j[2])]
    print(f"[fetch] mmcows: {len(labels)} labelled frames of {MM_DAY} (outside mct_eval's hours), "
          f"{sum(1 for j in jobs if j[3]) - len(labels)} frames of {', '.join(days) or 'no other day'} "
          f"(every {args.mm_uwb_every} s); {len(todo)} files to fetch ({SOURCES['mmcows']['licence']})", flush=True)

    def one(job):
        z, n, path, image = job
        _save(path, (lambda: shrink(z.read(n), args.mm_max_side)[0]) if image else (lambda: z.read(n)))

    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for _ in ex.map(one, todo):
            done += 1
            if done % 500 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"\r  {done}/{len(todo)}  {el / 60:.1f} min  ~{el / done * (len(todo) - done) / 60:.0f} min left",
                      end="", flush=True)
    print(flush=True)


def mm_mats(root, day):
    """{cam: floor mapper (box -> floor X, Y cm)} of one day, from its projection matrices."""
    from common import DEFAULTS, floor_mapper
    out = {}
    for cam in MM_CAMS:
        path = os.path.join(root, "proj_mat", day, f"proj_mat_cam{cam.split('_')[-1]}.xml")
        if os.path.exists(path):
            m = re.search(r"<projection_matrix.*?<data>(.*?)</data>", open(path, encoding="utf-8").read(), re.S)
            P = [float(x) for x in m.group(1).split()]
            out[cam] = (np.asarray(P).reshape(3, 4), floor_mapper({"proj_mat": P, "proj_size": MM_PROJ}, DEFAULTS["mct"]))
    return out


def mm_uwb(root, day):
    """{ts: [(tag, [x, y, z] cm)]} of one day's neck tags (a reading every 15 s, on the frames' seconds)."""
    import csv
    out = collections.defaultdict(list)
    folder = os.path.join(root, "uwb", day)
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        tag = int(f[1:3])
        for r in csv.DictReader(open(os.path.join(folder, f), encoding="utf-8")):
            xyz = [float(r[k]) for k in ("coord_x_cm", "coord_y_cm", "coord_z_cm")]
            if all(math.isfinite(v) for v in xyz):
                out[int(float(r["timestamp"]))].append((tag, xyz))
    return out


def in_view(P, xyz, edge=0.02):
    q = P @ np.array([*xyz, 1.0])
    u, v = q[0] / q[2] / MM_PROJ[0], q[1] / q[2] / MM_PROJ[1]
    return q[2] > 0 and edge <= u <= 1 - edge and edge <= v <= 1 - edge


def tag_tracks(frames, tags, floor, P, slack=60.0, gate=150.0, min_votes=4, share=0.9, iou_min=0.3, gap=2):
    """One camera's boxes over a day, labelled by the neck tags (no human labels): an IoU tracker (lying
    cows keep their box for hours); every second the tags in view go to the boxes by the Hungarian
    assignment on floor distance; a track takes a tag only when that tag won it at >= share of its
    seconds (>= min_votes). The tag sits on the neck, up to ~1 m from the body's centre along the stall
    (y), so y is forgiven by slack cm. Checked on 7/25 against the human labels: 95% right at share 0.9
    (the mistakes are mostly an untagged cow lying next to a tagged one), a seventh of the boxes kept.
    frames [(ts, [box])] in time order, tags {ts: [(tag, xyz)]} -> {(ts, j): tag}."""
    from scipy.optimize import linear_sum_assignment
    from common import iou
    step = min((b[0] - a[0] for a, b in zip(frames, frames[1:])), default=15) or 15
    tracks, out = [], {}
    for ts, boxes in frames:
        live = [k for k, t in enumerate(tracks) if ts - t["last"] <= gap * step + 1]
        pairs = {}
        if live and boxes:
            M = np.array([[iou(tracks[k]["box"], b) for b in boxes] for k in live])
            r, c = linear_sum_assignment(-M)
            pairs = {j: live[i] for i, j in zip(r, c) if M[i, j] >= iou_min}
        tid = []
        for j, b in enumerate(boxes):
            k = pairs.get(j)
            if k is None:
                tracks.append({"items": [], "votes": collections.Counter()})
                k = len(tracks) - 1
            tracks[k]["box"], tracks[k]["last"] = b, ts
            tracks[k]["items"].append((ts, j))
            tid.append(k)
        here = [(t, xyz) for t, xyz in tags.get(ts, []) if in_view(P, xyz)]
        if here and boxes:
            pts = [floor(b) for b in boxes]
            D = np.array([[math.hypot(x - p[0], max(0.0, abs(y - p[1]) - slack)) for p in pts]
                          for _, (x, y, _) in here])
            r, c = linear_sum_assignment(D)
            for i, j in zip(r, c):
                if D[i, j] <= gate:
                    tracks[tid[j]]["votes"][here[i][0]] += 1
    for tr in tracks:
        if tr["votes"]:
            t, v = tr["votes"].most_common(1)[0]
            if v >= min_votes and v / len(tr["items"]) >= share:
                out.update({key: t for key in tr["items"]})
    return out


def read_mm_labels(path):
    """MmCows' 'cow cx cy w h' (0-1) -> [(cow, [x1, y1, x2, y2])]."""
    rows = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 5:
            c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            rows.append((c, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return rows


def cmd_label(args):
    """(GPU) MmCows' frames -> the barn's crops with cow numbers -> raw/mmcows (as fetch does for the others):
    7/25 by its human labels (cows 1-10 train, 11-16 val: never trained on), the other days by the
    neck tags (tag_tracks on the detector's boxes); 7/25's tag labels are checked against the human ones."""
    import copy as _copy
    import pipeline
    from PIL import Image
    from common import DEFAULTS, crop, iou
    root = os.path.join(args.data, "mmcows")
    raw = os.path.join(args.data, "raw", "mmcows")
    days = sorted(os.listdir(os.path.join(root, "frames"))) if os.path.isdir(os.path.join(root, "frames")) else []
    if args.mm_days:                  # only these (fully fetched) days; the index still lists every labelled day
        days = [d for d in days if d in args.mm_days]
    if not days:
        sys.exit("no MmCows frames: reid.py fetch --sources mmcows first")
    cfg = _copy.deepcopy(DEFAULTS)
    cfg["model"]["checkpoint"] = args.model
    cfg["detector"]["weights"] = args.detector
    models = pipeline.Models(cfg)
    size, margin = models.crop_size, models.margin
    rows, check = [], collections.Counter()
    for day in days:
        mats, tags = mm_mats(root, day), mm_uwb(root, day)
        for cam in MM_CAMS:
            folder = os.path.join(root, "frames", day, cam)
            if not os.path.isdir(folder) or cam not in mats:
                continue
            done_path = os.path.join(root, "labelled", f"{day}_{cam}.jsonl")
            if os.path.exists(done_path) and not args.force:
                rows += read_jsonl(done_path)
                continue
            tss = sorted(int(f[:-4]) for f in os.listdir(folder) if f.endswith(".jpg"))
            load = lambda ts: np.asarray(Image.open(os.path.join(folder, f"{ts}.jpg")).convert("RGB"))
            frames, t0 = [], time.time()
            with cf.ThreadPoolExecutor(4) as ex:
                for k in range(0, len(tss), args.batch):
                    chunk = tss[k:k + args.batch]
                    frames += list(zip(chunk, models.detect_many(list(ex.map(load, chunk)), batch=args.batch)))
                    if (k // args.batch) % 25 == 0:
                        print(f"\r  {day} {cam}: {k + len(chunk)}/{len(tss)} frames  {(time.time() - t0) / 60:.1f} min",
                              end="", flush=True)
            P, floor = mats[cam]
            by_tag = tag_tracks(frames, tags, floor, P, share=args.mm_share)
            human = {}
            if day == MM_DAY:
                for ts, boxes in frames:
                    lab = os.path.join(root, "labels", cam, f"{ts}.txt")
                    gt = read_mm_labels(lab) if os.path.exists(lab) else []
                    for j, b in enumerate(boxes):                 # the detector's box of each labelled cow
                        best = max(gt, key=lambda g: iou(g[1], b), default=None)
                        if best is not None and iou(best[1], b) >= 0.5:
                            human[(ts, j)] = best[0]
                for key, t in by_tag.items():
                    if key in human:
                        check["tag_labels"] += 1
                        check["tag_right"] += human[key] == t
            labelled = human if day == MM_DAY else by_tag
            mine = []
            for ts, boxes in frames:
                todo = [(j, labelled[(ts, j)]) for j in range(len(boxes)) if (ts, j) in labelled]
                if not todo:
                    continue
                img = load(ts)
                for j, cow in todo:
                    name = f"{cow:02d}/{day}_{cam}_{ts}.jpg"
                    path = os.path.join(raw, name)
                    if not os.path.exists(path):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        Image.fromarray(crop(img, boxes[j], size, margin)).save(path, quality=92)
                    split = "val" if day == MM_DAY and cow not in MM_TAGGED else "train"
                    mine.append({"cow": f"{cow:02d}", "day": day, "cam": cam, "order": ts, "file": name,
                                 "ready": True, "split": split, "by": "human" if day == MM_DAY else "uwb"})
            os.makedirs(os.path.dirname(done_path), exist_ok=True)
            with open(done_path, "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in mine)
            rows += mine
            print(f"\r[label] {day} {cam}: {len(frames)} frames, {sum(len(b) for _, b in frames)} boxes, "
                  f"{len(mine)} crops with a cow's number ({'human labels' if day == MM_DAY else 'neck tags'})",
                  flush=True)
    os.makedirs(raw, exist_ok=True)
    done = os.path.join(root, "labelled")
    rows = [r for f in sorted(os.listdir(done)) if f.endswith(".jsonl") for r in read_jsonl(os.path.join(done, f))]
    with open(os.path.join(raw, "index.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    n = collections.Counter((r["by"], r["split"]) for r in rows)
    print(f"[label] mmcows: {n['human', 'train']} crops of cows 1-10 on {MM_DAY} (human labels), "
          f"{n['uwb', 'train']} of the other days (neck tags), {n['human', 'val']} of cows 11-16 held out", flush=True)
    if check["tag_labels"]:
        print(f"[label] check on {MM_DAY}: the neck tags' labels agree with the human ones on "
              f"{check['tag_right'] / check['tag_labels']:.1%} of {check['tag_labels']} boxes", flush=True)
    old = os.path.join(root, "label.json")
    if not check["tag_labels"] and os.path.exists(old):               # 7/25 labelled by an earlier call
        check = json.load(open(old, encoding="utf-8")).get("tag_check", {})
    write_json(old, {"rows": dict((f"{a}/{b}", c) for (a, b), c in n.items()), "tag_check": dict(check)})


# ------------------------------------------------------------------ extract

def stretch(m, slots=SLOTS):
    """m frames over the burst's slots: slot j shows frame floor(j * m / slots)."""
    return np.minimum((np.arange(slots) * m) // slots, m - 1)


def make_bursts(rows, rng, val_cows, tracklets):
    """One source's crops (rows with "k", the crop's index into the features) -> pseudo-bursts
    [{"cow", "day", "cam", "frames": [k...], "split"}]: per cow, day and camera in time order, windows
    of WIN crops (7 s at 5 Hz) for tracklets; stills (no order: one picture per "order") in random
    groups of up to PICS of one day."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["cow"], r["day"], r["cam"], r.get("split"))].append(r)
    out = []
    for (cow, day, cam, _), rs in sorted(groups.items(), key=lambda kv: kv[0][:3]):
        rs.sort(key=lambda r: r["order"])
        size = WIN if tracklets else PICS
        if not tracklets:
            rng.shuffle(rs)
        chunks = [rs[i:i + size] for i in range(0, len(rs), size)]
        if len(chunks) > 1 and len(chunks[-1]) < max(2, size // 3):
            tail = chunks.pop()
            chunks[-1] += tail                               # a short tail joins the window before it
        for ch in chunks:
            if len(ch) >= 2:
                split = ch[0].get("split") or ("val" if cow in val_cows else "train")    # mmcows: set by label
                out.append({"cow": cow, "day": day, "cam": cam, "frames": [r["k"] for r in ch], "split": split})
    return out


def write_bursts(folder, bursts, src, meta):
    """One set's pseudo-bursts as lists of crops (feat/<src>/bursts.jsonl, next to crops.npz): every
    crop's vector is kept once and a burst is stretched to the barn's 175 frames only when used."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "bursts.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(dict(b, identity=f"{src}:{b['cow']}")) + "\n" for b in bursts)
    write_json(os.path.join(folder, "meta.json"), dict(meta, source=src))


def encode_all(paths, meta, batch=256, ready=None):
    """Crops on disk -> the barn's frame vectors (fp16): each crop squared with the model's margin (a
    ready one - mmcows, cut from its frame as the barn cuts it - taken as it is)."""
    import torch
    from PIL import Image
    from common import crop, device
    from model import FrameEncoder
    dev = device()
    enc = FrameEncoder(meta["encoder"], meta["grid"]).to(dev).eval()
    ready = set() if ready is None else {p for p, r in zip(paths, ready) if r}
    load = lambda p: crop(np.asarray(Image.open(p).convert("RGB")), [0.0, 0.0, 1.0, 1.0], meta["crop"],
                          0.0 if p in ready else meta["margin"])
    out, t0 = [], time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for i in range(0, len(paths), batch):
            x = torch.from_numpy(np.stack(list(ex.map(load, paths[i:i + batch])))).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(enc(x).float().cpu().numpy().astype(np.float16))
            done = min(i + batch, len(paths))
            if (i // batch) % 20 == 0 or done == len(paths):
                el = time.time() - t0
                print(f"\r  {done}/{len(paths)} crops  {el / 60:.1f} min  ~{el / done * (len(paths) - done) / 60:.0f} min left",
                      end="", flush=True)
    print(flush=True)
    return np.concatenate(out) if out else np.zeros((0, enc.out_dim), np.float16)


def cmd_extract(args):
    import torch
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    f = ck.get("features", {})
    meta = {"encoder": f.get("encoder", "facebook/dinov2-small"), "grid": f.get("grid", 2),
            "crop": f.get("crop", 224), "margin": f.get("margin", 0.1)}
    rng = random.Random(args.seed)
    for src in args.sources:
        raw = os.path.join(args.data, "raw", src)
        rows = read_jsonl(os.path.join(raw, "index.jsonl"))
        if not rows:
            print(f"[extract] {src}: nothing fetched", flush=True)
            continue
        cache = os.path.join(args.data, "feat", src, "crops.npz")
        stamp = dict(meta, crops=len(rows), first=rows[0]["file"], last=rows[-1]["file"])   # a new fetch: anew
        if os.path.exists(cache) and json.load(open(cache + ".json")) in (stamp, meta) and \
                len(np.load(cache)["feats"]) == len(rows):
            feats = np.load(cache)["feats"]
        else:
            print(f"[extract] {src}: DINOv2 on {len(rows)} crops", flush=True)
            feats = encode_all([os.path.join(raw, r["file"]) for r in rows], meta,
                               ready=[r.get("ready", False) for r in rows])
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, feats=feats)
            json.dump(stamp, open(cache + ".json", "w"))
        for k, r in enumerate(rows):
            r["k"] = k
        cows = sorted({r["cow"] for r in rows})
        if any("split" in r for r in rows):            # mmcows: cows 11-16 (no neck tag) are its held-out cows
            val = {r["cow"] for r in rows if r.get("split") == "val"}
        else:
            val = set(random.Random(args.seed).sample(cows, max(1, round(len(cows) * args.val_share))))
        bursts = make_bursts(rows, rng, val, tracklets=src == "multicam")
        write_bursts(os.path.join(args.data, "feat", src), bursts, src, dict(meta, dim=int(feats.shape[1])))
        for part in ("train", "val"):                     # an older extract's stretched copies (GBs)
            old = os.path.join(args.data, "feat", src, part)
            if os.path.isdir(old):
                import shutil
                shutil.rmtree(old)
        n = collections.Counter(b["split"] for b in bursts)
        print(f"[extract] {src}: {len(cows)} cows ({len(val)} held out), {n['train']} train / {n['val']} val "
              f"bursts", flush=True)


# ------------------------------------------------------------------ train

class Bursts:
    """One part (train / val) of the sets: every crop's vector once (crops.npz), bursts as lists of crops."""

    def __init__(self, data, sources, part):
        self.feats, self.bursts = {}, []
        for src in sources:
            folder = os.path.join(data, "feat", src)
            rows = read_jsonl(os.path.join(folder, "bursts.jsonl"))
            if not rows:
                continue
            self.feats[src] = np.load(os.path.join(folder, "crops.npz"))["feats"]
            self.bursts += [dict(r, src=src) for r in rows if r["split"] == part]
        names = {}
        self.identity = np.array([names.setdefault(b["identity"], len(names)) for b in self.bursts], np.int64)
        self.dim = next(iter(self.feats.values())).shape[1] if self.feats else 0

    def frames(self, i):
        b = self.bursts[i]
        return self.feats[b["src"]][b["frames"]]

    def whole(self, i):
        """The burst stretched to the barn's 175 frames."""
        f = self.frames(i)
        return f[stretch(len(f))]


def sampler(data, P, K, rng):
    """P bursts a step: P // K cows with two or more bursts, K bursts of each (other days, other cameras)."""
    groups = collections.defaultdict(list)
    for i, g in enumerate(data.identity.tolist()):
        groups[g].append(i)
    keys = [g for g, v in groups.items() if len(v) > 1]
    per = max(2, P // K)
    while True:
        rng.shuffle(keys)
        for i in range(0, max(1, len(keys) - per + 1), per):
            yield [j for g in keys[i:i + per] for j in rng.sample(groups[g], min(K, len(groups[g])))]


def views(data, ids, rng, crop_s, single_p, dev, drop=0.15, block=0.3):
    """A training view of each burst: a random 2-5.6 s of it (at the barn's 25 fps), frames dropped,
    sometimes a stretch hidden, single_p of them cut to one frame -> (x, t, valid) padded."""
    import torch
    xs, vs = [], []
    for i in ids:
        f = data.frames(i)
        idx = stretch(len(f))
        n = int(round(rng.uniform(crop_s, 7.0 * 0.8) * FPS))
        s = rng.randint(0, SLOTS - n)
        v = np.array([rng.random() > drop for _ in range(n)])
        if rng.random() < block:
            k = rng.randint(n // 8, n // 3)
            j = rng.randint(0, n - k)
            v[j:j + k] = False
        if rng.random() < single_p:
            v[:] = False
        if not v.any():
            v[rng.randrange(n)] = True
        xs.append(f[idx[s:s + n]])
        vs.append(v)
    n = max(len(v) for v in vs)
    x = np.zeros((len(xs), n, data.dim), np.float32)
    m = np.zeros((len(xs), n), bool)
    for r, (a, v) in enumerate(zip(xs, vs)):
        x[r, :len(a)], m[r, :len(v)] = a, v
    t = torch.arange(n, device=dev).float().div(FPS).expand(len(xs), -1)
    return torch.from_numpy(x).to(dev), t, torch.from_numpy(m).to(dev)


def embed(model, data, single=False, batch=128):
    """Every burst whole (or its middle crop alone) -> fingerprints and the quality summary."""
    import torch
    from common import device
    dev = device()
    fps, qmax, qmean = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data.bursts), batch):
            ids = range(i, min(i + batch, len(data.bursts)))
            if single:
                x = np.stack([data.frames(k)[len(data.frames(k)) // 2][None] for k in ids])
            else:
                x = np.stack([data.whole(k) for k in ids])
            x = torch.from_numpy(x).float().to(dev)
            t = torch.arange(x.shape[1], device=dev).float().div(FPS).expand(len(x), -1)
            v = torch.ones(x.shape[:2], dtype=torch.bool, device=dev)
            o = model(x, t, v)
            fps.append(o["fingerprint"].float().cpu().numpy())
            qmax.append(o["weights"].max(1).values.float().cpu().numpy())
            qmean.append(o["quality"].float().mean(1).cpu().numpy())
    return np.concatenate(fps), np.concatenate(qmax), np.concatenate(qmean)


def reid_scores(split, fp, qfp=None, qmax=None, qmean=None):
    """Each burst (or its single frame: qfp) against the bursts of OTHER days (and, on its day, of other
    cameras): is the nearest one the same cow? -> metrics, and the rows the NaN model is fitted on."""
    qfp = fp if qfp is None else qfp
    ident = np.array([b["identity"] for b in split.bursts])
    day = np.array([b.get("day", "") for b in split.bursts])
    cam = np.array([b.get("cam", "") for b in split.bursts])
    src = np.array([i.split(":")[0] for i in ident])
    sim = qfp @ fp.T
    res, rows = {}, []
    for scope, allowed in (("other_day", day[None, :] != day[:, None]),
                           ("other_camera", (day[None, :] == day[:, None]) & (cam[None, :] != cam[:, None]))):
        allowed = allowed & (src[None, :] == src[:, None])       # within one set: its cows are its gallery
        s = np.where(allowed, sim, -2.0)
        askable = ((ident[None, :] == ident[:, None]) & allowed).any(1)
        if not askable.any():
            continue
        best = s.argmax(1)
        hit = ident[best] == ident
        res[f"{scope}_top1"] = float(hit[askable].mean())
        res[f"{scope}_queries"] = int(askable.sum())
        for s_ in sorted(set(src)):
            m = askable & (src == s_)
            if m.any():
                res[f"{scope}_top1_{s_}"] = float(hit[m].mean())
        if scope == "other_day":
            for q in np.where(askable)[0]:
                b = best[q]
                other = s[q][ident != ident[b]]
                second = float(other.max()) if other.size and other.max() > -2 else -1.0
                rows.append({"sim": float(s[q, b]), "margin": float(s[q, b]) - second, "correct": int(hit[q]),
                             "quality_max": float(qmax[q]) if qmax is not None else 0.0,
                             "quality_mean": float(qmean[q]) if qmean is not None else 0.0, "area": 0.05})
    return res, rows


def evaluate(model, split):
    fp, qmax, qmean = embed(model, split)
    m, rows = reid_scores(split, fp, qmax=qmax, qmean=qmean)
    one, _, _ = embed(model, split, single=True)
    m1, _ = reid_scores(split, fp, qfp=one)
    m.update({f"frame_{k}": v for k, v in m1.items()})
    parts = [m.get("other_day_top1", 0.0), m.get("frame_other_day_top1", 0.0)]
    if "other_camera_top1_mmcows" in m:        # MmCows' held-out cows 11-16 from another camera: the barn's view
        parts += [m["other_camera_top1_mmcows"], m.get("frame_other_camera_top1_mmcows", 0.0)]
    m["score"] = float(np.mean(parts))
    return m, rows


def mm_line(m, before=None):
    """MmCows' held-out cows from another camera, for the log ('' without mmcows)."""
    if "other_camera_top1_mmcows" not in m:
        return ""
    was = lambda k: f" (was {before.get(k, 0):.1%})" if before else ""
    return (f"; MmCows cows 11-16 another camera {m['other_camera_top1_mmcows']:.1%}"
            f"{was('other_camera_top1_mmcows')}, one frame {m.get('frame_other_camera_top1_mmcows', 0):.1%}"
            f"{was('frame_other_camera_top1_mmcows')}")


def fit_abstain(rows, max_error, out_path):
    """The NaN model on these fingerprints: abstain.py's logistic over (sim, margin, quality, size)."""
    import abstain
    X = abstain.feature_matrix(rows)
    y = np.array([r["correct"] for r in rows], float)
    m = abstain.fit_logistic(X, y)
    p = abstain.predict(m, X)
    budgets = sorted({max_error, 0.01, 0.02, 0.05})
    th = {e: abstain.cutoff(p, y, e) for e in budgets}
    same = np.array([r["sim"] for r in rows if r["correct"]])
    out = {"features": list(abstain.FEATURES), **m, "max_error": max_error, "threshold": th[max_error],
           "scope": "all", "dev_queries": len(rows), "dev_top1": float(y.mean()),
           "same_cow_sim_p05": float(np.percentile(same, 5)) if len(same) else None,
           "fitted_on": "reid.py val cows, other days",
           "dev": abstain.coverage_table(p, y, [(f"{e:.0%}", th[e]) for e in budgets])}
    write_json(out_path, out)
    return out


def cmd_train(args):
    import torch
    from common import device
    from model import HerdModel, supcon
    base, ck = HerdModel.load(args.model, map_location=device())
    tr = Bursts(args.data, args.sources, "train")
    va = Bursts(args.data, args.sources, "val")
    if not tr.bursts or not va.bursts:
        sys.exit("no features: reid.py extract first")
    if tr.dim != base.in_dim:
        sys.exit(f"features of {tr.dim} dims, the model takes {base.in_dim}: extract with this model")
    print(f"[train] {len(tr.bursts)} bursts of {len(set(tr.identity.tolist()))} cows; val {len(va.bursts)} bursts "
          f"of {len({b['identity'] for b in va.bursts})} unseen cows", flush=True)
    dev = device()
    model = copy.deepcopy(base.temporal).to(dev)
    if args.init:                              # fine-tune an identity model already trained (reid.py train)
        ck0 = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck0["state"])
        print(f"[train] from {args.init} (trained on {', '.join(ck0.get('sources', []))})", flush=True)
    m0, _ = evaluate(model, va)
    print(f"[train] before: same cow another day {m0.get('other_day_top1', 0):.1%}, another camera "
          f"{m0.get('other_camera_top1', 0):.1%}, one frame vs another day {m0.get('frame_other_day_top1', 0):.1%}"
          f"{mm_line(m0)}", flush=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, len(tr.bursts) // args.P)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.05)
    batches = sampler(tr, args.P, args.K, rng)
    os.makedirs(args.out, exist_ok=True)
    best, best_state, best_m, since = -1.0, None, None, 0
    hist = open(os.path.join(args.out, "history.jsonl"), "w", encoding="utf-8")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(steps):
            ids = next(batches)
            # some views one frame only: mct.py follows a cow by one frame a second
            x, t, v = views(tr, ids, rng, args.crop_s, args.single_p, dev)
            o = model(x, t, v)
            loss = supcon(o["fingerprint"], torch.as_tensor(tr.identity[ids], device=dev), args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        m, _ = evaluate(model, va)
        hist.write(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses)), **m}) + "\n")
        hist.flush()
        print(f"[train] epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.3f}  val: another day "
              f"{m.get('other_day_top1', 0):.1%}, another camera {m.get('other_camera_top1', 0):.1%}, one frame "
              f"{m.get('frame_other_day_top1', 0):.1%}{mm_line(m)}", flush=True)
        since += 1
        if m["score"] > best:
            best, best_m, since = m["score"], dict(m, epoch=epoch + 1), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if since >= args.patience:
            break
    hist.close()
    model.load_state_dict(best_state)
    _, rows = evaluate(model, va)
    ab = fit_abstain(rows, args.max_error, os.path.join(args.out, "abstain.json"))
    torch.save({"in_dim": base.in_dim, "kw": {k: v for k, v in base.kw.items() if k != "frame_id"},
                "state": {k: v.cpu() for k, v in model.state_dict().items()}, "features": ck.get("features", {}),
                "base": os.path.abspath(args.model), "init": os.path.abspath(args.init) if args.init else "",
                "sources": sorted(set(args.sources) | set(ck0.get("sources", []) if args.init else [])),
                "eval": best_m, "before": m0},
               os.path.join(args.out, "identity.pt"))
    write_json(os.path.join(args.out, "eval.json"), {"before": m0, "after": best_m,
                                                     "abstain": {k: ab[k] for k in ("threshold", "dev_top1", "dev")}})
    print(f"[train] best epoch {best_m['epoch']}: another day {best_m.get('other_day_top1', 0):.1%} (was "
          f"{m0.get('other_day_top1', 0):.1%}), another camera {best_m.get('other_camera_top1', 0):.1%} (was "
          f"{m0.get('other_camera_top1', 0):.1%}), one frame {best_m.get('frame_other_day_top1', 0):.1%} (was "
          f"{m0.get('frame_other_day_top1', 0):.1%}){mm_line(best_m, m0)}", flush=True)
    for r in ab["dev"]:
        e = r["error_when_answered"]
        print(f"  NaN model: budget {r['budget']:>4}  answers {r['answered']:.1%}  wrong when answering "
              f"{'-' if e is None else f'{e:.1%}'}")
    print(f"-> {os.path.join(args.out, 'identity.pt')} (+ abstain.json): [model] identity in barn.toml")


def load_identity(path, features_meta, dev):
    """identity.pt -> (TemporalModel on dev, its abstain.json or None); checks it reads the same vectors."""
    import torch
    from model import TemporalModel
    ck = torch.load(path, map_location=dev, weights_only=False)
    want = {k: ck.get("features", {}).get(k) for k in ("encoder", "grid", "crop", "margin")}
    have = {k: features_meta.get(k) for k in want}
    if want != have:
        raise SystemExit(f"{path} reads frame vectors {want}, the herd model's are {have}")
    m = TemporalModel(ck["in_dim"], **ck["kw"]).to(dev).eval()
    m.load_state_dict(ck["state"])
    ab = os.path.join(os.path.dirname(os.path.abspath(path)), "abstain.json")
    return m, (json.load(open(ab, encoding="utf-8")) if os.path.exists(ab) else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    lb = sub.add_parser("label", help="(GPU) MmCows' frames -> crops with cow numbers (human labels, neck tags)")
    e = sub.add_parser("extract")
    t = sub.add_parser("train")
    for q in (f, e, t):
        q.add_argument("--data", required=True)
        q.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    f.add_argument("--workers", type=int, default=48)
    f.add_argument("--max-per-source", type=int, default=0, help="a quick look: only this many crops a set")
    f.add_argument("--keep-zips", action="store_true", help="keep the zips downloaded whole (Zenodo's)")
    f.add_argument("--mm-uwb-days", nargs="*", default=["0722", "0728", "0801"],
                   help="MmCows days labelled by the neck tags (7/25 is the human-labelled one)")
    f.add_argument("--mm-uwb-every", type=int, default=30, help="seconds between those days' frames (15, 30, 60...)")
    f.add_argument("--mm-label-every", type=int, default=60, help="seconds between 7/25's labelled frames")
    f.add_argument("--mm-max-side", type=int, default=1920)
    lb.add_argument("--data", required=True)
    lb.add_argument("--model", required=True, help="the herd model (crop size and margin)")
    lb.add_argument("--detector", required=True)
    lb.add_argument("--batch", type=int, default=8)
    lb.add_argument("--mm-share", type=float, default=0.9, help="a track's seconds its tag must win")
    lb.add_argument("--force", action="store_true", help="label again (else each day and camera once)")
    lb.add_argument("--mm-days", nargs="*", default=None, help="only these days (the pod labels a day while the "
                    "next one downloads)")
    for q in (e, t):
        q.add_argument("--model", required=True, help="the herd model (its encoder settings; train starts from it)")
        q.add_argument("--seed", type=int, default=0)
    e.add_argument("--val-share", type=float, default=1 / 6, help="cows held out of training, per set")
    t.add_argument("--out", default=None, help="default DATA/id_model")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--P", type=int, default=64, help="bursts a step")
    t.add_argument("--K", type=int, default=4, help="bursts of each cow in a step")
    t.add_argument("--lr", type=float, default=None, help="default 1e-4, 3e-5 with --init")
    t.add_argument("--init", default=None, help="an identity.pt to fine-tune (else the herd model's transformer)")
    t.add_argument("--temperature", type=float, default=0.1)
    t.add_argument("--crop-s", type=float, default=2.0, help="the shortest view (s); up to 5.6 s")
    t.add_argument("--single-p", type=float, default=0.25, help="share of views cut to one frame")
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--max-error", type=float, default=0.01, help="the NaN model's error budget")
    args = p.parse_args(argv)
    if args.cmd == "train" and args.out is None:
        args.out = os.path.join(args.data, "id_model")
    if args.cmd == "train" and args.lr is None:
        args.lr = 3e-5 if args.init else 1e-4
    {"fetch": cmd_fetch, "label": cmd_label, "extract": cmd_extract, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
