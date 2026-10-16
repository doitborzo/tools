#!/usr/bin/env python3
"""How well one track per cow holds across cameras (mct.py), on MmCows' four synchronised cameras.

MmCows (Vu et al., Purdue NEIS, huggingface.co/datasets/neis-lab/mmcows, CC BY-NC-SA 4.0): one pen
of 16 cows seen by four high cameras from its corners, a frame a second on 7/25 12:00-15:00
(1s_interval_images_3hr, ~7.5 GB a camera), and every 15 s each camera's boxes with the cow's
number (visual_data.zip labels/combined). The other pens are blacked out in the frames.

    fetch   the frames of --minutes from --start (every --step s), shrunk to 1920 px, and the labels
            of that window; zips are read in place (HTTP Range), nothing else is downloaded
    embed   (GPU) every frame of every camera: the detector -> crops -> DINOv2 -> frame heads:
            boxes, the frame id (model.frame_fingerprint: the herd transformer on one frame) and
            DINOv2's CLS token, per camera
    score   (CPU) the trackers on those boxes, a second at a time, camera after camera:
              iou        the system without mct: an IoU tracker per camera (pipeline.Tracker)
              cam        mct.GlobalTracker without neighbours: per camera, IoU + frame id, reacquire
              mct        GlobalTracker with every camera a neighbour of every other (they overlap)
            each with the frame id and with DINOv2's CLS (the baseline without training), and for
            mct a sweep of handover_min_sim; scored on the labelled seconds against the cows' numbers:
              idf1_camera   IDF1 within each camera (a cow followed by one track per camera)
              idf1_global   IDF1 over all cameras at once (one track per cow across cameras)
              tracks_per_cow, id_switches   how often a cow's track breaks (per camera)
              cross_cam_same_track   a cow seen by two cameras at once: share of those pairs on one track
              false_merges  two different cows on one track at the same second (any cameras)
              false_cannot_link  track pairs seen together in one camera that are one cow
            and, for the frame id, the nightly grouping of new cows (gallery.enroll) on 7 s
            pseudo-bursts of the tracks, with and without cannot_link: how many cows it makes of
            the 16, how many ids one cow is split into, how pure a new cow is

    python herd/mct_eval.py fetch --data /workspace/mct --minutes 60
    python herd/mct_eval.py embed --data /workspace/mct --model /workspace/herd/run9/model.pt --detector <best/>
    python herd/mct_eval.py score --data /workspace/mct          # -> mct_eval.json + a table

Nothing is trained here and no other dataset is needed: the frame id is the herd model's own
transformer run on a burst of one frame.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import DEFAULTS, POSTURES, floor_mapper, iou, write_json  # noqa: E402

HF = "https://huggingface.co/datasets/neis-lab/mmcows/resolve/main"
DAY = "0725"
CAMS = ("cam_1", "cam_2", "cam_3", "cam_4")
DAY_START = 1690304400                       # 7/25 12:00:00 local, the first frame of the 3 h set
TAGGED = set(range(1, 11))                   # the cows with a UWB neck tag (sensor_data T01-T10)


# ------------------------------------------------------------------ fetch

def frames_zip(cam):
    return f"{HF}/1s_interval_images_3hr/{DAY}_{cam}_1200-1500.zip"


def _ts(name):
    m = re.search(r"(\d{10})_[\d-]+\.(jpg|txt)$", name)
    return int(m.group(1)) if m else None


def cmd_fetch(args):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cowbench"))
    from extra_det import RemoteZip, shrink
    t0 = DAY_START + int(args.start * 60)
    t1 = t0 + int(args.minutes * 60)
    os.makedirs(args.data, exist_ok=True)
    print(f"[fetch] 7/25 {12 + args.start // 60:.0f}:{args.start % 60:02.0f} + {args.minutes} min, a frame every "
          f"{args.step} s, cameras {', '.join(args.cams)}", flush=True)
    vz = RemoteZip(f"{HF}/visual_data.zip")
    names = vz.names()
    labels = [n for n in names if f"/labels/combined/{DAY}/" in n and n.endswith(".txt")
              and t0 <= (_ts(n) or 0) < t1 and n.split("/")[-2] in args.cams]
    extra = [n for n in names if f"/proj_mat/{DAY}/" in n or ("/visual_location/" in n and n.endswith(".csv"))]
    for n in labels + extra:
        sub = "labels/" + n.split("/")[-2] if n in labels else ("proj_mat" if "/proj_mat/" in n else "visual_location")
        out = os.path.join(args.data, sub, os.path.basename(n))
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(vz.read(n))
    print(f"[fetch] {len(labels)} label files, projection matrices and the cows' positions", flush=True)
    meta = {"day": DAY, "start_ts": t0, "end_ts": t1, "step": args.step, "cams": list(args.cams),
            "max_side": args.max_side, "source": HF, "licence": "CC BY-NC-SA 4.0"}
    for cam in args.cams:
        z = RemoteZip(frames_zip(cam))
        # every step-th second, and every labelled one whatever the step (the labels sit at :11, :26, ...)
        marked = {_ts(n) for n in labels if n.split("/")[-2] == cam}
        want = [n for n in z.names() if (_ts(n) or 0) >= t0 and _ts(n) < t1
                and ((_ts(n) - t0) % args.step == 0 or _ts(n) in marked)]
        folder = os.path.join(args.data, "frames", cam)
        os.makedirs(folder, exist_ok=True)
        todo = [n for n in want if not os.path.exists(os.path.join(folder, f"{_ts(n)}.jpg"))]
        start, done = time.time(), 0

        def one(n):
            data, size = shrink(z.read(n), args.max_side)
            path = os.path.join(folder, f"{_ts(n)}.jpg")
            open(path + ".part", "wb").write(data)
            os.replace(path + ".part", path)
            return size

        with cf.ThreadPoolExecutor(args.workers) as ex:
            for size in ex.map(one, todo):
                done += 1
                meta["frame_size"] = size
                if done % 200 == 0 or done == len(todo):
                    el = time.time() - start
                    print(f"\r  {cam}: {done}/{len(todo)} frames  {el / 60:.1f} min  "
                          f"~{el / done * (len(todo) - done) / 60:.0f} min left", end="", flush=True)
        print(f"\n[fetch] {cam}: {len(want)} frames ({len(want) - len(todo)} were there)", flush=True)
    write_json(os.path.join(args.data, "meta.json"), meta)


def read_labels(data, cams):
    """{cam: {ts: [(cow, [x1, y1, x2, y2])]}} from the YOLO-style 'id cx cy w h' lines."""
    out = {}
    for cam in cams:
        folder = os.path.join(data, "labels", cam)
        per = {}
        for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            rows = []
            for line in open(os.path.join(folder, f), encoding="utf-8"):
                p = line.split()
                if len(p) >= 5:
                    c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                    rows.append((c, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
            per[_ts(f)] = rows
        out[cam] = per
    return out


# ------------------------------------------------------------------ embed

def cmd_embed(args):
    import pipeline
    from PIL import Image
    meta = json.load(open(os.path.join(args.data, "meta.json"), encoding="utf-8"))
    cfg = copy.deepcopy(DEFAULTS)
    cfg["model"]["checkpoint"] = args.model
    cfg["detector"]["weights"] = args.detector
    if args.threshold is not None:
        cfg["detector"]["threshold"] = args.threshold
    if args.identity:
        cfg["model"]["identity"] = args.identity
    models = pipeline.Models(cfg)
    hidden = models.encoder.hidden
    source = ("the identity model (reid.py) on one frame" if models.id_model is not None else
              "the trained head" if models.model.frame_id else "the herd transformer on one frame")
    print(f"[embed] detector threshold {getattr(models.detector, 'threshold', '?')}; frame id: {source}", flush=True)
    from reid import identity_thresholds
    stamp = {"frame_id": source, "identity": os.path.abspath(args.identity) if args.identity else "",
             "thresholds": identity_thresholds(args.identity) if args.identity else {}}
    os.makedirs(os.path.join(args.data, "emb"), exist_ok=True)
    for cam in meta["cams"]:
        out = os.path.join(args.data, "emb", f"{cam}.npz")
        if os.path.exists(out) and not args.force and "lying" in np.load(out).files and \
                json.loads(str(np.load(out).get("stamp", "{}"))) == stamp:
            print(f"[embed] {cam}: there already", flush=True)
            continue
        folder = os.path.join(args.data, "frames", cam)
        tss = sorted(int(f[:-4]) for f in os.listdir(folder) if f.endswith(".jpg"))
        load = lambda ts: np.asarray(Image.open(os.path.join(folder, f"{ts}.jpg")).convert("RGB"))
        T, N, B, C, F, L = [], [], [], [], [], []
        start = time.time()
        with cf.ThreadPoolExecutor(4) as ex:
            for k in range(0, len(tss), args.batch):
                chunk = tss[k:k + args.batch]
                imgs = list(ex.map(load, chunk))
                dets = models.detect_many(imgs, batch=args.batch)
                pairs = [(img, b) for img, bs in zip(imgs, dets) for b in bs]
                boxes = [b for bs in dets for b in bs]
                if pairs:
                    feats = models.encode(pipeline.crops_of(pairs, models.crop_size, models.margin))
                    pp, _, ids = models.frame_heads(feats, boxes, urgent=False, with_id=True)
                    F.append(np.asarray(ids, np.float16))
                    L.append(pp[:, POSTURES.index("lying")].astype(np.float32))
                    C.append(feats[:, :hidden].float().cpu().numpy().astype(np.float16))
                    B.append(np.asarray(boxes, np.float32))
                T += chunk
                N += [len(bs) for bs in dets]
                if (k // args.batch) % 25 == 0:
                    el = time.time() - start
                    done = k + len(chunk)
                    print(f"\r  {cam}: {done}/{len(tss)} frames, {sum(N)} boxes  {el / 60:.1f} min  "
                          f"~{el / done * (len(tss) - done) / 60:.0f} min left", end="", flush=True)
        np.savez(out, ts=np.array(T, np.int64), n=np.array(N, np.int32),
                 boxes=np.concatenate(B) if B else np.zeros((0, 4), np.float32),
                 cls=np.concatenate(C) if C else np.zeros((0, hidden), np.float16),
                 fid=np.concatenate(F) if F else np.zeros((0, 0), np.float16),
                 lying=np.concatenate(L) if L else np.zeros((0,), np.float32), stamp=json.dumps(stamp))
        print(f"\n[embed] {cam}: {len(T)} frames, {sum(N)} boxes ({sum(N) / max(1, len(T)):.1f} a frame) -> {out}",
              flush=True)


# ------------------------------------------------------------------ score

def load_emb(data, cams):
    """{cam: {ts: [(box, fid or None, cls, p_lying or None)]}}"""
    out = {}
    for cam in cams:
        z = np.load(os.path.join(data, "emb", f"{cam}.npz"))
        boxes, cls, fid = z["boxes"], z["cls"].astype(np.float32), z["fid"].astype(np.float32)
        lying = z["lying"] if "lying" in z.files else None
        per, i = {}, 0
        for ts, n in zip(z["ts"].tolist(), z["n"].tolist()):
            per[ts] = [(boxes[j].tolist(), fid[j] if fid.size else None, cls[j],
                        None if lying is None else float(lying[j])) for j in range(i, i + n)]
            i += n
        out[cam] = per
    return out


def hull(points):
    """Convex hull (monotone chain) of [(x, y)] -> polygon [[x, y], ...]."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return [list(p) for p in pts]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def pen_masks(gt, grow=0.05):
    """Per camera, the pen as its labels show it: the hull of every labelled cow's box centre, grown
    by `grow` around its middle. The 1 s frames show the neighbouring pens too (their cows are not
    labelled) - in the barn a camera's mask leaves out what is not its floor in the same way."""
    out = {}
    for cam, per in gt.items():
        pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for rows in per.values() for _, b in rows]
        h = hull(pts)
        if len(h) >= 3:
            c = np.mean(h, 0)
            h = [list(c + (np.array(p) - c) * (1 + grow)) for p in h]
        out[cam] = h
    return out


def in_pen(emb, masks):
    """Only the detections whose centre is in the camera's pen."""
    from common import in_mask
    kept = dropped = 0
    out = {}
    for cam, per in emb.items():
        m = masks.get(cam) if len(masks.get(cam) or []) >= 3 else []
        out[cam] = {}
        for ts, v in per.items():
            keep = [x for x in v if in_mask(x[0], m)]
            kept += len(keep)
            dropped += len(v) - len(keep)
            out[cam][ts] = keep
    return out, dropped / max(1, kept + dropped)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def prepare(emb, kind):
    """The vectors a tracker follows: the frame id as it is, or DINOv2's CLS centred on its mean
    (the pen's look common to every cow taken out) and normalised."""
    if kind == "fid":
        return {c: {t: [(b, f, ly) for b, f, _, ly in v] for t, v in per.items()} for c, per in emb.items()}
    mean = np.mean([x for per in emb.values() for v in per.values() for _, _, x, _ in v], 0)
    return {c: {t: [(b, _unit(x - mean), ly) for b, _, x, ly in v] for t, v in per.items()} for c, per in emb.items()}


PROJ_SIZE = (4480, 2800)                      # MmCows' frames, the pixels its projection matrices map to


def proj_mats(data, cams):
    """{cam: 3x4 projection matrix (world cm -> pixels of the 4480x2800 frame)} from proj_mat/*.xml."""
    out = {}
    for cam in cams:
        path = os.path.join(data, "proj_mat", f"proj_mat_cam{cam.split('_')[-1]}.xml")
        if os.path.exists(path):
            m = re.search(r"<projection_matrix.*?<data>(.*?)</data>", open(path, encoding="utf-8").read(), re.S)
            out[cam] = [float(x) for x in m.group(1).split()]
    return out


def look_check(gt_map, dets):
    """By look alone: a labelled cow seen by one camera, against the cows another camera sees at the same
    second - is the nearest frame id hers? (chance: one in the number of cows there)."""
    at = collections.defaultdict(list)
    for (cam, ts, j), cow in gt_map.items():
        e = dets[cam][ts][j][1]
        if e is not None:
            at[ts].append((cam, cow, np.asarray(e, np.float32)))
    hit = n = 0
    chance = []
    group = collections.Counter()
    for v in at.values():
        for c1, w1, e1 in v:
            for c2 in {c for c, _, _ in v if c != c1}:
                cands = [(w, e) for c, w, e in v if c == c2]
                if w1 not in [w for w, _ in cands]:
                    continue
                best = max(cands, key=lambda x: float(x[1] @ e1))[0]
                hit += best == w1
                n += 1
                chance.append(1 / len(cands))
                g = "tagged" if int(w1) in TAGGED else "untagged"
                group[g, "n"] += 1
                group[g, "hit"] += best == w1
    out = {"look_cross_cam_top1": hit / n if n else None, "chance": float(np.mean(chance)) if chance else None,
           "queries": n}
    # cows 1-10 wear UWB tags: reid.py may train on them (other days); cows 11-16 it never sees
    for g in ("tagged", "untagged"):
        if group[g, "n"]:
            out[f"look_cross_cam_top1_{g}"] = group[g, "hit"] / group[g, "n"]
    return out


def floor_check(gt_map, dets, mats):
    """How well the floor points agree: on the labelled seconds, the same cow seen by two cameras
    (distance between their floor points) against two different cows (cm)."""
    import itertools
    mappers = {c: floor_mapper({"proj_mat": m, "proj_size": PROJ_SIZE}, DEFAULTS["mct"]) for c, m in mats.items()}
    at = collections.defaultdict(list)
    for (cam, ts, j), cow in gt_map.items():
        if cam in mappers:
            b, _, ly = dets[cam][ts][j]
            at[ts].append((cam, cow, mappers[cam](b, 0.5 if ly is None else ly)))
    same, other = [], []
    for v in at.values():
        for (c1, w1, p1), (c2, w2, p2) in itertools.combinations(v, 2):
            if c1 != c2:
                (same if w1 == w2 else other).append(float(np.hypot(p1[0] - p2[0], p1[1] - p2[1])))
    if not same:
        return None
    same, other = np.array(same), np.array(other)
    return {"same_cow_median_cm": float(np.median(same)), "same_cow_p90_cm": float(np.percentile(same, 90)),
            "same_cow_within_100cm": float(np.mean(same < 100)),
            "other_cows_p5_cm": float(np.percentile(other, 5)) if other.size else None,
            "other_cows_within_100cm": float(np.mean(other < 100)) if other.size else None}


def run_tracker(dets, kind, mct_over=None, mats=None):
    """Every second, camera after camera -> ({(cam, ts, j): track}, cannot_link pairs, tracker stats).
    kind: iou (pipeline.Tracker per camera), cam (GlobalTracker, no neighbours), mct (neighbours by
    look), geo (passed on by the floor point: mats, the cameras' projection matrices).
    cannot_link: same camera with evidence (mct.CoSeen, as the pipeline), across cameras from the
    tracker (apart_dist)."""
    import mct
    import pipeline
    cams = sorted(dets)
    seconds = sorted({t for per in dets.values() for t in per})
    got, links = {}, set()
    cfg = copy.deepcopy(DEFAULTS)
    cfg["cameras"] = [{"id": c, "mask": []} for c in cams]
    cfg["mct"].update(mct_over or {})
    if kind == "mct":
        cfg["mct"]["neighbours"] = [[a, b] for i, a in enumerate(cams) for b in cams[i + 1:]]
    if kind in ("geo", "fused", "link", "link-final"):
        for c in cfg["cameras"]:
            if c["id"] in (mats or {}):
                c.update(proj_mat=mats[c["id"]], proj_size=list(PROJ_SIZE))
    co = mct.CoSeen(cfg["identity"]["cannot_link_min_s"], cfg["identity"]["cannot_link_max_iou"])
    trackers = ({c: pipeline.Tracker(c, cfg["tracker"]["iou"], cfg["tracker"]["max_age_s"]) for c in cams}
                if kind == "iou" else None)
    if kind in ("fused", "link", "link-final"):
        return _run_fused(dets, cfg, co, kind)
    glob = mct.GlobalTracker(cfg, apart=lambda pairs, ts: links.update((min(a, b), max(a, b)) for a, b in pairs)) \
        if kind != "iou" else None
    for ts in seconds:
        for cam in cams:
            ds = dets[cam].get(ts)
            if ds is None:
                continue
            if trackers is not None:
                # the IoU tracker returns its tracks in its own order: map back by the box
                out = trackers[cam].update(float(ts), [d[0] for d in ds])
                left = list(range(len(ds)))
                ids = [None] * len(ds)
                for tid, b in out:
                    j = next(j for j in left if ds[j][0] is b)
                    left.remove(j)
                    ids[j] = tid
            else:
                ids = glob.update(cam, float(ts), [{"box": b, "emb": e, "p_lying": 0.5 if ly is None else ly}
                                                   for b, e, ly in ds])
            for j, g in enumerate(ids):
                got[(cam, ts, j)] = g
            links.update(co.add(ids, [d[0] for d in ds]))
    if glob is not None and glob.alias:          # cows taken back late: as the reports see them
        got = {k: glob.root(g) for k, g in got.items()}
        links = {(min(glob.root(a), glob.root(b)), max(glob.root(a), glob.root(b))) for a, b in links
                 if glob.root(a) != glob.root(b)}
    return got, links, (dict(glob.stats) if glob else {})


def _run_fused(dets, cfg, co, kind="fused"):
    """All cameras of a second at once: on one floor plan (mct.FloorTracker), or each camera's tracks
    joined by floor evidence (mct.TrackletLinker; link-final: as the reports see them, every track
    under the one it was finally joined to)."""
    import mct
    ft = mct.FloorTracker(cfg) if kind == "fused" else mct.TrackletLinker(cfg)
    got, links = {}, set()
    cams = sorted(dets)
    for ts in sorted({t for per in dets.values() for t in per}):
        per = {c: [{"box": d[0], "emb": d[1], "p_lying": 0.5 if d[2] is None else d[2]} for d in dets[c][ts]]
               for c in cams if ts in dets[c]}
        out = ft.update_all(float(ts), per)
        for c, ids in out.items():
            for j, g in enumerate(ids):
                got[(c, ts, j)] = g
            links.update(co.add(ids, [d["box"] for d in per[c]]))
    if kind == "link-final":
        got = {k: ft.root(g) for k, g in got.items()}
        links = {(min(ft.root(a), ft.root(b)), max(ft.root(a), ft.root(b))) for a, b in links if ft.root(a) != ft.root(b)}
    return got, links, dict(ft.stats)


def floor_area(data, grow=150.0):
    """The pen's floor as MmCows' own cow positions cover it: their hull, grown by `grow` cm."""
    pos = [p for v in cow_positions(data).values() for p in v.values() if np.isfinite(p).all()]
    if len(pos) < 3:
        return []
    h = np.array(hull(pos), float)
    c = h.mean(0)
    d = h - c
    return (c + d * (1 + grow / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6))).tolist()


def match_gt(gt, dets, thr=0.5):
    """Labelled seconds: detections to annotated cows by IoU (best assignment) -> {(cam, ts, j): cow},
    and the counts for detector recall / precision there."""
    from mct import NEG, assign
    m, n_gt, n_det = {}, 0, 0
    for cam, per in gt.items():
        for ts, cows in per.items():
            ds = dets.get(cam, {}).get(ts)
            if ds is None:
                continue
            n_gt += len(cows)
            n_det += len(ds)
            S = np.full((len(cows), len(ds)), NEG)
            for i, (_, g) in enumerate(cows):
                for j, d in enumerate(ds):
                    v = iou(g, d[0])
                    if v >= thr:
                        S[i, j] = v
            for i, j in assign(S):
                m[(cam, ts, j)] = cows[i][0]
    return m, n_gt, n_det


def idf1(pairs, n_gt, n_pred):
    """pairs [(track, cow)] of matched detections -> IDF1 with the best one-to-one track <-> cow map."""
    from mct import assign
    if not pairs:
        return 0.0
    tracks = sorted({t for t, _ in pairs})
    cows = sorted({c for _, c in pairs})
    ti, ci = {t: k for k, t in enumerate(tracks)}, {c: k for k, c in enumerate(cows)}
    M = np.zeros((len(tracks), len(cows)))
    for t, c in pairs:
        M[ti[t], ci[c]] += 1
    idtp = sum(M[r, c] for r, c in assign(M))
    return float(2 * idtp / max(1, n_gt + n_pred))


def score_run(got, links, gt_map, gt, dets, n_gt, n_det):
    lab = {k: (got[k], c) for k, c in gt_map.items() if k in got}
    cams = sorted(gt)
    labelled = {(cam, ts) for cam, per in gt.items() for ts in per if ts in dets.get(cam, {})}
    n_pred = {cam: sum(len(dets[cam][ts]) for c2, ts in labelled if c2 == cam) for cam in cams}
    n_gtc = {cam: sum(len(gt[cam][ts]) for c2, ts in labelled if c2 == cam) for cam in cams}
    per_cam = [idf1([v for k, v in lab.items() if k[0] == cam], n_gtc[cam], n_pred[cam]) for cam in cams]
    r = {"idf1_camera": float(np.mean(per_cam)), "idf1_camera_each": dict(zip(cams, map(float, per_cam))),
         "idf1_global": idf1(list(lab.values()), n_gt, n_det)}
    # breaks: per camera, a cow's track over the labelled seconds
    seq = collections.defaultdict(list)
    for (cam, ts, _), (t, c) in sorted(lab.items()):
        seq[(cam, c)].append(t)
    r["id_switches"] = int(sum(sum(a != b for a, b in zip(s, s[1:])) for s in seq.values()))
    r["id_switches_per_cow_hour"] = r["id_switches"] / max(1, len(seq)) / max(1e-9, _hours(gt))
    tracks_of = collections.defaultdict(set)
    for t, c in lab.values():
        tracks_of[c].add(t)
    r["tracks_per_cow"] = float(np.mean([len(v) for v in tracks_of.values()])) if tracks_of else None
    # the same second across cameras
    at = collections.defaultdict(list)
    for (cam, ts, _), (t, c) in lab.items():
        at[ts].append((cam, t, c))
    same = pairs = merges = 0
    for v in at.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                (ca, ta, xa), (cb, tb, xb) = v[i], v[j]
                if xa == xb and ca != cb:
                    pairs += 1
                    same += ta == tb
                elif xa != xb and ta == tb:
                    merges += 1
    r["cross_cam_pairs"] = pairs
    r["cross_cam_same_track"] = same / pairs if pairs else None
    r["false_merges"] = merges
    # a track's cow: the majority of its labelled detections
    votes = collections.defaultdict(collections.Counter)
    for t, c in lab.values():
        votes[t][c] += 1
    major = {t: v.most_common(1)[0][0] for t, v in votes.items()}
    purity = sum(v.most_common(1)[0][1] for v in votes.values()) / max(1, sum(sum(v.values()) for v in votes.values()))
    r["track_purity"] = float(purity)
    known = [(a, b) for a, b in links if a in major and b in major]
    r["cannot_link_pairs"] = len(known)
    r["false_cannot_link"] = int(sum(major[a] == major[b] for a, b in known))
    return r, major


def _hours(gt):
    tss = [ts for per in gt.values() for ts in per]
    return (max(tss) - min(tss) + 15) / 3600 if tss else 0.0


def pseudo_bursts(got, dets, every=60, length=7, min_frames=5):
    """Per track, every `every` s: the mean frame id over `length` s -> a burst-like fingerprint."""
    obs = collections.defaultdict(list)
    for (cam, ts, j), t in got.items():
        e = dets[cam][ts][j][1]
        if e is not None:
            obs[t].append((ts, e))
    out = []
    for t, v in obs.items():
        v.sort(key=lambda x: x[0])
        tss = np.array([x[0] for x in v])
        E = np.stack([x[1] for x in v])
        start = tss[0]
        while start <= tss[-1]:
            sel = (tss >= start) & (tss < start + length)
            if sel.sum() >= min_frames:
                out.append({"ts": float(start), "track": t, "fp": _unit(E[sel].mean(0))})
            start += every
    return out


def enroll_eval(bursts, links, major, min_bursts, similarity, max_units=600):
    """gallery.enroll on the pseudo-bursts: the cows it makes, against the cows' numbers."""
    import gallery as gmod
    by_track = collections.Counter(b["track"] for b in bursts)
    keep = {t for t, _ in by_track.most_common(max_units)}
    bursts = [b for b in bursts if b["track"] in keep]
    res = {}
    for name, cl in (("with_cannot_link", links), ("without", set())):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["identity"].update(new_cow_min_bursts=min_bursts, new_cow_similarity=similarity)
        with tempfile.TemporaryDirectory() as d:
            g = gmod.Gallery(d, cfg)
            g.pool = [dict(b) for b in bursts]
            made = g.enroll(time.time(), {(min(a, b), max(a, b)) for a, b in cl})
            by_fp = {id(b["fp"]): b for b in bursts}       # a new cow's history keeps the pool's own arrays
            per_cow = collections.defaultdict(set)
            purities, sizes = [], []
            for cid in made:
                members = [by_fp[id(f)] for _, f in g.history[cid] if id(f) in by_fp]
                cows = collections.Counter(major.get(b["track"]) for b in members if major.get(b["track"]) is not None)
                if cows:
                    top, n = cows.most_common(1)[0]
                    purities.append(n / sum(cows.values()))
                    per_cow[top].add(cid)
                sizes.append(len(members))
            res[name] = {"cows_made": len(made), "true_cows": len(set(major.values())),
                         "ids_per_cow": float(np.mean([len(v) for v in per_cow.values()])) if per_cow else None,
                         "cows_without_id": len(set(major.values()) - set(per_cow)),
                         "purity": float(np.mean(purities)) if purities else None,
                         "pool_left": len(g.pool), "units": len(keep)}
    return res


def load_thresholds(data, cams):
    """The identity model's calibrated thresholds the embeddings were made with ({} for the herd model)."""
    for cam in cams:
        path = os.path.join(data, "emb", f"{cam}.npz")
        if os.path.exists(path):
            z = np.load(path)
            if "stamp" in z.files:
                return json.loads(str(z["stamp"])).get("thresholds", {}) or {}
    return {}


def variants(args, has_mats, cal=None):
    """(name, vector, kind, [mct] overrides) of every tracker scored; with an identity model's calibrated
    thresholds (cal), every fid tracker starts from them (late reacquire on), the sweeps still set theirs."""
    v = [("iou", "fid", "iou", {}),
         ("fid/cam", "fid", "cam", {"still_s": 0}),           # stage B as it was
         ("fid/still", "fid", "cam", {})]                      # + a cow lying still held by her place
    v += [(f"fid/still~{r:g}", "fid", "cam", {"reacquire_sim": r}) for r in args.reacquire_sweep]
    if has_mats:
        v += [("fid/geo", "fid", "geo", {}),
              (f"fid/geo+apart{args.apart:g}", "fid", "geo", {"apart_dist": args.apart})]
        v += [(f"fid/geo@{d:g}", "fid", "geo", {"handover_dist": d}) for d in args.dist_sweep]
        area = {"floor_area": args.floor_area} if args.floor_area else {}
        v += [("fid/fused", "fid", "fused", dict(area))]
        v += [(f"fid/fused@{d:g}", "fid", "fused", dict(area, fuse_dist=d)) for d in args.fuse_sweep]
        v += [("fid/fused-noarea", "fid", "fused", {})] if area else []
        r9 = {} if cal and "reacquire_sim" in cal else {"reacquire_sim": 0.9}   # uncalibrated: 0.9 was best
        v += [("fid/link", "fid", "link", dict(r9)),
              ("fid/link-final", "fid", "link-final", dict(r9))]
        v += [(f"fid/link-final@{d:g}", "fid", "link-final", dict(r9, link_dist=d)) for d in args.link_sweep]
        v += [("fid/link-final-nomargin", "fid", "link-final", dict(r9, link_margin=0.0)),
              ("fid/link-final+contra", "fid", "link-final", dict(r9, link_contra=0.3))]
        if cal and cal.get("late_sim"):
            v += [("fid/link-final-nolate", "fid", "link-final", {"late_sim": 0.0})]
    if cal and cal.get("late_sim"):
        v += [("fid/still-nolate", "fid", "cam", {"late_sim": 0.0})]
    if args.appearance:
        v += [("fid/mct", "fid", "mct", {})]
        v += [(f"fid/mct@{x:g}", "fid", "mct", {"handover_min_sim": x, "reacquire_sim": max(x, 0.7)})
              for x in args.sweep]
    if args.dino:
        v += [("dino/still", "dino", "cam", {}), ("dino/mct", "dino", "mct", {})]
    return v


def load_all(args):
    meta = json.load(open(os.path.join(args.data, "meta.json"), encoding="utf-8"))
    args.floor_area = floor_area(args.data)
    cams = meta["cams"]
    gt = read_labels(args.data, cams)
    emb, outside = in_pen(load_emb(args.data, cams), pen_masks(gt))
    return meta, cams, gt, emb, outside, proj_mats(args.data, cams)


def cmd_score(args):
    meta, cams, gt, emb, outside, mats = load_all(args)
    print(f"[score] {outside:.1%} of the detections are outside the pen (the neighbouring pens) and left out; "
          f"projection matrices for {len(mats)} of {len(cams)} cameras", flush=True)
    has_lying = any(x[3] is not None for per in emb.values() for v in per.values() for x in v)
    if mats and not has_lying:
        print("[score] !! no p(lying) in the embeddings (an older embed): the floor point is taken at one body "
              "height; mct_eval.py embed again for the posture", flush=True)
    results = {"data": {"cams": cams, "minutes": (meta["end_ts"] - meta["start_ts"]) / 60, "step_s": meta["step"],
                        "labelled_frames": sum(len(p) for p in gt.values()), "outside_pen": outside,
                        "cannot_link_rule": {k: DEFAULTS["identity"][k] for k in ("cannot_link_min_s", "cannot_link_max_iou")}}}
    cal = load_thresholds(args.data, cams)
    from reid import MCT_KEYS
    cal_mct = {k: cal[k] for k in MCT_KEYS if k in cal}
    if cal_mct:
        print("[score] the identity model's calibrated thresholds: " + ", ".join(f"{k} {v:.3f}" for k, v in cal_mct.items())
              + (f", new_cow_similarity {cal['new_cow_similarity']:.3f}" if "new_cow_similarity" in cal else ""), flush=True)
        results["data"]["thresholds"] = cal
        if "new_cow_similarity" in cal:
            args.similarity = sorted(set(args.similarity) | {round(cal["new_cow_similarity"], 3)})
    prepared = {}
    rows = []
    for name, vec, kind, over in variants(args, bool(mats), cal):
        if vec == "fid" and cal_mct:
            over = dict(cal_mct, **over)
        if vec not in prepared:
            prepared[vec] = prepare(emb, vec)
        dets = prepared[vec]
        if "gt" not in prepared:
            prepared["gt"] = match_gt(gt, dets)
            gt_map, n_gt, n_det = prepared["gt"]
            results["data"]["detector_recall"] = len(gt_map) / max(1, n_gt)
            results["data"]["detector_precision"] = len(gt_map) / max(1, n_det)
            results["data"]["look"] = lk = look_check(gt_map, dets)
            if lk["queries"]:
                print(f"[score] by look: a cow in one camera, the nearest frame id among another camera's cows "
                      f"at that second is hers in {lk['look_cross_cam_top1']:.1%} (chance {lk['chance']:.1%}"
                      + "".join(f"; {name} {lk[k]:.1%}" for k, name in (
                          ("look_cross_cam_top1_tagged", "cows 1-10"),
                          ("look_cross_cam_top1_untagged", "cows 11-16 that reid.py never trains on")) if k in lk)
                      + ")", flush=True)
            if mats:
                results["data"]["floor_points"] = fc = floor_check(gt_map, dets, mats)
                if fc:
                    print(f"[score] floor points, labelled cows: one cow in two cameras {fc['same_cow_median_cm']:.0f} cm "
                          f"apart (median; p90 {fc['same_cow_p90_cm']:.0f}, {fc['same_cow_within_100cm']:.0%} within 1 m); "
                          f"two cows within 1 m {fc['other_cows_within_100cm']:.1%}", flush=True)
        gt_map, n_gt, n_det = prepared["gt"]
        t = time.time()
        got, links, stats = run_tracker(dets, kind, over, mats)
        r, major = score_run(got, links, gt_map, gt, dets, n_gt, n_det)
        r.update(tracker=stats, tracks=len(set(got.values())), seconds=round(time.time() - t, 1), mct=over, kind=kind)
        if name in args.enroll:
            pb = pseudo_bursts(got, dets)
            r["enroll"] = {f"{k}@{sim:g}": e for sim in args.similarity
                           for k, e in enroll_eval(pb, links, major, args.min_bursts, sim).items()}
        results[name] = r
        rows.append((name, r))
        print(f"[score] {name:18s} IDF1 camera {r['idf1_camera']:.3f} global {r['idf1_global']:.3f}  "
              f"tracks/cow {r['tracks_per_cow'] or 0:.1f}  switches {r['id_switches']}  cross-camera same "
              f"{(r['cross_cam_same_track'] or 0):.1%}  false merges {r['false_merges']}  false cannot_link "
              f"{r['false_cannot_link']}/{r['cannot_link_pairs']}  {stats}  ({r['seconds']} s)", flush=True)
    best = max(rows, key=lambda x: x[1]["idf1_global"])[0]
    results["data"]["best"] = best
    write_json(os.path.join(args.data, "mct_eval.json"), results)
    d = results["data"]
    print(f"\nMmCows 7/25, {d['minutes']:.0f} min, {len(cams)} cameras; detector at the labelled seconds: recall "
          f"{d['detector_recall']:.1%}, precision {d['detector_precision']:.1%}; cannot_link after "
          f"{d['cannot_link_rule']['cannot_link_min_s']} s together\n")
    print("| tracker | IDF1 per camera | IDF1 all cameras | tracks per cow | ID switches / cow / h | "
          "cow in 2 cameras on one track | false merges | false cannot_link | track purity |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, r in rows:
        print(f"| {key} | {r['idf1_camera']:.3f} | {r['idf1_global']:.3f} | {r['tracks_per_cow'] or 0:.1f} | "
              f"{r['id_switches_per_cow_hour']:.1f} | {(r['cross_cam_same_track'] or 0):.1%} | {r['false_merges']} | "
              f"{r['false_cannot_link']} of {r['cannot_link_pairs']} | {r['track_purity']:.1%} |")
    for key, r in rows:
        if "enroll" in r:
            print(f"\nNew cows from {key}'s tracks (gallery.enroll on 7 s pseudo-bursts, min {args.min_bursts} "
                  f"bursts; name@new_cow_similarity):")
            for name, e in r["enroll"].items():
                print(f"  {name:17s} {e['cows_made']} cows made for {e['true_cows']}; ids per cow "
                      f"{e['ids_per_cow'] or 0:.2f}; cows with no id {e['cows_without_id']}; purity "
                      f"{(e['purity'] or 0):.1%}; bursts left in the pool {e['pool_left']}")
    print(f"\nbest IDF1 over all cameras: {best}\n-> {os.path.join(args.data, 'mct_eval.json')}")


# ------------------------------------------------------------------ show

def cow_positions(data):
    """{ts: {cow: (x, y)}} from visual_location/*.csv (MmCows' own 3D positions, every 15 s)."""
    import csv
    out = collections.defaultdict(dict)
    folder = os.path.join(data, "visual_location")
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        m = re.match(r"C(\d+)_", f)
        if m:
            for r in csv.DictReader(open(os.path.join(folder, f), encoding="utf-8")):
                out[int(float(r["timestamp"]))][int(m.group(1))] = (float(r["loc_x_cm"]), float(r["loc_y_cm"]))
    return out


def cmd_show(args):
    """The four cameras and the floor, a frame a second: every box in its track's colour with the
    track's number, the cow that track mostly is, and - on the labelled seconds - the cow it really
    is (a red frame when the two differ)."""
    import cv2
    from show_ids import Writer, colour, draw_box, header
    meta, cams, gt, emb, _, mats = load_all(args)
    known = {n: (vec, kind, over) for n, vec, kind, over in variants(args, bool(mats))}
    if args.variant not in known:
        sys.exit(f"--variant: one of {', '.join(known)}")
    vec, kind, over = known[args.variant]
    dets = prepare(emb, vec)
    got, links, _ = run_tracker(dets, kind, over, mats)
    gt_map, n_gt, n_det = match_gt(gt, dets)
    r, major = score_run(got, links, gt_map, gt, dets, n_gt, n_det)
    order = {}
    for key in sorted(got, key=lambda k: (k[1], k[0], k[2])):
        order.setdefault(got[key], len(order) + 1)
    mappers = {c: floor_mapper({"proj_mat": m, "proj_size": PROJ_SIZE}, DEFAULTS["mct"]) for c, m in mats.items()}
    truth = cow_positions(args.data)
    t0 = meta["start_ts"] + int(args.from_min * 60)
    t1 = min(meta["end_ts"], t0 + int(args.minutes * 60))
    seconds = [t for t in sorted({t for per in dets.values() for t in per}) if t0 <= t < t1]
    xy = [mappers[c](d[0], 0.5 if d[2] is None else d[2]) for c in mappers for t in seconds for d in dets[c].get(t, [])]
    TW, TH, PW = 960, 600, 600
    if xy:
        xs, ys = np.array(xy).T
        # the panel is tall: the pen's long side goes down it
        flip = np.ptp(np.percentile(xs, [1, 99])) > np.ptp(np.percentile(ys, [1, 99]))
        if flip:
            xs, ys = ys, xs
        lo, blo = np.percentile(xs, 1) - 100, np.percentile(ys, 1) - 100
        # one scale for both axes: the pen keeps its shape
        k = min((PW - 40) / (np.percentile(xs, 99) + 100 - lo), (2 * TH - 120) / (np.percentile(ys, 99) + 100 - blo))
    out = os.path.join(args.out or os.path.join(args.data, "show"), args.variant.replace("/", "_"))
    w = Writer(out, args.fps, args.still_every, f"MmCows: who is who ({args.variant})")
    wrong = shown = 0
    for ts in seconds:
        tiles = []
        for cam in cams:
            path = os.path.join(args.data, "frames", cam, f"{ts}.jpg")
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) if os.path.exists(path) else np.zeros((TH, TW, 3), np.uint8)
            img = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
            for j, d in enumerate(dets[cam].get(ts, [])):
                g = got[(cam, ts, j)]
                mc, cow = major.get(g), gt_map.get((cam, ts, j))
                lines = [f"#{order[g]}" + (f" = cow {mc}" if mc is not None else "")]
                bad = cow is not None and mc is not None and cow != mc
                if cow is not None:
                    lines.append(f"true: cow {cow}" + ("  WRONG" if bad else ""))
                draw_box(img, d[0], colour(g), lines, bad, scale=0.45)
                shown += 1
                wrong += bad
            header(img, f"{cam}  {len(dets[cam].get(ts, []))} cows", 0.6)
            tiles.append(img)
        while len(tiles) < 4:
            tiles.append(np.zeros((TH, TW, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        panel = np.full((2 * TH, PW, 3), 25, np.uint8)
        if xy:
            to = lambda p: (int(((p[1] if flip else p[0]) - lo) * k) + 20, int(((p[0] if flip else p[1]) - blo) * k) + 50)
            cv2.line(panel, (20, 2 * TH - 30), (20 + int(100 * k), 2 * TH - 30), (200, 200, 200), 2)
            cv2.putText(panel, "1 m", (26 + int(100 * k), 2 * TH - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            near = min(truth, key=lambda t: abs(t - ts), default=None)
            if near is not None and abs(near - ts) <= 7:
                for cow, p in truth[near].items():
                    cv2.circle(panel, to(p), 16, (230, 230, 230), 1, cv2.LINE_AA)
                    cv2.putText(panel, str(cow), (to(p)[0] + 17, to(p)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (230, 230, 230), 1, cv2.LINE_AA)
            for c in mappers:
                for j, d in enumerate(dets[c].get(ts, [])):
                    g = got[(c, ts, j)]
                    q = to(mappers[c](d[0], 0.5 if d[2] is None else d[2]))
                    cv2.circle(panel, q, 6, colour(g), -1, cv2.LINE_AA)
                    cv2.putText(panel, f"#{order[g]}", (q[0] - 10, q[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                colour(g), 1, cv2.LINE_AA)
        header(panel, "floor: dots = tracks, rings = true cows", 0.5)
        canvas = np.vstack([np.zeros((32, grid.shape[1] + PW, 3), np.uint8), np.hstack([grid, panel])])
        stamp = time.strftime("%H:%M:%S", time.gmtime(ts - 5 * 3600))     # the farm's clock (CDT)
        header(canvas, f"MmCows 7/25 {stamp}   {args.variant}   colour and # = track; '= cow N' = the cow the "
                       f"track mostly is; red = that is not this cow" + ("   [labelled second]" if any(ts in gt[c] for c in cams) else ""), 0.6)
        w.add(canvas, f"{stamp}")
    page = w.close(f"{args.variant}: IDF1 over all cameras {r['idf1_global']:.3f}, one cow in two cameras on one "
                   f"track {(r['cross_cam_same_track'] or 0):.0%}, {r['tracks_per_cow'] or 0:.1f} tracks per cow. "
                   f"{wrong} wrong boxes on the labelled seconds shown.")
    print(f"[show] {len(seconds)} seconds -> {page} and {os.path.join(out, 'ids.mp4')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--data", required=True)
    f.add_argument("--start", type=float, default=0, help="minutes after 12:00")
    f.add_argument("--minutes", type=float, default=60, help="up to 180 (the whole 3 h set)")
    f.add_argument("--step", type=int, default=1, help="a frame every N s")
    f.add_argument("--cams", nargs="+", default=list(CAMS))
    f.add_argument("--max-side", type=int, default=1920, help="the barn's frame width")
    f.add_argument("--workers", type=int, default=32)
    e = sub.add_parser("embed")
    e.add_argument("--data", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--detector", required=True)
    e.add_argument("--threshold", type=float, default=None)
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--force", action="store_true")
    e.add_argument("--identity", default="", help="reid.py's identity.pt: the frame id from it")
    s = sub.add_parser("score")
    v = sub.add_parser("show")
    for q in (s, v):
        q.add_argument("--data", required=True)
        q.add_argument("--apart", type=float, default=500.0, help="geo+apart: cows this far apart (cm) are two")
        q.add_argument("--dist-sweep", type=float, nargs="*", default=[60.0, 150.0], help="handover_dist tried (cm)")
        q.add_argument("--appearance", action="store_true", help="also hand over by look (stage B) with --sweep")
        q.add_argument("--sweep", type=float, nargs="*", default=[0.6, 0.8], help="handover_min_sim tried with --appearance")
        q.add_argument("--dino", action="store_true", help="also DINOv2's CLS as the vector")
        q.add_argument("--reacquire-sweep", type=float, nargs="*", default=[0.8, 0.9],
                       help="reacquire_sim tried (the frame id's space differs with an identity model)")
        q.add_argument("--fuse-sweep", type=float, nargs="*", default=[120.0, 200.0], help="fused: fuse_dist tried (cm)")
        q.add_argument("--link-sweep", type=float, nargs="*", default=[80.0, 160.0], help="link: link_dist tried (cm)")
    s.add_argument("--enroll", nargs="*", default=["fid/still~0.9", "fid/link-final"],
                   help="trackers whose tracks enroll is tried on")
    s.add_argument("--min-bursts", type=int, default=10, help="enroll: bursts a new cow needs")
    s.add_argument("--similarity", type=float, nargs="+", default=[0.55, 0.7, 0.8, 0.9],
                   help="new_cow_similarity values tried by enroll (an identity model's space differs)")
    v.add_argument("--variant", default="fid/geo", help="the tracker drawn (a name from score's table)")
    v.add_argument("--from-min", type=float, default=0.0)
    v.add_argument("--minutes", type=float, default=10.0)
    v.add_argument("--fps", type=float, default=4)
    v.add_argument("--still-every", type=int, default=30)
    v.add_argument("--out", default=None, help="default <data>/show/<variant>")
    args = p.parse_args(argv)
    {"fetch": cmd_fetch, "embed": cmd_embed, "score": cmd_score, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
