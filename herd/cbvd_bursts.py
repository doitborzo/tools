"""CBVD-5 as herd training data, and Stage A: encode every crop once.

    specs    annotations -> bursts (one per cow track per clip) + keyframe crops
    extract  videos/keyframes -> frozen DINOv2 vectors on disk, per clip:
             features/<split>/<clip>.npz   burst_feats (B, T, D) fp16, burst_times (B, T)
                                           key_feats (K, D) fp16
             features/<split>/index.jsonl  one row per burst and per keyframe, with labels
    motion   videos -> the rhythm of each burst's crops (motion.py), no GPU:
             features/<split>/<clip>.motion.npz   motion (B, DIM), ok (B,)
    degrade  stretches of each burst spoilt (degrade.py: occlusion by another cow,
             mud, blur, dark) and encoded, for the quality head to learn on:
             features/<split>/<clip>.degraded.npz   deg_idx (B, K) frame or -1,
                                                    deg_kind (B, K), deg_feats (B, K, D)
    keys     the detector on every keyframe -> its boxes, matched to the annotated
             cows, and one jittered copy of each annotated box, encoded like the
             keyframes: what the once-a-second heads get in the barn
             features/<split>/keys_aug.npz (feats), keys_aug.jsonl (rows)

A burst is 7 s at 25 fps of one cow. CBVD-5 clips are 10 s at 25 fps with
keyframes at whole seconds (keyframe t = frame t*25, checked in cowbench's
render.py); the cow's box is annotated on keyframes only, so on the frames in
between it is interpolated. Labels per burst are the majority over its
keyframes. CBVD-5 has no cow identities: a track within a clip is one
identity, which teaches "same cow over time, other cows apart" but not across
days - that needs the barn's own footage.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np

from common import ACTIVITIES, POSTURES, crop, interpolate_box, read_jsonl

import cbvd  # noqa: E402  (cowbench)
import tracks as tracks_mod  # noqa: E402

CLIP_SECONDS = 10.0
VIDEO_FPS = 25.0


def labels_of(box):
    """CBVD-5 labels -> herd's: posture, activity (feeding/drinking/none) and
    rumination as its own yes/no (a ruminating cow is neither feeding nor drinking)."""
    act = box.activity
    return {"posture": POSTURES.index(box.posture) if box.posture in POSTURES else -1,
            "activity": ACTIVITIES.index(act) if act in ACTIVITIES else ACTIVITIES.index("none"),
            "rumination": 1 if act == "ruminating" else 0}


def split_boxes(root, split):
    ann = os.path.join(root, "annotations")
    val = cbvd.partition(cbvd.load_boxes(os.path.join(ann, "ava_val_v2.1.csv")))[0]
    if split == "val":
        return val
    train = cbvd.partition(cbvd.load_boxes(os.path.join(ann, "ava_train_v2.1.csv")))[0]
    val_clips = {b.video_id for b in val}
    return [b for b in train if b.video_id not in val_clips]


def majority(values, center_first=None):
    vals = [v for v in values if v >= 0]
    if not vals:
        return -1
    c = collections.Counter(vals).most_common()
    if len(c) > 1 and c[0][1] == c[1][1] and center_first is not None and center_first >= 0:
        return center_first
    return c[0][0]


def build_specs(boxes, burst_seconds=7.0):
    """-> (bursts, keyframes) per clip: {clip: {"bursts": [...], "keys": [...]}}."""
    rows = [{"id": b.uid, "video_id": b.video_id, "timestamp": b.timestamp,
             "bbox": list(b.xyxy), "box": b} for b in boxes]
    clips = collections.defaultdict(lambda: {"bursts": [], "keys": []})
    for r in rows:
        b = r["box"]
        clips[b.video_id]["keys"].append({"uid": b.uid, "timestamp": b.timestamp, "bbox": list(b.xyxy),
                                          **labels_of(b)})
    for n, track in enumerate(tracks_mod.build(rows, threshold=0.3, min_length=1)):
        vid = track[0]["video_id"]
        keys = {r["timestamp"]: r["bbox"] for r in track}
        labs = [labels_of(r["box"]) for r in track]
        times = sorted(keys)
        center = (times[0] + times[-1]) / 2
        start = min(max(0.0, center - burst_seconds / 2), max(0.0, CLIP_SECONDS - burst_seconds))
        mid = min(range(len(track)), key=lambda i: abs(track[i]["timestamp"] - center))
        clips[vid]["bursts"].append({
            "track": f"{vid}_t{len(clips[vid]['bursts'])}", "start": round(start, 3),
            "uids": [r["id"] for r in track],
            "seconds": burst_seconds, "keys": {str(t): keys[t] for t in times},
            "posture": majority([l["posture"] for l in labs], labs[mid]["posture"]),
            "activity": majority([l["activity"] for l in labs], labs[mid]["activity"]),
            "rumination": majority([l["rumination"] for l in labs], labs[mid]["rumination"]),
            "n_keyframes": len(track), "area": float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in keys.values()])),
        })
    return dict(clips)


def photo_aug(crops, rng):
    """One random brightness / contrast / greyscale for the whole burst (day,
    night, infrared): the same cow must stay the same cow."""
    x = crops.astype(np.float32)
    x = (x - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-30, 30)
    if rng.random() < 0.2:
        g = x.mean(-1, keepdims=True)
        x = np.repeat(g, 3, -1)
    return np.clip(x, 0, 255).astype(np.uint8)


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    cap.release()
    return frames, fps


def encode(encoder, crops, batch=256):
    import torch
    out = []
    dev = next(encoder.parameters()).device
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(np.ascontiguousarray(crops[i:i + batch])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(encoder(x).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, encoder.out_dim), np.float16)


def cmd_extract(args):
    import torch
    from model import FrameEncoder
    from common import device

    boxes = split_boxes(args.root, args.split)
    clips = build_specs(boxes, args.burst_seconds)
    if args.limit_clips:
        clips = dict(sorted(clips.items(), key=lambda kv: int(kv[0]))[:args.limit_clips])
    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.jsonl")
    done = {r["clip"] for r in read_jsonl(index_path) if r.get("kind") == "clip_done"}
    todo = [c for c in sorted(clips, key=int) if c not in done]
    print(f"[extract] {args.split}: {len(clips)} clips, {sum(len(c['bursts']) for c in clips.values())} bursts, "
          f"{sum(len(c['keys']) for c in clips.values())} keyframe cows; {len(todo)} clips to do", flush=True)
    if not todo:
        return
    enc = FrameEncoder(args.encoder, args.grid).to(device()).eval()
    rng = random.Random(args.seed)
    n_frames = int(round(args.burst_seconds * VIDEO_FPS))
    t0 = time.time()
    for k, clip in enumerate(todo, 1):
        spec = clips[clip]
        rows = []
        # keyframe crops: the exact annotated frame and box
        key_crops = []
        for key in spec["keys"]:
            img = np.asarray(cbvd_frame(args.root, clip, key["timestamp"]))
            key_crops.append(crop(img, key["bbox"], args.crop, args.margin))
        key_feats = encode(enc, np.stack(key_crops)) if key_crops else np.zeros((0, enc.out_dim), np.float16)
        for i, key in enumerate(spec["keys"]):
            rows.append({"kind": "key", "clip": clip, "i": i, **{k2: key[k2] for k2 in
                         ("uid", "timestamp", "bbox", "posture", "activity", "rumination")}})
        # bursts: one decode of the clip, every cow's crops from it
        burst_feats = np.zeros((0, n_frames, enc.out_dim), np.float16)
        burst_times = np.zeros((0, n_frames), np.float32)
        try:
            frames, fps = read_video(cbvd.video_path(args.root, clip)) if spec["bursts"] else ([], VIDEO_FPS)
        except FileNotFoundError:
            frames, fps = [], VIDEO_FPS
            print(f"  clip {clip}: no video - keyframes only", flush=True)
        if frames:
            feats, times = [], []
            for b in spec["bursts"]:
                keys = {float(t): v for t, v in b["keys"].items()}
                ts = [b["start"] + j / VIDEO_FPS for j in range(n_frames)]
                idx = [min(len(frames) - 1, int(round(t * fps))) for t in ts]
                crops = np.stack([crop(frames[ix], interpolate_box(keys, t), args.crop, args.margin)
                                  for ix, t in zip(idx, ts)])
                if args.split == "train" and args.photo_aug and rng.random() < 0.5:
                    crops = photo_aug(crops, rng)
                feats.append(encode(enc, crops))
                times.append(np.asarray(ts, np.float32) - b["start"])
            burst_feats, burst_times = np.stack(feats), np.stack(times)
            for i, b in enumerate(spec["bursts"]):
                rows.append({"kind": "burst", "clip": clip, "i": i, **{k2: b[k2] for k2 in
                             ("track", "uids", "posture", "activity", "rumination", "n_keyframes", "area", "start")}})
            del frames
        np.savez(os.path.join(out_dir, f"{clip}.npz"), burst_feats=burst_feats, burst_times=burst_times,
                 key_feats=key_feats)
        rows.append({"kind": "clip_done", "clip": clip})
        with open(index_path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        el = time.time() - t0
        print(f"\r  {k}/{len(todo)} clips  {el / 60:.1f} min  ~{el / k * (len(todo) - k) / 60:.0f} min left",
              end="", flush=True)
    print(flush=True)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"encoder": args.encoder, "grid": args.grid, "crop": args.crop, "margin": args.margin,
                   "dim": enc.out_dim,
                   "burst_seconds": args.burst_seconds, "fps": VIDEO_FPS}, fh, indent=2)


def _motion_clip(job):
    """One clip's bursts -> motion features (a worker process)."""
    from motion import DIM, motion_features
    root, clip, bursts, size, margin, n_frames, path = job
    try:
        frames, fps = read_video(cbvd.video_path(root, clip))
    except FileNotFoundError:
        return clip, 0
    feats, oks = [], []
    for b in bursts:
        keys = {float(t): v for t, v in b["keys"].items()}
        ts = [b["start"] + j / VIDEO_FPS for j in range(n_frames)]
        idx = [min(len(frames) - 1, int(round(t * fps))) for t in ts]
        crops = np.stack([crop(frames[ix], interpolate_box(keys, t), size, margin) for ix, t in zip(idx, ts)])
        f, ok = motion_features(crops, np.asarray(ts) - b["start"])
        feats.append(f)
        oks.append(ok)
    np.savez(path + ".part.npz", motion=np.stack(feats) if feats else np.zeros((0, DIM), np.float32),
             ok=np.array(oks, bool))
    os.replace(path + ".part.npz", path)
    return clip, len(bursts)


def cmd_motion(args):
    """Motion features for every burst already extracted, from the same crops
    (the features' own margin and size); clip by clip, resumable, CPU only."""
    import multiprocessing as mp
    from motion import BANDS, DIM, GRID, SIDE
    out_dir = os.path.join(args.out, args.split)
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    rows = [r for r in read_jsonl(os.path.join(out_dir, "index.jsonl")) if r["kind"] == "burst"]
    tracks = collections.defaultdict(dict)
    for r in rows:
        tracks[r["clip"]][r["i"]] = r["track"]
    clips = build_specs(split_boxes(args.root, args.split), meta.get("burst_seconds", 7.0))
    n_frames = int(round(meta.get("burst_seconds", 7.0) * VIDEO_FPS))
    jobs = []
    for clip, have in sorted(tracks.items(), key=lambda kv: int(kv[0])):
        bursts = clips.get(clip, {}).get("bursts", [])
        if [b["track"] for b in bursts] != [have[i] for i in sorted(have)]:
            sys.exit(f"clip {clip}: bursts differ from the extracted ones - re-run extract")
        path = os.path.join(out_dir, f"{clip}.motion.npz")
        if not os.path.exists(path):
            jobs.append((args.root, clip, bursts, meta.get("crop", 224), meta.get("margin", 0.1), n_frames, path))
    print(f"[motion] {args.split}: {len(tracks)} clips with bursts, {len(jobs)} to do, {args.workers} processes",
          flush=True)
    t0, n = time.time(), 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, (clip, nb) in enumerate(pool.imap_unordered(_motion_clip, jobs), 1):
            n += nb
            el = time.time() - t0
            print(f"\r  {k}/{len(jobs)} clips, {n} bursts  {el / 60:.1f} min  ~{el / k * (len(jobs) - k) / 60:.0f} min left",
                  end="", flush=True)
    print(flush=True)
    with open(os.path.join(out_dir, "motion_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"dim": DIM, "bands": BANDS, "grid": GRID, "side": SIDE, "crop": meta.get("crop", 224),
                   "margin": meta.get("margin", 0.1)}, fh, indent=2)


def cmd_degrade(args):
    """Spoilt twins of a part of every burst's frames (degrade.py), encoded with
    the features' own encoder - clip by clip, resumable. Only the spoilt frames
    are stored; the clean ones are in <clip>.npz already."""
    import torch
    import degrade as deg
    from model import FrameEncoder
    from common import device
    out_dir = os.path.join(args.out, args.split)
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    rows = [r for r in read_jsonl(os.path.join(out_dir, "index.jsonl")) if r["kind"] == "burst"]
    tracks = collections.defaultdict(dict)
    for r in rows:
        tracks[r["clip"]][r["i"]] = r["track"]
    clips = build_specs(split_boxes(args.root, args.split), meta.get("burst_seconds", 7.0))
    n_frames = int(round(meta.get("burst_seconds", 7.0) * VIDEO_FPS))
    size, margin = meta.get("crop", 224), meta.get("margin", 0.1)
    todo = [c for c in sorted(tracks, key=int) if not os.path.exists(os.path.join(out_dir, f"{c}.degraded.npz"))]
    print(f"[degrade] {args.split}: {len(tracks)} clips with bursts, {len(todo)} to do", flush=True)
    if not todo:
        return
    enc = FrameEncoder(meta["encoder"], meta["grid"]).to(device()).eval()
    t0 = time.time()
    for k, clip in enumerate(todo, 1):
        bursts = clips.get(clip, {}).get("bursts", [])
        if [b["track"] for b in bursts] != [tracks[clip][i] for i in sorted(tracks[clip])]:
            sys.exit(f"clip {clip}: bursts differ from the extracted ones - re-run extract")
        rng = random.Random(f"{args.seed}-{clip}")
        try:
            frames, fps = read_video(cbvd.video_path(args.root, clip))
        except FileNotFoundError:
            continue
        all_crops = []
        for b in bursts:
            keys = {float(t): v for t, v in b["keys"].items()}
            ts = [b["start"] + j / VIDEO_FPS for j in range(n_frames)]
            idx = [min(len(frames) - 1, int(round(t * fps))) for t in ts]
            all_crops.append(np.stack([crop(frames[ix], interpolate_box(keys, t), size, margin)
                                       for ix, t in zip(idx, ts)]))
        del frames
        K = int(n_frames * 0.65)
        deg_idx = np.full((len(bursts), K), -1, np.int16)
        deg_kind = np.zeros((len(bursts), K), np.int8)
        deg_feats = np.zeros((len(bursts), K, meta["dim"]), np.float16)
        for i, crops in enumerate(all_crops):
            others = [c for j, c in enumerate(all_crops) if j != i]
            ix, kinds, spoilt = deg.spoil(crops, deg.plan(len(crops), rng), others, rng)
            ix, kinds, spoilt = ix[:K], kinds[:K], spoilt[:K]
            if len(ix):
                with torch.no_grad():
                    deg_feats[i, :len(ix)] = encode(enc, spoilt)
                deg_idx[i, :len(ix)] = ix
                deg_kind[i, :len(ix)] = kinds
        path = os.path.join(out_dir, f"{clip}.degraded.npz")
        np.savez(path + ".part.npz", deg_idx=deg_idx, deg_kind=deg_kind, deg_feats=deg_feats)
        os.replace(path + ".part.npz", path)
        el = time.time() - t0
        print(f"\r  {k}/{len(todo)} clips  {el / 60:.1f} min  ~{el / k * (len(todo) - k) / 60:.0f} min left",
              end="", flush=True)
    print(flush=True)


def jitter_box(b, rng, shift=0.12, scale=0.15):
    """An annotated box as a detector might draw it: centre moved up to `shift`
    of its size, each side scaled by up to `scale`."""
    w, h = b[2] - b[0], b[3] - b[1]
    cx = (b[0] + b[2]) / 2 + rng.uniform(-shift, shift) * w
    cy = (b[1] + b[3]) / 2 + rng.uniform(-shift, shift) * h
    w *= rng.uniform(1 - scale, 1 + scale)
    h *= rng.uniform(1 - scale, 1 + scale)
    return [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]


def cmd_keys(args):
    """Keyframe crops as the barn makes them: from the detector's boxes (matched
    to the annotated cows at IoU >= --match-iou, loose boxes included - the
    heads have to answer about those too) and jittered annotated boxes. Labels
    are the annotated cow's. Same encoder, grid, crop and margin as extract."""
    import torch
    import detect as detect_mod
    import detector as det_mod
    from model import FrameEncoder
    from common import device
    out_dir = os.path.join(args.out, args.split)
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    keys = [r for r in read_jsonl(os.path.join(out_dir, "index.jsonl")) if r["kind"] == "key"]
    by_frame = collections.defaultdict(list)
    for r in keys:
        by_frame[(r["clip"], r["timestamp"])].append(r)
    out_path = os.path.join(out_dir, "keys_aug.npz")
    meta_path = os.path.join(out_dir, "keys_aug_meta.json")
    det = det_mod.Live(args.detector, args.det_threshold, tiles=args.det_tiles)
    made_with = {"detector": os.path.abspath(args.detector), "threshold": det.threshold, "tiles": det.tiles,
                 "match_iou": args.match_iou, "jitter": args.jitter}
    if os.path.exists(out_path) and os.path.exists(meta_path) and not args.fresh:
        if json.load(open(meta_path, encoding="utf-8")) == made_with:
            print(f"[keys] done already with this detector: {out_path} (--fresh to redo)")
            return
        print("[keys] made with another detector or settings - redoing", flush=True)
    enc = FrameEncoder(meta["encoder"], meta["grid"]).to(device()).eval()
    size, margin = meta.get("crop", 224), meta.get("margin", 0.1)
    rng = random.Random(args.seed)
    rows, feats = [], []
    n_det = n_missed = 0
    t0 = time.time()
    frames = sorted(by_frame.items(), key=lambda kv: (int(kv[0][0]), kv[0][1]))
    for k, ((clip, ts), gts) in enumerate(frames, 1):
        img = cbvd_frame(args.root, clip, ts)
        arr = np.asarray(img)
        boxes = [c["bbox"] for c in det(img)]
        pairs = detect_mod.match(boxes, [g["bbox"] for g in gts], args.match_iou)
        crops, new = [], []
        for i, j, v in pairs:
            crops.append(crop(arr, boxes[i], size, margin))
            new.append({"source": "detector", "iou": round(v, 3), "bbox": boxes[i], "g": gts[j]})
        n_det += len(pairs)
        n_missed += len(gts) - len(pairs)
        for g in gts:
            for _ in range(args.jitter):
                b = jitter_box(g["bbox"], rng)
                crops.append(crop(arr, b, size, margin))
                new.append({"source": "jitter", "iou": None, "bbox": b, "g": g})
        if crops:
            with torch.no_grad():
                feats.append(encode(enc, np.stack(crops)))
            for r in new:
                g = r.pop("g")
                rows.append({"clip": clip, "uid": g["uid"], "timestamp": ts, "posture": g["posture"],
                             "activity": g["activity"], "rumination": g["rumination"], **r})
        el = time.time() - t0
        print(f"\r  {k}/{len(frames)} keyframes, {n_det} detector boxes on cows, {n_missed} cows it missed  "
              f"{el / 60:.1f} min  ~{el / k * (len(frames) - k) / 60:.0f} min left", end="", flush=True)
    print(flush=True)
    feats = np.concatenate(feats) if feats else np.zeros((0, meta["dim"]), np.float16)
    np.savez(out_path + ".part.npz", feats=feats)
    with open(os.path.join(out_dir, "keys_aug.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(out_path + ".part.npz", out_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(made_with, fh, indent=2)
    print(f"[keys] {args.split}: {len(rows)} crops ({n_det} from the detector, "
          f"{sum(1 for r in rows if r['source'] == 'jitter')} jittered) -> {out_path}")


def cbvd_frame(root, clip, ts):
    from PIL import Image
    return Image.open(cbvd.frame_path(root, cbvd.Box(clip, ts, 0, 0, 0, 0, "1", ()))).convert("RGB")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = argv[0] if argv and argv[0] in ("motion", "keys", "degrade") else "extract"
    if sub != "extract":
        argv = argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 2),
                   help="motion: processes (each holds one decoded clip)")
    p.add_argument("--out", default="/workspace/herd/features")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--encoder", default="facebook/dinov2-small")
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--margin", type=float, default=0.1,
                   help="context around the box, per side, as a share of its long side: feeding and "
                        "drinking are told apart by what is around the head (feed barrier, trough)")
    p.add_argument("--burst-seconds", type=float, default=7.0)
    p.add_argument("--photo-aug", type=int, default=1)
    p.add_argument("--limit-clips", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detector", default="/workspace/lora-runs/detector/best", help="keys: RT-DETRv2 best/")
    p.add_argument("--det-threshold", type=float, default=None, help="keys: default the detector's own")
    p.add_argument("--det-tiles", type=int, default=None, help="keys: whole frame + tiles; default as trained")
    p.add_argument("--match-iou", type=float, default=0.3, help="keys: detector box <-> annotated cow")
    p.add_argument("--jitter", type=int, default=1, help="keys: jittered copies of each annotated box")
    p.add_argument("--fresh", action="store_true", help="keys: redo")
    args = p.parse_args(argv)
    {"motion": cmd_motion, "keys": cmd_keys, "degrade": cmd_degrade, "extract": cmd_extract}[sub](args)


if __name__ == "__main__":
    sys.exit(main())
