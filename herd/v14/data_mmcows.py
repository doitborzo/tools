#!/usr/bin/env python3
"""MmCows for v14: labelled frames (cow ids on every box) and the 1-second stream.

From huggingface.co/datasets/neis-lab/mmcows (CC BY-NC-SA 4.0), read member by member
through HTTP Range requests (cowbench/extra_det.py's RemoteZip), nothing downloaded whole:

    labels   visual_data.zip: every label file of 25 July (a frame every 15 s per camera, 4
             cameras): "cow_id cx cy w h" -> <out>/labels.jsonl {cam, ts, image, boxes}
    frames   the labelled images the association transformer learns from: every
             --train-every s OUTSIDE the test window, and every --window-every s inside it
             (to score it), longer side --max-side px -> <out>/frames/<cam>/<ts>.jpg; the
             other pens are painted black in these, which gives each camera's pen mask
             (<out>/masks.npz)
    live     1s_interval_images_3hr/0725_cam_<k>_1200-1500.zip: a frame every --live-step s
             from --start to --end (local time, CDT) -> <out>/live/<cam>/<ts>.jpg - the
             "barn" the onboarding test runs on (not masked: the mask is applied there)
    feats    DINOv2 of every labelled box (the barn's 224 px crop, margin 0.1) and the scene
             grid of every saved labelled frame -> <out>/assoc.npz + <out>/assoc_rows.jsonl
    idcrops  the box vectors of `feats` outside the test window as ID sequences (per cow and
             camera, consecutive frames) for idnet.py --split cows: part "A" before the window,
             "B" after it -> <out>/idcrops/{feats.npy, crops.jsonl, meta.json}
    inspect  how the member names of a zip read as camera and time

The test window (12:00-15:00 by default) is never used for training. The detector of
version 10 was fine-tuned on MmCows' labelled frames except the hours 0, 6, 12 and 18 local
time: so 12:00-13:00 is new to it, 13:00-15:00 is not (its boxes there may be a little
flattering; identification is what is scored).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import time

import numpy as np

import v14common as C
from v14common import MMCOWS_DAY, local_hour, parse_id_boxes, read_jsonl, unix_from_name, write_json

HF = "https://huggingface.co/datasets/neis-lab/mmcows/resolve/main/"
VISUAL = HF + "visual_data.zip"
LIVE = HF + "1s_interval_images_3hr/0725_cam_{k}_1200-1500.zip"
LABEL = re.compile(r"(?:^|/)labels/combined/(\d{4})/(cam_\d)/([^/]+)\.txt$")
IMAGE = re.compile(r"(?:^|/)images/(\d{4})/(cam_\d)/([^/]+)\.(?:jpe?g|png)$", re.I)


def remote(url):
    import extra_det
    return extra_det.RemoteZip(url)


def in_window(ts, a):
    h = local_hour(ts)
    return a.start <= h < a.end


def save_small(data, path, max_side):
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BILINEAR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path + ".part.jpg", quality=92)
    os.replace(path + ".part.jpg", path)
    return img


# ------------------------------------------------------------------ commands

def cmd_labels(args):
    out = os.path.join(args.out, "labels.jsonl")
    if os.path.exists(out):
        print(f"[mmcows] labels done already: {out}")
        return
    z = remote(args.visual)
    names = z.names()
    have = set(names)
    todo = []
    for n in names:
        m = LABEL.search(n)
        if m and m.group(1) == MMCOWS_DAY:
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = n.replace("/labels/combined/", "/images/").rsplit(".", 1)[0] + ext
                if cand in have:
                    img = cand
                    break
            ts = unix_from_name(m.group(3))
            if ts is not None:
                todo.append((n, img, m.group(2), ts))
    print(f"[mmcows] {len(todo)} label files of {MMCOWS_DAY}", flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        texts = list(ex.map(lambda t: z.read(t[0]).decode("utf-8", "replace"), todo))
    os.makedirs(args.out, exist_ok=True)
    cows = set()
    with open(out + ".part", "w", encoding="utf-8") as fh:
        for (n, img, cam, ts), text in sorted(zip(todo, texts), key=lambda x: (x[0][3], x[0][2])):
            boxes = parse_id_boxes(text)
            cows |= {c for c, _ in boxes}
            fh.write(json.dumps({"cam": cam, "ts": ts, "label": n, "image": img, "boxes": boxes}) + "\n")
    os.replace(out + ".part", out)
    print(f"[mmcows] {len(todo)} labelled frames, cows {sorted(cows)} -> {out}", flush=True)


def cmd_frames(args):
    rows = read_jsonl(os.path.join(args.out, "labels.jsonl"))
    if not rows:
        sys.exit("run `labels` first")
    pick, last = [], {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        if not r["image"]:
            continue
        win = in_window(r["ts"], args)
        every = args.window_every if win else args.train_every
        key = (r["cam"], win)
        if r["ts"] - last.get(key, -1e18) >= every - 0.5:
            last[key] = r["ts"]
            pick.append(r)
    todo = [r for r in pick if not os.path.exists(frame_path(args.out, r))]
    print(f"[mmcows] {len(pick)} labelled frames to keep ({sum(in_window(r['ts'], args) for r in pick)} in the "
          f"test window), {len(todo)} to download", flush=True)
    z = remote(args.visual) if todo else None
    t0 = time.time()

    def one(r):
        save_small(z.read(r["image"]), frame_path(args.out, r), args.max_side)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(one, todo), 1):
            if k % 50 == 0:
                print(f"\r  {k}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", end="", flush=True)
    print(flush=True)
    # the pen mask per camera: black (other pens) in nearly all labelled frames
    from PIL import Image
    masks = {}
    for cam in sorted({r["cam"] for r in pick}):
        mine = [r for r in pick if r["cam"] == cam][:: max(1, len(pick) // 200)][:60]
        imgs = [Image.open(frame_path(args.out, r)) for r in mine]
        masks[cam] = C.pen_mask(imgs)
        print(f"[mmcows] {cam}: pen mask keeps {masks[cam].mean():.0%} of the frame", flush=True)
    np.savez(os.path.join(args.out, "masks.npz"), **masks)
    with open(os.path.join(args.out, "frames.jsonl"), "w", encoding="utf-8") as fh:
        for r in pick:
            fh.write(json.dumps({**r, "split": "window" if in_window(r["ts"], args) else "train",
                                 "path": frame_path(args.out, r)}) + "\n")


def frame_path(out, r):
    return os.path.join(out, "frames", r["cam"], f"{r['ts']:.1f}.jpg")


def live_names(z):
    out = []
    for n in z.names():
        if not re.search(r"\.(jpe?g|png)$", n, re.I):
            continue
        ts = unix_from_name(n)
        if ts is not None:
            out.append((ts, n))
    return sorted(out)


def cmd_live(args):
    os.makedirs(os.path.join(args.out, "live"), exist_ok=True)
    index = []
    for k in args.cams.split(","):
        cam = f"cam_{k}"
        z = remote(args.live.format(k=k))
        names = live_names(z)
        if not names:
            sys.exit(f"[mmcows] {cam}: no frame names with a readable time - run `inspect --zip-url ...`")
        pick, last = [], -1e18
        for ts, n in names:
            if in_window(ts, args) and ts - last >= args.live_step - 0.25:
                last = ts
                pick.append((ts, n))
        todo = [(ts, n) for ts, n in pick if not os.path.exists(live_path(args.out, cam, ts))]
        print(f"[mmcows] {cam}: {len(names)} frames in the zip, {len(pick)} every {args.live_step} s in the "
              f"window, {len(todo)} to download", flush=True)
        t0 = time.time()
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for j, _ in enumerate(ex.map(lambda x: save_small(z.read(x[1]), live_path(args.out, cam, x[0]),
                                                              args.max_side), todo), 1):
                if j % 100 == 0:
                    print(f"\r  {j}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", end="", flush=True)
        print(flush=True)
        index += [{"cam": cam, "ts": ts, "path": live_path(args.out, cam, ts)} for ts, _ in pick]
    with open(os.path.join(args.out, "live.jsonl"), "w", encoding="utf-8") as fh:
        for r in sorted(index, key=lambda r: (r["ts"], r["cam"])):
            fh.write(json.dumps(r) + "\n")


def live_path(out, cam, ts):
    return os.path.join(out, "live", cam, f"{ts:.1f}.jpg")


def cmd_feats(args):
    from PIL import Image
    from cbvd_bursts import encode
    from common import crop, device
    from model import FrameEncoder
    import scene
    frames = read_jsonl(os.path.join(args.out, "frames.jsonl"))
    if not frames:
        sys.exit("run `frames` first")
    out_npz = os.path.join(args.out, "assoc.npz")
    if os.path.exists(out_npz):
        print(f"[mmcows] feats done already: {out_npz}")
        return
    dev = device()
    enc = FrameEncoder(args.encoder, args.grid).to(dev).eval()
    box_rows, box_feats, scenes = [], [], []
    t0 = time.time()
    for s in range(0, len(frames), 16):
        part = frames[s:s + 16]
        imgs = [Image.open(r["path"]).convert("RGB") for r in part]
        scenes.append(scene.scene_grid(enc, imgs))
        crops = []
        for fi, (r, img) in enumerate(zip(part, imgs)):
            a = np.asarray(img)
            for cow, b in r["boxes"]:
                crops.append(crop(a, b, args.crop, args.margin))
                box_rows.append({"frame": s + fi, "cow": cow, "box": b})
        if crops:
            box_feats.append(encode(enc, np.stack(crops)))
        print(f"\r  {min(s + 16, len(frames))}/{len(frames)} frames  {(time.time() - t0) / 60:.1f} min", end="", flush=True)
    print(flush=True)
    np.savez(out_npz + ".part.npz", box_feats=np.concatenate(box_feats).astype(np.float16),
             scene=np.concatenate(scenes).astype(np.float16))
    os.replace(out_npz + ".part.npz", out_npz)
    with open(os.path.join(args.out, "assoc_rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in box_rows:
            fh.write(json.dumps(r) + "\n")
    write_json(os.path.join(args.out, "assoc_meta.json"), {"encoder": args.encoder, "grid": args.grid,
                                                           "crop": args.crop, "margin": args.margin,
                                                           "dim": int(enc.out_dim), "scene_grid": scene.GRID,
                                                           "scene_dim": int(enc.hidden)})
    print(f"[mmcows] {len(box_rows)} boxes, {len(frames)} scene grids -> {out_npz}", flush=True)


def cmd_idcrops(args):
    """One barn, different cows (idnet.py --split cows): every labelled box outside the test
    window, chained per (cow, camera) into sequences of up to --seq-len frames with no gap
    over 2.5 x --train-every; times in seconds."""
    frames = read_jsonl(os.path.join(args.out, "frames.jsonl"))
    rows = read_jsonl(os.path.join(args.out, "assoc_rows.jsonl"))
    if not rows:
        sys.exit("run `feats` first")
    feats = np.load(os.path.join(args.out, "assoc.npz"))["box_feats"]
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        f = frames[r["frame"]]
        if f["split"] != "train":
            continue
        part = "A" if local_hour(f["ts"]) < args.start else "B"
        by[(part, r["cow"], f["cam"])].append((f["ts"], i))
    out_rows, idx = [], []
    for (part, cow, cam), seen in sorted(by.items()):
        seen.sort()
        t0 = last = None
        count = 0
        for ts, i in seen:
            if last is None or ts - last > 2.5 * args.train_every or count >= args.seq_len:
                t0, count = ts, 0                       # a new sequence
            count += 1
            last = ts
            out_rows.append({"name": f"{cam}/{ts:.1f}/{cow}", "day": part, "cow": cow, "seq": f"{cam}_{int(t0)}",
                             "t": round(ts - t0, 2), "cam": cam})
            idx.append(i)
    d = os.path.join(args.out, "idcrops")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "feats.npy"), feats[np.array(idx, int)].astype(np.float16))
    with open(os.path.join(d, "crops.jsonl"), "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    meta = json.load(open(os.path.join(args.out, "assoc_meta.json"), encoding="utf-8"))
    write_json(os.path.join(d, "meta.json"), {**meta, "source": "MmCows labelled boxes outside the test window",
                                              "parts": {"A": f"before {args.start:g}:00", "B": f"after {args.end:g}:00"}})
    seqs = collections.Counter((r["day"], r["cow"], r["seq"]) for r in out_rows)
    cows = collections.Counter(r["cow"] for r in out_rows)
    print(f"[mmcows] idcrops: {len(out_rows)} crops, {len(seqs)} sequences "
          f"(A {sum(1 for k in seqs if k[0] == 'A')}, B {sum(1 for k in seqs if k[0] == 'B')}), "
          f"{len(cows)} cows ({min(cows.values())}-{max(cows.values())} crops each) -> {d}", flush=True)


def cmd_inspect(args):
    z = remote(args.zip_url)
    names = z.names()
    print(f"[mmcows] {len(names)} members in {args.zip_url}")
    for n in names[:: max(1, len(names) // 15)][:15]:
        ts = unix_from_name(n)
        print(f"   {n}   -> {ts} ({'%.2f h local' % local_hour(ts) if ts else 'no time'})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("labels", "frames", "live", "feats", "idcrops", "inspect"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="/workspace/v14/mmcows")
        s.add_argument("--visual", default=VISUAL)
        s.add_argument("--live", default=LIVE)
        s.add_argument("--start", type=float, default=12.0, help="test window start, local hour")
        s.add_argument("--end", type=float, default=15.0, help="test window end, local hour")
        s.add_argument("--workers", type=int, default=24)
        s.add_argument("--max-side", type=int, default=1920)
        s.add_argument("--train-every", type=float, default=60.0, help="s between kept training frames per camera")
        s.add_argument("--window-every", type=float, default=15.0, help="s between kept test-window frames")
        s.add_argument("--live-step", type=float, default=2.0, help="s between 1-second frames kept")
        s.add_argument("--cams", default="1,2,3,4")
        s.add_argument("--encoder", default="facebook/dinov2-small")
        s.add_argument("--grid", type=int, default=2)
        s.add_argument("--crop", type=int, default=224)
        s.add_argument("--margin", type=float, default=0.1)
        s.add_argument("--zip-url", default=LIVE.format(k=1))
        s.add_argument("--seq-len", type=int, default=12, help="idcrops: frames per sequence at most")
    args = p.parse_args(argv)
    return {"labels": cmd_labels, "frames": cmd_frames, "live": cmd_live, "feats": cmd_feats,
            "idcrops": cmd_idcrops, "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
