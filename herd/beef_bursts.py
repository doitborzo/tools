#!/usr/bin/env python3
"""A labelled behaviour set from another barn: Kaggle's Beef Cattle Behavior Data Set
(lucyfirst/beef-cattle-behavior-data-set, licence "Unknown" on Kaggle - personal use).

Its "Category Videos/cows/<behaviour>/*.mp4" are clips of ONE cow each, cut out by the
authors (YOLOv8 + ByteTrack, then FFmpeg), labelled drink / eat / lie / ruminate /
stand. This picks N clips per behaviour at random, downloads only those, and turns
each into one 7 s burst in herd's feature format - the same DINOv2 vectors (one per
frame, e.g. 1920 numbers), frame times and chewing rhythm the tiny transformer trains
on - so train.py can take it next to CBVD-5:

    python herd/beef_bursts.py list                                    # clips per behaviour (no key needed)
    python herd/beef_bursts.py build --out /workspace/herd/features_beef --per-class 75 \\
        --like /workspace/herd/features/train/meta.json                # encoder, grid, crop as CBVD-5's
    python herd/beef_bursts.py sheet --out /workspace/herd/features_beef   # one frame per clip: LOOK at it
    python herd/herd.py train ... --extra-train /workspace/herd/features_beef

Labels (herd's: posture standing/lying, activity feeding/drinking/none, rumination):
    drink     activity drinking, rumination no, posture unknown
    eat       activity feeding,  rumination no, posture unknown
    lie       posture lying,     activity none, rumination no
    ruminate  rumination yes,    activity none, posture unknown
    stand     posture standing,  activity none, rumination no
"unknown" is -1: that head learns nothing from the burst. The classes are the
authors' one label per clip, so "lie" is lying without chewing - an assumption.

Every clip is its own cow (no identities across clips). A clip shorter than the
burst is not stretched: it is replaced by another clip of the same behaviour.
Each clip's frame is used whole as the cow's crop (they are one-cow clips); check
that with `sheet` before training - if the clips turn out to be whole barn frames,
they need the detector first.

Downloads need a Kaggle API key: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and
KAGGLE_KEY, or KAGGLE_API_TOKEN. Or download the clips yourself and pass
--local-dir <folder holding drink/ eat/ lie/ ruminate/ stand/>.
"""

from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ACTIVITIES, POSTURES, crop, read_jsonl  # noqa: E402

DATASET = "lucyfirst/beef-cattle-behavior-data-set"
PREFIX = "Category Videos/cows/"
FPS = 25.0
FIRST_CLIP = 800000          # pseudo-clip ids: clear of CBVD-5's and the barn's (900000)
CHUNK = 200                  # bursts per pseudo-clip file
P, A = {n: i for i, n in enumerate(POSTURES)}, {n: i for i, n in enumerate(ACTIVITIES)}
LABELS = {                   # behaviour -> (posture, activity, rumination); -1 = unknown
    "drink": (-1, A["drinking"], 0),
    "eat": (-1, A["feeding"], 0),
    "lie": (P["lying"], A["none"], 0),
    "ruminate": (-1, A["none"], 1),
    "stand": (P["standing"], A["none"], 0),
}


# ------------------------------------------------------------------ Kaggle

def list_clips(dataset=DATASET):
    """{behaviour: [file path in the dataset]} - the public file list, no key needed."""
    base = f"https://www.kaggle.com/api/v1/datasets/list/{dataset}"
    out, token = collections.defaultdict(list), None
    for _ in range(1000):
        url = base + "?pageSize=1000" + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        d = json.load(urllib.request.urlopen(url, timeout=120))
        for f in d.get("datasetFiles", []):
            name = f["name"]
            if name.startswith(PREFIX) and name.lower().endswith(".mp4"):
                beh = name[len(PREFIX):].split("/")[0].lower()
                out[beh].append(name)
        token = d.get("nextPageTokenNullable") or d.get("nextPageToken")
        if not token:
            break
    return {k: sorted(v) for k, v in out.items()}


def auth_header():
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        path = os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle")), "kaggle.json")
        if os.path.exists(path):
            j = json.load(open(path, encoding="utf-8"))
            user, key = j.get("username"), j.get("key")
    if not (user and key):
        sys.exit("downloads need a Kaggle API key: ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New "
                 "Token), or KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN - or use --local-dir")
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()}


def download(name, dest, headers, dataset=DATASET, tries=4):
    """One file of the dataset -> dest (unzipped if Kaggle sends it zipped)."""
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/{urllib.parse.quote(name)}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
                data = r.read()
            break
        except Exception as e:  # noqa: BLE001 - network: retried, then reported
            if getattr(e, "code", None) in (401, 403):
                sys.exit(f"Kaggle refused the download ({e}): check the API key, and accept the dataset's "
                         f"terms on its Kaggle page if it asks")
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(z.namelist()[0])
    if data[:15].lower().startswith((b"<!doctype", b"<html")):
        raise RuntimeError(f"{name}: got a web page, not the video")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as fh:
        fh.write(data)
    os.replace(dest + ".part", dest)


# ------------------------------------------------------------------ one clip -> one burst

def read_clip(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    cap.release()
    return frames, float(fps)


def burst_crops(frames, fps, seconds, size, margin, rng):
    """A random `seconds` window of the clip at 25 fps -> (T, size, size, 3) crops of the
    whole frame, or None if the clip is shorter. Frames are taken by real time, so a
    clip at another frame rate keeps its true timing."""
    n = int(round(seconds * FPS))
    dur = len(frames) / fps
    if dur + 0.5 / fps < seconds:
        return None
    start = rng.uniform(0.0, max(0.0, dur - seconds))
    idx = [min(len(frames) - 1, int((start + j / FPS) * fps)) for j in range(n)]
    whole = [0.0, 0.0, 1.0, 1.0]
    return np.stack([crop(frames[i], whole, size, margin) for i in idx]), start


def make_encoder(meta):
    from common import device
    from model import FrameEncoder
    return FrameEncoder(meta["encoder"], meta["grid"]).to(device()).eval()


# ------------------------------------------------------------------ commands

def pick(clips, per_class, seed, behaviours):
    """Per behaviour: a seeded shuffle; the first per_class are the picks, the rest replacements."""
    rng = random.Random(seed)
    out = {}
    for b in behaviours:
        names = list(clips.get(b, []))
        rng.shuffle(names)
        out[b] = names
    return out


def cmd_list(args):
    clips = list_clips()
    for b, names in sorted(clips.items()):
        print(f"{b:10s} {len(names):5d} clips" + ("" if b in LABELS else "   (not mapped to herd's labels)"))
    print(f"{sum(len(v) for v in clips.values())} clips under '{PREFIX}'")


def cmd_build(args, encoder_factory=make_encoder, clips=None):
    from cbvd_bursts import encode
    from motion import motion_features
    meta = {"encoder": args.encoder, "grid": args.grid, "crop": args.crop, "margin": args.margin}
    if args.like:
        like = json.load(open(args.like, encoding="utf-8"))
        # the vectors must match CBVD-5's (encoder, grid, crop size); the margin stays this
        # script's: the clip is already cut to the cow, a margin would only add grey border
        meta.update({k: like[k] for k in ("encoder", "grid", "crop") if k in like})
    behaviours = [b for b in args.behaviours.split(",") if b]
    unknown = [b for b in behaviours if b not in LABELS]
    if unknown:
        sys.exit(f"no labels for {unknown}; known: {sorted(LABELS)}")
    work = args.work or os.path.join(args.out, "_work")
    if clips is None:
        if args.local_dir:
            clips = {b: sorted(os.path.join(b, f) for f in os.listdir(os.path.join(args.local_dir, b))
                               if f.lower().endswith(".mp4"))
                     for b in behaviours if os.path.isdir(os.path.join(args.local_dir, b))}
        else:
            clips = list_clips()
    order = pick(clips, args.per_class, args.seed, behaviours)
    headers = None
    enc = None
    rng = random.Random(args.seed)
    n_frames = int(round(args.burst_seconds * FPS))
    made = collections.defaultdict(list)          # behaviour -> [cached burst file]
    skipped = collections.Counter()
    for b in behaviours:
        for name in order[b]:
            if len(made[b]) >= args.per_class:
                break
            stem = os.path.splitext(os.path.basename(name))[0]
            cache = os.path.join(work, "bursts", b, stem + ".npz")
            if os.path.exists(cache):
                made[b].append(cache)
                continue
            if args.local_dir:
                path = os.path.join(args.local_dir, name)
            else:
                path = os.path.join(work, "clips", b, os.path.basename(name))
                if not os.path.exists(path):
                    headers = headers or auth_header()
                    try:
                        download(name, path, headers)
                    except Exception as e:  # noqa: BLE001
                        print(f"\n  {name}: download failed ({e}) - replaced", flush=True)
                        skipped["download failed"] += 1
                        continue
            frames, fps = read_clip(path)
            got = burst_crops(frames, fps, args.burst_seconds, meta["crop"], meta["margin"], rng) if frames else None
            if got is None:
                skipped["shorter than the burst" if frames else "unreadable"] += 1
                continue
            crops, start = got
            enc = enc or encoder_factory(meta)
            feats = encode(enc, crops)
            motion, ok = motion_features(crops, fps=FPS)
            h, w = frames[0].shape[:2]
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache + ".part.npz", feats=feats, motion=motion.astype(np.float32), ok=ok,
                     start=start, fps=fps, seconds=len(frames) / fps, size=np.array([w, h]), name=name)
            os.replace(cache + ".part.npz", cache)
            made[b].append(cache)
            print(f"\r  {b}: {len(made[b])}/{args.per_class}", end="", flush=True)
        print(flush=True)
        if len(made[b]) < args.per_class:
            print(f"  {b}: only {len(made[b])} usable clips (of {len(order[b])})", flush=True)
    if not any(made.values()):
        sys.exit("no burst made")
    write_split(made, meta, n_frames, args.out)
    print(f"[beef] {sum(len(v) for v in made.values())} bursts -> {args.out}  "
          + ", ".join(f"{b} {len(v)}" for b, v in made.items())
          + (f"; skipped: {dict(skipped)}" if skipped else ""))
    print(f"[beef] look first: python {sys.argv[0]} sheet --out {args.out};  "
          f"then: herd.py train ... --extra-train {args.out}")


def write_split(made, meta, n_frames, out_dir):
    """Cached bursts -> herd's feature folder (index.jsonl, <clip>.npz, <clip>.motion.npz, meta.json)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [(b, f) for b in sorted(made) for f in made[b]]
    index, dim = [], None
    for c, s in enumerate(range(0, len(rows), CHUNK)):
        clip = str(FIRST_CLIP + c)
        feats, motion, ok = [], [], []
        for i, (b, path) in enumerate(rows[s:s + CHUNK]):
            z = np.load(path)
            feats.append(z["feats"])
            motion.append(z["motion"])
            ok.append(bool(z["ok"]))
            dim = z["feats"].shape[1]
            posture, activity, rumination = LABELS[b]
            index.append({"kind": "burst", "clip": clip, "i": i, "track": f"beef-{b}-{i}-{c}",
                          "posture": posture, "activity": activity, "rumination": rumination,
                          "n_keyframes": 0, "area": 0.25, "start": float(z["start"]), "uids": [],
                          "pos": [0.5, 0.5, 1.0, 1.0], "source": "beef", "behaviour": b,
                          "file": str(z["name"])})
        np.savez(os.path.join(out_dir, f"{clip}.npz"), burst_feats=np.stack(feats).astype(np.float16),
                 burst_times=np.tile((np.arange(n_frames) / FPS).astype(np.float32), (len(feats), 1)),
                 key_feats=np.zeros((0, dim), np.float16))
        np.savez(os.path.join(out_dir, f"{clip}.motion.npz"), motion=np.stack(motion), ok=np.array(ok, bool))
        index.append({"kind": "clip_done", "clip": clip})
    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as fh:
        for r in index:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({**meta, "dim": int(dim), "burst_seconds": n_frames / FPS, "fps": FPS, "source": "beef",
                   "dataset": DATASET}, fh, indent=2)


def cmd_sheet(args):
    """One frame of a few clips per behaviour on one PNG: each should show one cow."""
    from PIL import Image, ImageDraw
    rows = [r for r in read_jsonl(os.path.join(args.out, "index.jsonl")) if r["kind"] == "burst"]
    work = args.work or os.path.join(args.out, "_work")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["behaviour"]].append(r)
    tile, tiles = 200, []
    for b in sorted(by):
        for r in by[b][:args.n]:
            path = (os.path.join(args.local_dir, r["file"]) if args.local_dir
                    else os.path.join(work, "clips", b, os.path.basename(r["file"])))
            frames, _ = read_clip(path) if os.path.exists(path) else ([], 0)
            img = Image.fromarray(frames[len(frames) // 2]) if frames else Image.new("RGB", (tile, tile), (60, 60, 60))
            img.thumbnail((tile, tile))
            t = Image.new("RGB", (tile, tile + 16), (30, 30, 30))
            t.paste(img, (0, 16))
            ImageDraw.Draw(t).text((3, 2), b, fill=(255, 255, 0))
            tiles.append(t)
    cols = args.n
    sheet = Image.new("RGB", (cols * tile, -(-len(tiles) // cols) * (tile + 16)), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tile, (i // cols) * (tile + 16)))
    out = os.path.join(args.out, "check.png")
    sheet.save(out)
    print(f"[beef] {len(tiles)} clips, one row per behaviour -> {out}  (each tile should be one cow)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("build")
    b.add_argument("--out", required=True, help="the feature folder to write (for --extra-train)")
    b.add_argument("--per-class", type=int, default=75)
    b.add_argument("--behaviours", default="drink,eat,lie,ruminate,stand")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--like", default=None, help="CBVD-5's features meta.json: same encoder, grid, crop size")
    b.add_argument("--encoder", default="facebook/dinov2-small")
    b.add_argument("--grid", type=int, default=2)
    b.add_argument("--crop", type=int, default=224)
    b.add_argument("--margin", type=float, default=0.0, help="the clip is the cow already: no context added")
    b.add_argument("--burst-seconds", type=float, default=7.0)
    b.add_argument("--work", default=None, help="downloads and per-clip cache (default <out>/_work)")
    b.add_argument("--local-dir", default=None, help="clips already on disk: <dir>/<behaviour>/*.mp4")
    s = sub.add_parser("sheet")
    s.add_argument("--out", required=True)
    s.add_argument("--work", default=None)
    s.add_argument("--local-dir", default=None)
    s.add_argument("--n", type=int, default=6, help="clips per behaviour")
    args = p.parse_args(argv)
    {"list": cmd_list, "build": cmd_build, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
