#!/usr/bin/env python3
"""v14 (F): the pictures themselves, not DINOv2 vectors - the same cattle version 14 (D) used, for OpenCowID.

    wr10k    WildlifeReID-10k's cattle, exactly data_wr10k.py's plan (the same sets, cows, held-out cows and
             sequences): each image -> a 224 px square (the whole image, common.crop as v14 D) ->
             <out>/images/<n>.jpg, <out>/crops.jsonl (data_wr10k.py's rows + "image"), meta.json, samples.png
    mmcows   MmCows' labelled boxes outside the test window, data_mmcows.py idcrops' sequences (per cow and
             camera; part A before the window, B after), cut from the saved labelled frames with the barn's crop
             (224 px, margin 0.1) -> the same layout
    combine  the two -> one folder of rows (the pictures stay where they are): train / valA / valB (WildlifeReID-10k)
             and testA / testB (MmCows), for opencowid.py

    python herd/v14/data_images.py wr10k --out /workspace/v14/wr10k_img
    python herd/v14/data_images.py mmcows --mm /workspace/v14/mmcows --out /workspace/v14/mmcows_img
    python herd/v14/data_images.py combine --wr /workspace/v14/wr10k_img --mm /workspace/v14/mmcows_img --out /workspace/v14/iddi

WildlifeReID-10k: non-commercial, no re-upload of its images, every source set attributed. MmCows: CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import io
import json
import os
import sys
import time

import numpy as np

import v14common as C  # noqa: F401  (paths)
from v14common import local_hour, read_jsonl, write_json


def image_path(n):
    return os.path.join("images", f"{n:07d}.jpg")


def save_jpeg(arr, path, quality=92):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.asarray(arr)).save(path + ".part.jpg", quality=quality)
    os.replace(path + ".part.jpg", path)


def write_rows(out, rows, meta):
    with open(os.path.join(out, "crops.jsonl.part"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(os.path.join(out, "crops.jsonl.part"), os.path.join(out, "crops.jsonl"))
    write_json(os.path.join(out, "meta.json"), meta)


def cmd_wr10k(args):
    from PIL import Image
    from common import crop
    import data_wr10k as W
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "crops.jsonl")):
        print(f"[images] wr10k done already: {args.out}")
        return
    z = W.open_zip(args.url)
    keep = W.plan(W.read_metadata(z), args.sets, args.min_images, args.val_share, args.seq_len, args.spacing, args.seed)
    W.summary(keep)
    rows = [{**r, "image": image_path(i)} for i, r in enumerate(keep)]
    todo = [r for r in rows if not os.path.exists(os.path.join(args.out, r["image"]))]
    print(f"[images] {len(rows)} images, {len(todo)} to fetch", flush=True)
    t0 = time.time()

    def one(r):
        img = Image.open(io.BytesIO(z.read(r["name"])))
        img.draft("RGB", (2 * args.crop, 2 * args.crop))
        img = img.convert("RGB")
        img.thumbnail((4 * args.crop, 4 * args.crop))
        save_jpeg(crop(np.asarray(img), [0.0, 0.0, 1.0, 1.0], args.crop, 0.0), os.path.join(args.out, r["image"]))

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(one, todo), 1):
            if k % 200 == 0:
                print(f"\r  {k}/{len(todo)} images  {(time.time() - t0) / 60:.1f} min", end="", flush=True)
    print(flush=True)
    samples = collections.defaultdict(list)
    for r in rows:
        if len(samples[r["dataset"]]) < 8 and r["t"] == 0:
            samples[r["dataset"]].append(Image.open(os.path.join(args.out, r["image"])))
    import data_mcc
    data_mcc.sheet(samples, os.path.join(args.out, "samples.png"))
    write_rows(args.out, rows, {
        "crop": args.crop, "margin": 0.0, "source": "WildlifeReID-10k cattle (pictures)", "sets": args.sets, "url": args.url,
        "min_images": args.min_images, "val_share": args.val_share, "seq_len": args.seq_len, "spacing_s": args.spacing,
        "seed": args.seed, "licence": "non-commercial, no re-upload, attribute every source set (WildlifeReID-10k citation folder)"})
    print(f"[images] {len(rows)} images -> {args.out}", flush=True)


def cmd_mmcows(args):
    from PIL import Image
    from common import crop
    import data_mmcows as M
    frames = read_jsonl(os.path.join(args.mm, "frames.jsonl"))
    if not frames:
        sys.exit("run data_mmcows.py labels + frames first")
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "crops.jsonl")):
        print(f"[images] mmcows done already: {args.out}")
        return
    items = [(fi, cow, b) for fi, f in enumerate(frames) for cow, b in f["boxes"]]
    rows, idx = M.id_sequences(frames, items, args.start, args.train_every, args.seq_len)
    rows = [{**r, "image": image_path(i)} for i, r in enumerate(rows)]
    by_frame = collections.defaultdict(list)
    for r, i in zip(rows, idx):
        by_frame[items[i][0]].append((r, items[i][2]))
    print(f"[images] {len(rows)} crops from {len(by_frame)} labelled frames", flush=True)
    t0 = time.time()
    for k, fi in enumerate(sorted(by_frame), 1):
        todo = [(r, b) for r, b in by_frame[fi] if not os.path.exists(os.path.join(args.out, r["image"]))]
        if todo:
            a = np.asarray(Image.open(frames[fi]["path"]).convert("RGB"))
            for r, b in todo:
                save_jpeg(crop(a, b, args.crop, args.margin), os.path.join(args.out, r["image"]))
        if k % 100 == 0:
            print(f"\r  {k}/{len(by_frame)} frames  {(time.time() - t0) / 60:.1f} min", end="", flush=True)
    print(flush=True)
    write_rows(args.out, rows, {"crop": args.crop, "margin": args.margin, "source": "MmCows labelled boxes outside the test window (pictures)",
                                "parts": {"A": f"before {args.start:g}:00", "B": f"after {args.end:g}:00"},
                                "seq_len": args.seq_len, "train_every": args.train_every})
    seqs = collections.Counter((r["day"], r["cow"], r["seq"]) for r in rows)
    cows = collections.Counter(r["cow"] for r in rows)
    print(f"[images] mmcows: {len(rows)} crops, {len(seqs)} sequences, {len(cows)} cows -> {args.out}", flush=True)


def cmd_combine(args):
    wm = json.load(open(os.path.join(args.wr, "meta.json"), encoding="utf-8"))
    mm = json.load(open(os.path.join(args.mm, "meta.json"), encoding="utf-8"))
    if wm.get("crop") != mm.get("crop"):
        sys.exit(f"[images] the two folders differ in crop size: {wm.get('crop')} vs {mm.get('crop')}")
    absolute = lambda folder, r: {**r, "image": os.path.abspath(os.path.join(folder, r["image"]))}
    rows = [absolute(args.wr, r) for r in read_jsonl(os.path.join(args.wr, "crops.jsonl"))]
    rows += [absolute(args.mm, {**r, "day": "test" + r["day"], "dataset": "MmCows"}) for r in read_jsonl(os.path.join(args.mm, "crops.jsonl"))]
    os.makedirs(args.out, exist_ok=True)
    write_rows(args.out, rows, {**wm, "source": "WildlifeReID-10k cattle (train, val) + MmCows labelled boxes (test, cows never "
                                                "trained on) - pictures", "mmcows": mm})
    days = collections.Counter(r["day"] for r in rows)
    print(f"[images] combined: {dict(days)}; test cows {sorted({r['cow'] for r in rows if r['day'].startswith('test')})} -> {args.out}", flush=True)


def main(argv=None):
    import data_wr10k as W
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("wr10k")
    s.add_argument("--out", required=True)
    s.add_argument("--url", default=W.URL)
    s.add_argument("--sets", default=W.SETS)
    s.add_argument("--min-images", type=int, default=4)
    s.add_argument("--val-share", type=float, default=0.1)
    s.add_argument("--seq-len", type=int, default=8)
    s.add_argument("--spacing", type=float, default=4.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--crop", type=int, default=224)
    s.add_argument("--workers", type=int, default=32)
    s = sub.add_parser("mmcows")
    s.add_argument("--mm", required=True, help="data_mmcows.py's folder (frames.jsonl, frames/)")
    s.add_argument("--out", required=True)
    s.add_argument("--start", type=float, default=12.0)
    s.add_argument("--end", type=float, default=15.0)
    s.add_argument("--train-every", type=float, default=60.0)
    s.add_argument("--seq-len", type=int, default=12)
    s.add_argument("--crop", type=int, default=224)
    s.add_argument("--margin", type=float, default=0.1)
    s = sub.add_parser("combine")
    s.add_argument("--wr", required=True)
    s.add_argument("--mm", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"wr10k": cmd_wr10k, "mmcows": cmd_mmcows, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
