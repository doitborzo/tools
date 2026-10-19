#!/usr/bin/env python3
"""v14 (D): the cattle of WildlifeReID-10k -> DINOv2 vectors, for the ID transformer.

WildlifeReID-10k (kaggle.com/datasets/wildlifedatasets/wildlifereid-10k; non-commercial, no
re-upload, every source dataset attributed - see its citation folder) gathers 37 animal re-id
sets in one 26 GB zip, with metadata.csv (identity, path, date, split, dataset). Its cattle are
read member by member through HTTP Range requests (cowbench/extra_det.py's RemoteZip, 2.2 GB of
the 26), nothing downloaded whole and no Kaggle account needed.

Sets used (--sets): Cows2021 (179 cows), MultiCamCows2024 (90, every 20th frame), OpenCows2020
(46), CowDataset (13, side view). Left out: AerialCattle2017 and FriesianCattle2015 (already
inside OpenCows2020: one cow under two names) and FriesianCattle2017 (whole frames with several
cows under one label).

    plan     cows with >= --min-images images; --val-share of each set's cows held out for picking
             the epoch, the "not sure" model and the enrolment threshold (album = their earlier
             half, queries = the later half); each cow's images in date order, cut into
             sequences of <= --seq-len (one image every --spacing s)
    fetch    images -> 224 px crop of the whole image -> DINOv2 -> <out>/feats.npy, crops.jsonl
             {name, day: train | valA | valB, cow, seq, t, cam, dataset}, meta.json, samples.png
    combine  + MmCows' idcrops (data_mmcows.py idcrops) as the test: part A = testA (album),
             part B = testB (queries) -> one folder for idnet.py --split groups

    python herd/v14/data_wr10k.py fetch --out /workspace/v14/wr10k
    python herd/v14/data_wr10k.py combine --wr /workspace/v14/wr10k --mm /workspace/v14/mmcows/idcrops \\
        --out /workspace/v14/idd
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import io
import json
import os
import random
import sys
import time

import numpy as np

import v14common as C  # noqa: F401  (paths)
from v14common import read_jsonl, write_json

URL = "https://www.kaggle.com/api/v1/datasets/download/wildlifedatasets/wildlifereid-10k"
SETS = "Cows2021,MultiCamCows2024,OpenCows2020,CowDataset"


def plan(rows, sets, min_images=4, val_share=0.1, seq_len=8, spacing=4.0, seed=0):
    """metadata.csv rows -> the images kept, each {name, day, cow, seq, t, cam, dataset}."""
    keep_sets = [s.strip() for s in sets.split(",") if s.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        if r["dataset"] in keep_sets:
            by[(r["dataset"], r["identity"])].append(r)
    by = {k: v for k, v in by.items() if len(v) >= min_images}
    rng = random.Random(seed)
    val = set()
    for ds in keep_sets:
        cows = sorted(k for k in by if k[0] == ds)
        n = max(1, int(round(val_share * len(cows)))) if len(cows) >= 4 else 0
        val |= set(rng.sample(cows, n))
    out = []
    for (ds, cow), imgs in sorted(by.items()):
        imgs.sort(key=lambda r: (r.get("date") or "", r["path"]))
        parts = ([("valA", imgs[:len(imgs) // 2]), ("valB", imgs[len(imgs) // 2:])] if (ds, cow) in val
                 else [("train", imgs)])
        for day, part in parts:
            for i, r in enumerate(part):
                out.append({"name": r["path"], "day": day, "cow": cow, "seq": f"{day}_{i // seq_len}",
                            "t": (i % seq_len) * spacing, "cam": None, "dataset": ds})
    return out


def open_zip(url):
    import extra_det
    return extra_det.RemoteZip(url)


def read_metadata(z):
    return list(csv.DictReader(io.StringIO(z.read("metadata.csv").decode("utf-8"))))


def summary(keep):
    s = collections.defaultdict(lambda: collections.Counter())
    cows = collections.defaultdict(set)
    for r in keep:
        s[r["dataset"]][r["day"]] += 1
        cows[(r["dataset"], r["day"])].add(r["cow"])
    for ds in sorted(s):
        print(f"[wr10k] {ds:18s} " + "  ".join(f"{d} {n} images / {len(cows[(ds, d)])} cows"
                                               for d, n in sorted(s[ds].items())), flush=True)


def cmd_plan(args):
    z = open_zip(args.url)
    keep = plan(read_metadata(z), args.sets, args.min_images, args.val_share, args.seq_len, args.spacing, args.seed)
    summary(keep)
    print(f"[wr10k] {len(keep)} images, {len({r['cow'] for r in keep if r['day'] == 'train'})} training cows, "
          f"{len({r['cow'] for r in keep if r['day'] != 'train'})} held-out cows", flush=True)


def cmd_fetch(args):
    from PIL import Image
    from cbvd_bursts import encode
    from common import crop, device
    from model import FrameEncoder
    import data_mcc
    os.makedirs(args.out, exist_ok=True)
    done = os.path.join(args.out, "feats.npy")
    if os.path.exists(done) and os.path.exists(os.path.join(args.out, "crops.jsonl")):
        print(f"[wr10k] done already: {args.out}")
        return
    z = open_zip(args.url)
    keep = plan(read_metadata(z), args.sets, args.min_images, args.val_share, args.seq_len, args.spacing, args.seed)
    summary(keep)
    dev = device()
    enc = FrameEncoder(args.encoder, args.grid).to(dev).eval()
    part_dir = os.path.join(args.out, "_parts")
    os.makedirs(part_dir, exist_ok=True)
    samples = collections.defaultdict(list)
    t0, chunk = time.time(), 1024

    def load(r):
        img = Image.open(io.BytesIO(z.read(r["name"])))
        img.draft("RGB", (2 * args.crop, 2 * args.crop))      # big JPEGs (CowDataset: 4000 px) decoded small
        img = img.convert("RGB")
        img.thumbnail((4 * args.crop, 4 * args.crop))
        return crop(np.asarray(img), [0.0, 0.0, 1.0, 1.0], args.crop, 0.0), img

    for c0 in range(0, len(keep), chunk):
        part = os.path.join(part_dir, f"{c0:08d}.npy")
        if os.path.exists(part):
            continue
        sel = keep[c0:c0 + chunk]
        with cf.ThreadPoolExecutor(args.workers) as ex:
            got = list(ex.map(load, sel))
        for r, (_, img) in zip(sel, got):
            if len(samples[r["dataset"]]) < 8 and r["t"] == 0:
                samples[r["dataset"]].append(img.copy())
        np.save(part + ".tmp.npy", encode(enc, np.stack([g[0] for g in got])))
        os.replace(part + ".tmp.npy", part)
        print(f"\r  {min(c0 + chunk, len(keep))}/{len(keep)} images  {(time.time() - t0) / 60:.1f} min", end="", flush=True)
    print(flush=True)
    parts = [np.load(os.path.join(part_dir, f"{c0:08d}.npy")) for c0 in range(0, len(keep), chunk)]
    np.save(done + ".tmp.npy", np.concatenate(parts).astype(np.float16))
    with open(os.path.join(args.out, "crops.jsonl"), "w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    os.replace(done + ".tmp.npy", done)
    write_json(os.path.join(args.out, "meta.json"), {
        "encoder": args.encoder, "grid": args.grid, "crop": args.crop, "margin": 0.0, "dim": int(enc.out_dim),
        "source": "WildlifeReID-10k cattle", "sets": args.sets, "url": args.url, "min_images": args.min_images,
        "val_share": args.val_share, "seq_len": args.seq_len, "spacing_s": args.spacing, "seed": args.seed,
        "licence": "non-commercial, no re-upload, attribute every source set (WildlifeReID-10k citation folder)"})
    data_mcc.sheet(samples, os.path.join(args.out, "samples.png"))
    print(f"[wr10k] {len(keep)} images -> {args.out}", flush=True)


def cmd_combine(args):
    """WildlifeReID-10k (train / valA / valB) + MmCows idcrops (A -> testA, B -> testB) -> one store."""
    wm = json.load(open(os.path.join(args.wr, "meta.json"), encoding="utf-8"))
    mm = json.load(open(os.path.join(args.mm, "meta.json"), encoding="utf-8"))
    for k in ("encoder", "grid", "crop", "dim"):
        if wm.get(k) != mm.get(k):
            sys.exit(f"[wr10k] the two folders differ in {k}: {wm.get(k)} vs {mm.get(k)}")
    rw, rm = read_jsonl(os.path.join(args.wr, "crops.jsonl")), read_jsonl(os.path.join(args.mm, "crops.jsonl"))
    fw, fm = np.load(os.path.join(args.wr, "feats.npy")), np.load(os.path.join(args.mm, "feats.npy"))
    rows = rw + [{**r, "day": "test" + r["day"], "dataset": "MmCows"} for r in rm]
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "feats.npy"), np.concatenate([fw, fm]).astype(np.float16))
    with open(os.path.join(args.out, "crops.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    write_json(os.path.join(args.out, "meta.json"), {**wm, "source": "WildlifeReID-10k cattle (train, val) + MmCows "
                                                                     "labelled boxes (test, cows never trained on)",
                                                     "mmcows": mm})
    days = collections.Counter(r["day"] for r in rows)
    print(f"[wr10k] combined: {dict(days)}; test cows {sorted({r['cow'] for r in rm})} -> {args.out}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "fetch", "combine"):
        s = sub.add_parser(name)
        if name == "combine":
            s.add_argument("--wr", required=True)
            s.add_argument("--mm", required=True)
            s.add_argument("--out", required=True)
            continue
        s.add_argument("--url", default=URL)
        s.add_argument("--sets", default=SETS)
        s.add_argument("--min-images", type=int, default=4)
        s.add_argument("--val-share", type=float, default=0.1)
        s.add_argument("--seq-len", type=int, default=8)
        s.add_argument("--spacing", type=float, default=4.0, help="s between a sequence's images")
        s.add_argument("--seed", type=int, default=0)
        if name == "fetch":
            s.add_argument("--out", required=True)
            s.add_argument("--encoder", default="facebook/dinov2-small")
            s.add_argument("--grid", type=int, default=2)
            s.add_argument("--crop", type=int, default=224)
            s.add_argument("--workers", type=int, default=32)
    args = p.parse_args(argv)
    return {"plan": cmd_plan, "fetch": cmd_fetch, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
