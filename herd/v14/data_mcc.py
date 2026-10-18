#!/usr/bin/env python3
"""MultiCamCows2024 -> DINOv2 vectors of every cow crop, grouped by day, cow and sequence.

The dataset (data.bris.ac.uk, doi 10.5523/bris.2inu67jru7a6821kkgehxg3cv2, UK Non-Commercial
Government Licence) is one 36.5 GB zip: MultiCamCows2024Root/<day>/<cow>/... torso crops
(5 Hz tracklets from 3 ceiling cameras, 14-20 Aug 2023, 90 cows) and the raw videos. Only
the crops are read, member by member through HTTP Range requests (cowbench/extra_det.py's
RemoteZip) - the videos are never downloaded.

Names are read as .../<day: 2023Aug14>/<cow: 001>/<rest>. A sequence is the folder under the
cow if there is one, else the file name without its trailing frame number; a frame's time
is that number / 5 Hz (or its order in the sequence). `inspect` prints how the names were
read - run it first on a new copy of the dataset.

    python herd/v14/data_mcc.py inspect
    python herd/v14/data_mcc.py fetch --out /workspace/v14/mcc            (GPU: DINOv2 on every crop)
    python herd/v14/data_mcc.py fetch --out /workspace/v14/mcc --zip /path/MultiCamCows2024.zip

Writes <out>/feats.npy (n, 1920) fp16, <out>/crops.jsonl (day, cow, seq, t, cam, name per
row), <out>/meta.json, <out>/samples.png (a few crops per day, to look at).
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
from v14common import read_jsonl, write_json

URL = "https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip"
IMG = re.compile(r"(?:^|/)(20\d\d[A-Z][a-z]{2}\d\d)/(\d{1,4})/(.+\.(?:jpe?g|png|bmp))$", re.I)
FPS = 5.0


def parse(names):
    """Zip member names -> rows {name, day, cow, seq, frame, cam} for the crop images."""
    rows = []
    for n in names:
        m = IMG.search(n.replace("\\", "/"))
        if not m or "/videos/" in n.lower():
            continue
        day, cow, rest = m.group(1), int(m.group(2)), m.group(3)
        sub, base = os.path.split(rest)
        stem = os.path.splitext(base)[0]
        num = re.search(r"(\d+)(?!.*\d)", stem)
        frame = int(num.group(1)) if num else None
        seq = sub if sub else (stem[:num.start()].rstrip("_-. ") if num else "all") or "all"
        cam = re.search(r"cam(?:era)?[_-]?(\d)", rest, re.I)
        rows.append({"name": n, "day": day, "cow": cow, "seq": seq, "frame": frame,
                     "cam": int(cam.group(1)) if cam else None})
    # times: the frame number at 5 Hz where the sequence's numbers are distinct, else the order
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["day"], r["cow"], r["seq"])].append(r)
    for rs in by.values():
        nums = [r["frame"] for r in rs]
        if None in nums or len(set(nums)) < len(nums) or max(nums) - min(nums) > 50 * len(nums):
            rs.sort(key=lambda r: r["name"])
            for i, r in enumerate(rs):
                r["t"] = i / FPS
        else:
            lo = min(nums)
            for r in rs:
                r["t"] = (r["frame"] - lo) / FPS
    return rows


def open_source(args):
    import extra_det
    if args.zip:
        return extra_det.LocalZip(args.zip)
    return extra_det.RemoteZip(args.url)


def cmd_inspect(args):
    z = open_source(args)
    names = z.names()
    rows = parse(names)
    print(f"[mcc] {len(names)} members, {len(rows)} read as crops")
    other = collections.Counter(os.path.splitext(n)[1].lower() for n in names if not IMG.search(n.replace('\\', '/')))
    print(f"[mcc] not crops (by extension): {dict(other.most_common(8))}")
    by_day = collections.Counter(r["day"] for r in rows)
    print(f"[mcc] crops by day: {dict(sorted(by_day.items()))}")
    cows = collections.defaultdict(set)
    seqs = collections.Counter()
    for r in rows:
        cows[r["day"]].add(r["cow"])
        seqs[(r["day"], r["cow"], r["seq"])] += 1
    print(f"[mcc] cows by day: {({d: len(c) for d, c in sorted(cows.items())})}, all {len({r['cow'] for r in rows})}")
    lens = np.array(list(seqs.values()))
    if len(lens):
        print(f"[mcc] sequences {len(lens)}, crops per sequence p10/p50/p90 "
              f"{np.percentile(lens, 10):.0f}/{np.percentile(lens, 50):.0f}/{np.percentile(lens, 90):.0f}")
    print(f"[mcc] cameras read from names: {collections.Counter(r['cam'] for r in rows).most_common(5)}")
    for r in rows[:: max(1, len(rows) // 12)][:12]:
        print("   ", {k: r[k] for k in ("name", "day", "cow", "seq", "t", "cam")})


def cmd_fetch(args):
    import torch
    from PIL import Image
    from cbvd_bursts import encode
    from common import crop, device
    from model import FrameEncoder
    os.makedirs(args.out, exist_ok=True)
    done_path = os.path.join(args.out, "feats.npy")
    if os.path.exists(done_path) and os.path.exists(os.path.join(args.out, "crops.jsonl")):
        print(f"[mcc] done already: {args.out}")
        return
    z = open_source(args)
    rows = parse(z.names())
    if not rows:
        sys.exit("[mcc] no crops found in the zip - run `inspect` and adjust the name pattern")
    # at most --max-per-seq crops per sequence, evenly spread (5 Hz neighbours are near-duplicates)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["day"], r["cow"], r["seq"])].append(r)
    keep = []
    for rs in by.values():
        rs.sort(key=lambda r: r["t"])
        if len(rs) > args.max_per_seq:
            rs = [rs[i] for i in np.linspace(0, len(rs) - 1, args.max_per_seq).round().astype(int)]
        keep += rs
    print(f"[mcc] {len(rows)} crops, keeping {len(keep)} ({len(by)} sequences)", flush=True)
    dev = device()
    enc = FrameEncoder(args.encoder, args.grid).to(dev).eval()
    part_dir = os.path.join(args.out, "_parts")
    os.makedirs(part_dir, exist_ok=True)
    samples = collections.defaultdict(list)
    t0 = time.time()
    chunk = 2048
    for c0 in range(0, len(keep), chunk):
        part = os.path.join(part_dir, f"{c0:08d}.npy")
        if os.path.exists(part):
            continue
        sel = keep[c0:c0 + chunk]

        def load(r):
            img = Image.open(io.BytesIO(z.read(r["name"]))).convert("RGB")
            return crop(np.asarray(img), [0.0, 0.0, 1.0, 1.0], args.crop, 0.0), img

        with cf.ThreadPoolExecutor(args.workers) as ex:
            got = list(ex.map(load, sel))
        for r, (_, img) in zip(sel, got):
            if len(samples[r["day"]]) < 8 and r["t"] == 0:
                samples[r["day"]].append(img.copy())
        crops = np.stack([g[0] for g in got])
        np.save(part + ".tmp.npy", encode(enc, crops))
        os.replace(part + ".tmp.npy", part)
        el = time.time() - t0
        n = min(c0 + chunk, len(keep))
        print(f"\r  {n}/{len(keep)} crops  {el / 60:.1f} min", end="", flush=True)
    print(flush=True)
    parts = [np.load(os.path.join(part_dir, f"{c0:08d}.npy")) for c0 in range(0, len(keep), chunk)]
    np.save(done_path + ".tmp.npy", np.concatenate(parts).astype(np.float16))
    with open(os.path.join(args.out, "crops.jsonl"), "w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps({k: r[k] for k in ("name", "day", "cow", "seq", "t", "cam")}) + "\n")
    os.replace(done_path + ".tmp.npy", done_path)
    write_json(os.path.join(args.out, "meta.json"), {"encoder": args.encoder, "grid": args.grid, "crop": args.crop,
                                                     "margin": 0.0, "dim": int(enc.out_dim), "source": "MultiCamCows2024",
                                                     "url": args.url, "max_per_seq": args.max_per_seq})
    sheet(samples, os.path.join(args.out, "samples.png"))
    print(f"[mcc] {len(keep)} crops -> {args.out}", flush=True)


def sheet(samples, path, tile=160):
    from PIL import Image, ImageDraw
    days = sorted(samples)
    if not days:
        return
    cols = max(len(v) for v in samples.values())
    out = Image.new("RGB", (cols * tile, len(days) * (tile + 14)), (25, 25, 25))
    for r, d in enumerate(days):
        for c, img in enumerate(samples[d]):
            im = img.copy()
            im.thumbnail((tile, tile))
            out.paste(im, (c * tile, r * (tile + 14) + 14))
        ImageDraw.Draw(out).text((3, r * (tile + 14)), d, fill=(255, 255, 0))
    out.save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "fetch"):
        s = sub.add_parser(name)
        s.add_argument("--url", default=URL)
        s.add_argument("--zip", default=None, help="the zip already on disk")
        if name == "fetch":
            s.add_argument("--out", required=True)
            s.add_argument("--max-per-seq", type=int, default=48)
            s.add_argument("--encoder", default="facebook/dinov2-small")
            s.add_argument("--grid", type=int, default=2)
            s.add_argument("--crop", type=int, default=224)
            s.add_argument("--workers", type=int, default=32)
    args = p.parse_args(argv)
    return {"inspect": cmd_inspect, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
