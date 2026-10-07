#!/usr/bin/env python3
"""The barn's own footage as training data for identity - the part CBVD-5 cannot teach.

CBVD-5 has no cow ids and one 10 s clip per cow, so the fingerprint only
learns "the same cow 3.5 s later". What the brief wants - gait, the same cow
lying and walking, clean and dirty, Monday and Thursday - needs bursts of one
cow far apart in time. The barn makes them by itself:

  1. cache   (pipeline.py, [training_cache] enabled = true) every N-th burst's
             frame vectors (the encoder's output, 175 x 1920 numbers), its
             track, the gallery's guess and a thumbnail of the cow are kept
             under training_cache/<day>/<camera>/.
  2. sheet   an HTML page per day: one thumbnail per track, with the gallery's
             guess - to see which tracks are the same cow.
  3. merges  (optional, by hand) merges.csv:  identity,track  - tracks that are
             one cow (e.g. her ear tag), across hours or days. Without it a
             track is a cow (the tracker keeps her for minutes to hours:
             lying -> standing -> walking, already more than 7 s).
  4. build   -> a features folder in the CBVD-5 format, every burst with an
             "identity"; train.py --extra-train <folder>/train trains the
             fingerprint so that all bursts of one identity are positives;
             train.py eval --extra-val <folder>/val scores re-identification
             across tracks (and days, with --val-from).

    python herd/barn_dataset.py sheet --cache /workspace/herd/barn/training_cache --out sheet.html
    python herd/barn_dataset.py build --cache /workspace/herd/barn/training_cache \\
        --out /workspace/herd/features_barn --merges merges.csv --val-from 2026-11-01
    python herd/herd.py train --features /workspace/herd/features --extra-train /workspace/herd/features_barn/train ...
    python herd/herd.py eval  --features /workspace/herd/features --out RUN --extra-val /workspace/herd/features_barn/val

--gallery-labels 1 also joins tracks the gallery confirmed as one cow (2/3 of
their bursts): self-training - more data, but the gallery's mistakes are
learnt too; use it only once the NaN cut-off is strict.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import io
import json
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_FRAMES, FPS = 175, 25.0
CHUNK = 200                       # bursts per pseudo-clip file
FIRST_CLIP = 900000               # barn pseudo-clip ids, clear of CBVD-5's


# ------------------------------------------------------------------ cache

class TrainingCache:
    """Written by the pipeline: every every_n-th burst's frame vectors + its
    decision + a thumbnail, until max_gb is used. Thread-safe."""

    def __init__(self, cfg, features_meta):
        c = cfg["training_cache"]
        self.folder = c["folder"] if os.path.isabs(c["folder"]) else os.path.join(
            os.path.dirname(os.path.abspath(cfg["store"]["path"])), c["folder"])
        self.every_n, self.max_bytes = max(1, int(c["every_n"])), float(c["max_gb"]) * 2 ** 30
        self.tz = dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))
        self.lock = threading.Lock()
        self.n = 0
        os.makedirs(self.folder, exist_ok=True)
        with open(os.path.join(self.folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(features_meta, fh, indent=2)
        self.used = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(self.folder) for f in fs)
        self.full = False

    def add(self, cam, track, ts, feats, times, valid, motion, pos, thumb, row):
        with self.lock:
            self.n += 1
            if self.n % self.every_n or self.full:
                return
            if self.used >= self.max_bytes:
                self.full = True
                print(f"[cache] {self.folder} reached {self.max_bytes / 2 ** 30:.0f} GB - no more bursts kept",
                      flush=True)
                return
            day = dt.datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d")
            folder = os.path.join(self.folder, day, cam)
            os.makedirs(folder, exist_ok=True)
            name = f"{ts:.0f}_{track.replace('/', '_')}"
            path = os.path.join(folder, name + ".npz")
            np.savez(path, feats=np.asarray(feats, np.float16), times=np.asarray(times, np.float32),
                     valid=np.asarray(valid, bool), pos=np.asarray(pos, np.float32),
                     motion=np.asarray(motion if motion is not None else [], np.float32))
            from PIL import Image
            Image.fromarray(thumb).resize((112, 112)).save(os.path.join(folder, name + ".jpg"), quality=80)
            rec = {"file": os.path.relpath(path, self.folder), "thumb": os.path.relpath(path[:-4] + ".jpg", self.folder),
                   "day": day, "cam": cam, "track": track, "ts": ts, **row}
            with open(os.path.join(self.folder, day, "index.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            self.used += os.path.getsize(path) + 6000


def read_cache(folder):
    rows = []
    for day in sorted(os.listdir(folder)):
        p = os.path.join(folder, day, "index.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh if line.strip()]
    return rows


# --------------------------------------------------------------- identities

def identities(rows, merges=None, gallery_labels=False):
    """track -> identity: a hand-made merge first, then (optionally) the
    gallery's confirmed majority, else the track itself."""
    ident = {r["track"]: f"track:{r['track']}" for r in rows}
    if gallery_labels:
        votes = collections.defaultdict(collections.Counter)
        n = collections.Counter()
        for r in rows:
            n[r["track"]] += 1
            if r.get("state") == "confirmed" and r.get("cow"):
                votes[r["track"]][r["cow"]] += 1
        for track, v in votes.items():
            cow, k = v.most_common(1)[0]
            if k >= 2 / 3 * n[track]:
                ident[track] = f"gallery:{cow}"
    for track, name in (merges or {}).items():
        if track in ident:
            ident[track] = f"label:{name}"
    return ident


def read_merges(path):
    """merges.csv: identity,track per line (a header line is skipped)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].strip() and row[0].strip().lower() != "identity":
                out[row[1].strip()] = row[0].strip()
    return out


def resample(times, n=N_FRAMES, fps=FPS):
    """Frame indices of a burst at the camera's own rate -> n frames on a 25 fps grid."""
    times = np.asarray(times, float)
    grid = np.arange(n) / fps
    return np.clip(np.searchsorted(times, grid), 0, len(times) - 1)


def write_split(rows, ident, cache, out_dir):
    meta = json.load(open(os.path.join(cache, "meta.json"), encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for c, start in enumerate(range(0, len(rows), CHUNK)):
        clip = str(FIRST_CLIP + c)
        chunk = rows[start:start + CHUNK]
        feats, times, motion, ok = [], [], [], []
        for i, r in enumerate(chunk):
            z = np.load(os.path.join(cache, r["file"]))
            idx = resample(z["times"])
            feats.append(z["feats"][idx])
            times.append((np.arange(N_FRAMES) / FPS).astype(np.float32))
            m = z["motion"]
            if len(m):
                motion.append(m)
                ok.append(True)
            else:
                motion.append(None)
                ok.append(False)
            index.append({"kind": "burst", "clip": clip, "i": i, "track": r["track"], "identity": ident[r["track"]],
                          "posture": -1, "activity": -1, "rumination": -1, "n_keyframes": 0,
                          "area": float(r.get("area", 0.05)), "start": 0.0, "uids": [],
                          "pos": [float(v) for v in z["pos"]], "day": r["day"], "cam": r["cam"]})
        np.savez(os.path.join(out_dir, f"{clip}.npz"), burst_feats=np.stack(feats), burst_times=np.stack(times),
                 key_feats=np.zeros((0, meta["dim"]), np.float16))
        dims = [len(m) for m in motion if m is not None]
        if dims:
            D = dims[0]
            np.savez(os.path.join(out_dir, f"{clip}.motion.npz"),
                     motion=np.stack([m if m is not None else np.zeros(D, np.float32) for m in motion]),
                     ok=np.array(ok, bool))
        index.append({"kind": "clip_done", "clip": clip})
    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as fh:
        for r in index:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({**meta, "burst_seconds": N_FRAMES / FPS, "fps": FPS, "source": "barn"}, fh, indent=2)


def cmd_build(args):
    rows = read_cache(args.cache)
    if not rows:
        sys.exit(f"no cached bursts under {args.cache} - is [training_cache] enabled in the barn config?")
    ident = identities(rows, read_merges(args.merges) if args.merges else None, bool(args.gallery_labels))
    count = collections.Counter(ident[r["track"]] for r in rows)
    rows = [r for r in rows if count[ident[r["track"]]] >= args.min_bursts]
    parts = {"train": [r for r in rows if not args.val_from or r["day"] < args.val_from],
             "val": [r for r in rows if args.val_from and r["day"] >= args.val_from]}
    for name, part in parts.items():
        if part:
            write_split(part, ident, args.cache, os.path.join(args.out, name))
            ids = collections.Counter(ident[r["track"]] for r in part)
            print(f"[barn] {name}: {len(part)} bursts, {len(ids)} identities "
                  f"({sum(1 for k in ids if k.startswith('label:'))} labelled by hand, "
                  f"{sum(1 for k in ids if k.startswith('gallery:'))} from the gallery), "
                  f"days {min(r['day'] for r in part)} - {max(r['day'] for r in part)} -> {os.path.join(args.out, name)}")


def cmd_sheet(args):
    """An HTML page: per day, one row per track - thumbnails, camera, time, the gallery's guess."""
    rows = read_cache(args.cache)
    if args.day:
        rows = [r for r in rows if r["day"] == args.day]
    by_track = collections.defaultdict(list)
    for r in rows:
        by_track[(r["day"], r["track"])].append(r)
    out = io.StringIO()
    out.write("<!doctype html><meta charset=utf-8><title>Tracks</title><style>body{font:14px sans-serif;"
              "background:#fff;color:#111}img{width:84px;height:84px;margin:1px}td{vertical-align:top;"
              "padding:4px;border-bottom:1px solid #ddd}</style><h1>Tracks for merges.csv</h1>"
              "<p>One row per track. Tracks that are the same cow get one line each in merges.csv: "
              "<code>identity,track</code> (identity: her ear tag or any name).</p>")
    day = None
    for (d, track), rs in sorted(by_track.items(), key=lambda kv: (kv[0][0], kv[1][0]["ts"])):
        if d != day:
            out.write(f"{'</table>' if day else ''}<h2>{d}</h2><table>")
            day = d
        guess = collections.Counter(r.get("cow") for r in rs if r.get("state") == "confirmed").most_common(1)
        t0 = dt.datetime.fromtimestamp(rs[0]["ts"]).strftime("%H:%M")
        imgs = "".join(f"<img src='{os.path.relpath(os.path.join(args.cache, r['thumb']), os.path.dirname(os.path.abspath(args.out)))}'>"
                       for r in rs[:12])
        out.write(f"<tr><td><b>{track}</b><br>{rs[0]['cam']} {t0}<br>{len(rs)} bursts<br>gallery: "
                  f"{guess[0][0] if guess else '-'}</td><td>{imgs}</td></tr>")
    out.write("</table>" if day else "<p>No cached bursts.</p>")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(out.getvalue())
    print(f"[barn] {len(by_track)} tracks -> {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--merges", default=None, help="identity,track per line")
    b.add_argument("--gallery-labels", type=int, default=0)
    b.add_argument("--min-bursts", type=int, default=2, help="identities with fewer bursts are left out")
    b.add_argument("--val-from", default=None, help="YYYY-MM-DD: days from it on go to val (across-day re-ID)")
    s = sub.add_parser("sheet")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", default="tracks.html")
    s.add_argument("--day", default=None)
    args = p.parse_args(argv)
    {"build": cmd_build, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
