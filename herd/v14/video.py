#!/usr/bin/env python3
"""v14 (A) on video: what the onboarding test saw and said, 4 cameras side by side.

Reads what system.py left in --system (tracks_<tag>.jsonl: every box with its track,
names_<tag>.json: the album and the name the ID desk gave each track each minute) and
draws it on MmCows' 1-second frames (data_mmcows.py live), one video frame per step:

    box colour   one colour per album cow (the same cow keeps its colour on every camera)
    box label    setup hour: the album cow the track was enrolled as (or T<track> if none)
                 after it:   "A03 = cow 7" (the album cow named, and the true cow that
                 album cow stands for), "not sure (A03 = cow 7?)" with its best guess, or
                 "-" (no fingerprint that minute)
    white box    MmCows' label (every 15 s): "cow 7 OK" (named right), "cow 7 X" (wrong), "cow 7 ?"
                 (not named; "?ok" / "?x" when its best guess would have been right / wrong)
    header       time, setup or identification, and the running tally of labelled boxes
    darkened     outside the pen (the other pens, which the detector never looks at)

    python herd/v14/video.py --data /workspace/v14/mmcows --system /workspace/v14/system_video \\
        --from 13:00 --to 13:10 --out /workspace/V14A_VIDEO/test_1300-1310.mp4
"""

from __future__ import annotations

import argparse
import collections
import colorsys
import hashlib
import json
import os
import sys

import numpy as np

import v14common as C
from v14common import iou, local_hour, read_jsonl


def hour_of(text):
    """HH:MM or HH:MM:SS -> hours."""
    p = [int(x) for x in text.split(":")] + [0]
    return p[0] + p[1] / 60.0 + p[2] / 3600.0


def colour(name):
    """A stable bright colour per name (BGR for OpenCV)."""
    if name is None:
        return (150, 150, 150)
    h = int(hashlib.md5(str(name).encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def label(img, text, x, y, fg, bg=(0, 0, 0), scale=0.5):
    import cv2
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(h + 4, y)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + base - 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


class Writer:
    """H.264 through imageio-ffmpeg when it is there (plays everywhere), else OpenCV's mp4v."""

    def __init__(self, path, fps, size):
        self.cv = None
        try:
            import imageio.v2 as imageio
            self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=2,
                                        ffmpeg_log_level="error")
        except Exception as e:                      # noqa: BLE001
            import cv2
            print(f"[video] no imageio-ffmpeg ({e.__class__.__name__}): mp4v - play it in VLC", flush=True)
            self.w = None
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def add(self, bgr):
        if self.cv is not None:
            self.cv.write(bgr)
        else:
            self.w.append_data(bgr[:, :, ::-1])

    def close(self):
        (self.cv.release() if self.cv is not None else self.w.close())


def load(args):
    tag = args.tag
    names = json.load(open(os.path.join(args.system, f"names_{tag}.json"), encoding="utf-8"))
    t_start, step = names["t_start"], names["step"]
    key = lambda ts: int(round((ts - t_start) / step))
    boxes = collections.defaultdict(list)                       # (cam, key) -> [(box, track)]
    for r in read_jsonl(os.path.join(args.system, f"tracks_{tag}.jsonl")):
        boxes[(r["cam"], key(r["t"]))].append((r["box"], r["track"]))
    said = {(tid, m): (n, a) for tid, m, n, a in names["names"]}
    gt = {}
    for r in read_jsonl(os.path.join(args.data, "labels.jsonl")):
        k = key(r["ts"])
        if abs(r["ts"] - (t_start + k * step)) <= step / 2 + 0.6:
            gt[(r["cam"], k)] = r["boxes"]
    frames = collections.defaultdict(dict)
    lo, hi = hour_of(args.time_from), hour_of(args.time_to)
    for r in read_jsonl(os.path.join(args.data, "live.jsonl")):
        if lo <= local_hour(r["ts"]) < hi:
            frames[key(r["ts"])][r["cam"]] = r
    masks = dict(np.load(os.path.join(args.data, "masks.npz"))) if os.path.exists(os.path.join(args.data, "masks.npz")) else {}
    return names, boxes, said, gt, frames, masks, key


def render(args):
    import cv2
    from PIL import Image
    names, boxes, said, gt, frames, masks, _ = load(args)
    if not frames:
        sys.exit(f"no live frames between {args.time_from} and {args.time_to}")
    t_start, t_setup, step = names["t_start"], names["t_setup"], names["step"]
    track_album = {int(k): v for k, v in names["track_album"].items()}
    to_cow = {k: v for k, v in names["album_to_cow"].items()}
    cams = sorted({c for f in frames.values() for c in f})
    cols = 2 if len(cams) > 1 else 1
    rows_n = (len(cams) + cols - 1) // cols
    W = args.width // cols
    first = next(iter(frames.values()))
    w0, h0 = Image.open(next(iter(first.values()))["path"]).size
    H = int(round(W * h0 / w0 / 2) * 2)
    head = 64
    size = (W * cols, H * rows_n + head)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out = Writer(args.out, args.fps, size)
    repeat = max(1, int(round(args.fps * step))) if args.realtime else 1     # real time: each step shown `step` s
    dark = {}
    for c in cams:
        if masks.get(c) is not None:
            dark[c] = cv2.resize(masks[c].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) == 0
    tally = collections.Counter()
    for n, k in enumerate(sorted(frames)):
        t = t_start + k * step
        minute = int((t - t_start) // 60)
        setup = t < t_setup
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        for ci, cam in enumerate(cams):
            tile = np.zeros((H, W, 3), np.uint8)
            if cam in frames[k]:
                tile = cv2.resize(cv2.imread(frames[k][cam]["path"]), (W, H), interpolation=cv2.INTER_AREA)
            if cam in dark:
                tile[dark[cam]] = (tile[dark[cam]] * 0.35).astype(np.uint8)
            px = lambda b: (int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))
            pred = []                                          # (box, album cow named or None, best guess)
            for b, tid in boxes.get((cam, k), []):
                if setup:
                    a = track_album.get(tid)
                    text, who, guess = (a or f"T{tid}"), a, None
                    col = colour(a if a else f"T{tid}")
                else:
                    n_, guess = said.get((tid, minute), (None, None))
                    who = n_
                    if n_ is not None:
                        text = f"{n_} = cow {to_cow.get(n_, '?')}"
                    elif guess is not None:
                        text = f"not sure ({guess} = cow {to_cow.get(guess, '?')}?)"
                    else:
                        text = "-"
                    col = colour(n_)
                x1, y1, x2, y2 = px(b)
                cv2.rectangle(tile, (x1, y1), (x2, y2), col, 2)
                label(tile, text, x1, y1 - 2, (0, 0, 0), col)
                pred.append((b, who, guess))
            for cow, b in gt.get((cam, k), []):                # the truth, every 15 s
                x1, y1, x2, y2 = px(b)
                cv2.rectangle(tile, (x1, y1), (x2, y2), (255, 255, 255), 1)
                best = max(pred, key=lambda p: iou(p[0], b), default=None)
                hit = best is not None and iou(best[0], b) >= args.match_iou
                if setup:
                    mark, mc = "", (255, 255, 255)
                elif not hit or best[1] is None:
                    mark, mc = " ?", (170, 170, 170)
                    tally["no name" if hit else "no box"] += 1
                    if hit and best[2] is not None:
                        ok = str(to_cow.get(best[2])) == str(cow)
                        mark += "ok" if ok else "x"
                        tally["guess right" if ok else "guess wrong"] += 1
                elif str(to_cow.get(best[1])) == str(cow):
                    mark, mc = " OK", (60, 220, 60)
                    tally["right"] += 1
                else:
                    mark, mc = " X", (60, 60, 240)
                    tally["wrong"] += 1
                label(tile, f"cow {cow}{mark}", x1, y2 + 16, mc, (0, 0, 0))
            label(tile, cam, 6, 20, (255, 255, 255), (0, 0, 0), 0.6)
            r, c = divmod(ci, cols)
            canvas[head + r * H: head + (r + 1) * H, c * W:(c + 1) * W] = tile
        clock = C.dt.datetime.fromtimestamp(t, C.MMCOWS_TZ).strftime("%H:%M:%S")
        phase = "SETUP (building the album)" if setup else "IDENTIFYING"
        cv2.putText(canvas, f"{args.title}MmCows 25 Jul {clock}   {phase}   tracking: {args.tag}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, (f"labelled boxes so far: right {tally['right']}  wrong {tally['wrong']}  "
                             f"not named {tally['no name']} (its guess right {tally['guess right']})  "
                             f"not boxed {tally['no box']}") if not setup else
                    "white = MmCows label (every 15 s); colours = album cows; T<n> = track not in the album",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        for _ in range(repeat):
            out.add(canvas)
        if n % 50 == 0:
            print(f"\r  {n + 1}/{len(frames)} frames", end="", flush=True)
    out.close()
    print(f"\n[video] {len(frames)} moments ({len(frames) * repeat / args.fps:.0f} s of video"
          f"{', real time' if args.realtime else f' at {args.fps} fps'}) -> {args.out}"
          + (f"; labelled boxes: {dict(tally)}" if tally else ""), flush=True)
    return dict(tally)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data_mmcows.py's folder (live.jsonl, labels.jsonl, masks.npz)")
    p.add_argument("--system", required=True, help="system.py's --out (tracks_<tag>.jsonl, names_<tag>.json)")
    p.add_argument("--tag", default="assoc", choices=("assoc", "iou_per_camera"))
    p.add_argument("--from", dest="time_from", default="13:00", help="local time, HH:MM or HH:MM:SS")
    p.add_argument("--to", dest="time_to", default="13:10")
    p.add_argument("--fps", type=float, default=5.0, help="video frames per second (one per 2-s step: 5 = 10x speed)")
    p.add_argument("--realtime", action="store_true", help="play at the barn's speed: each moment repeated to last its step")
    p.add_argument("--title", default="", help="text before the header, e.g. 'v14 (C)  '")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--out", required=True)
    render(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
