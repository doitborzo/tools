#!/usr/bin/env python3
"""Which cow the system thinks each box is, drawn on the video: a way to watch identification work.

    python herd/herd.py show-ids --config barn.toml --cam cam1 --video cam1_0600.mp4 \\
        --start 2026-10-07T06:00:00 [--from-min 0 --minutes 10] --out /workspace/show_cam1

The barn's own run on a recorded file (herd.py run --start ...) wrote every second's boxes by track
(table seconds) and every burst's decision (bursts); this draws them back on that file, a frame a
second. Each box shows

    line 1   the cow the track was given (report.track_map: the majority of its confirmed bursts, with
             cow_aliases and names.json), or NaN - a track no cow could be given
    line 2   the track, and the last burst decision on it so far: confirmed / tentative / unknown, p
    line 3   posture and activity of that second

and its colour is the cow's (one colour per cow, the same in every camera and every video), grey for
NaN. Out: <out>/ids.mp4 (MPEG-4: VLC plays it everywhere), a still every --still-every seconds in
<out>/stills/ and <out>/index.html with them (any browser).

mct_eval.py show draws MmCows' four cameras the same way, with the true cows' numbers beside.
"""

from __future__ import annotations

import argparse
import colorsys
import datetime as dt
import hashlib
import html
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GREY = (150, 150, 150)
RED = (230, 30, 30)


# ------------------------------------------------------------------ drawing (RGB images)

def colour(key):
    """A stable, well-saturated RGB colour per key (a cow, a track)."""
    h = int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.75 + (h >> 9) % 25 / 100, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_box(img, box, col, lines, bad=False, scale=0.5):
    """box 0-1 on an RGB image (in place): a rectangle in col, the label lines on a band of col above
    it; bad: a thick red frame around it as well (a wrong identity)."""
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    if bad:
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), RED, 4)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 1
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t in lines]
    lh = max((s[1] for s in sizes), default=0) + 6
    bw = max((s[0] for s in sizes), default=0) + 6
    top = max(0, y1 - lh * len(lines))
    cv2.rectangle(img, (x1, top), (min(w - 1, x1 + bw), top + lh * len(lines)), col, -1)
    dark = sum(col) > 380                                  # dark text on a light band
    for k, t in enumerate(lines):
        cv2.putText(img, t, (x1 + 3, top + lh * (k + 1) - 4), font, scale, (0, 0, 0) if dark else (255, 255, 255),
                    th, cv2.LINE_AA)


def header(img, text, scale=0.7):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


class Writer:
    """Frames -> an MPEG-4 video, and every still_every-th frame -> a JPEG for index.html."""

    def __init__(self, out_dir, fps=4, still_every=60, title="identities"):
        os.makedirs(os.path.join(out_dir, "stills"), exist_ok=True)
        self.out, self.fps, self.every, self.title = out_dir, fps, still_every, title
        self.video, self.stills, self.k = None, [], 0

    def add(self, img_rgb, caption):
        import cv2
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if self.video is None:
            h, w = bgr.shape[:2]
            self.video = cv2.VideoWriter(os.path.join(self.out, "ids.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         self.fps, (w, h))
        self.video.write(bgr)
        if self.every and self.k % self.every == 0:
            name = f"stills/{len(self.stills):04d}.jpg"
            cv2.imwrite(os.path.join(self.out, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.stills.append((name, caption))
        self.k += 1

    def close(self, notes=""):
        if self.video is not None:
            self.video.release()
        with open(os.path.join(self.out, "index.html"), "w", encoding="utf-8") as fh:
            fh.write(f"<!doctype html><meta charset='utf-8'><title>{html.escape(self.title)}</title>"
                     "<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}"
                     "img{max-width:100%;display:block;margin:4px 0 18px}</style>"
                     f"<h1>{html.escape(self.title)}</h1><p>{notes}</p><p>video: <a href='ids.mp4'>ids.mp4</a> "
                     f"({self.k} frames, {self.fps} a second)</p>")
            for name, cap in self.stills:
                fh.write(f"<div>{html.escape(cap)}</div><img src='{name}' loading='lazy'>")
        return os.path.join(self.out, "index.html")


# ------------------------------------------------------------------ the barn's store

def _lookup(store, cfg, cam, t0, t1):
    import report
    secs = store.query("SELECT track, ts, x1, y1, x2, y2, posture, activity FROM seconds "
                       "WHERE cam = ? AND ts >= ? AND ts < ? ORDER BY ts", (cam, t0 - 0.5, t1 + 0.5))
    tm = report.track_map(store)
    nm = report.names(cfg)
    bursts = {}
    for track, ts, state, cow, p in store.query("SELECT track, ts, state, cow, p FROM bursts WHERE ts < ? "
                                                "ORDER BY ts", (t1,)):
        bursts.setdefault(track, []).append((ts, state, cow, p))
    return secs, tm, nm, bursts


def cmd_barn(args):
    import cv2
    from common import load_config
    from store import Store
    cfg = load_config(args.config)
    store = Store(cfg["store"]["path"])
    start = dt.datetime.fromisoformat(args.start).timestamp()
    t0 = start + args.from_min * 60
    t1 = t0 + args.minutes * 60
    secs, tm, nm, bursts = _lookup(store, cfg, args.cam, t0, t1)
    if not secs:
        sys.exit(f"no seconds of {args.cam} between {dt.datetime.fromtimestamp(t0)} and {dt.datetime.fromtimestamp(t1)} "
                 f"in {cfg['store']['path']} - was the file run with herd.py run --start {args.start}?")
    by_sec = {}
    for row in secs:
        by_sec.setdefault(int(round(row[1])), []).append(row)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 - start) * fps))
    w = Writer(args.out, args.fps, args.still_every, f"{args.cam}: who is who")
    i, shown, named = int((t0 - start) * fps), 0, 0
    next_s = int(np.ceil(t0))
    while next_s < t1:
        ok, f = cap.read()
        if not ok:
            break
        ts = start + i / fps
        i += 1
        if ts + 0.5 / fps < next_s:
            continue
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if img.shape[1] > args.width:
            img = cv2.resize(img, (args.width, int(img.shape[0] * args.width / img.shape[1])), interpolation=cv2.INTER_AREA)
        rows = by_sec.get(next_s, [])
        for track, _, x1, y1, x2, y2, posture, activity in rows:
            cow = tm.get(track)
            last = [b for b in bursts.get(track, []) if b[0] <= next_s]
            dec = (f"{last[-1][1]} {nm.get(last[-1][2], last[-1][2]) or ''} p{last[-1][3]:.2f}"
                   if last and last[-1][3] is not None else (last[-1][1] if last else "no burst yet"))
            label = nm.get(cow, cow) if cow else "NaN"
            named += cow is not None
            draw_box(img, [x1, y1, x2, y2], colour(cow) if cow else GREY,
                     [f"{label}", f"trk ..{track[-6:]}  {dec}", f"{posture} / {activity}"])
        shown += len(rows)
        stamp = dt.datetime.fromtimestamp(next_s).strftime("%Y-%m-%d %H:%M:%S")
        header(img, f"{args.cam}  {stamp}  cows {len(rows)}, identified {sum(tm.get(r[0]) is not None for r in rows)}")
        w.add(img, f"{args.cam} {stamp}")
        next_s += 1
    cap.release()
    page = w.close(f"{shown} boxes, {named} on a track with a cow ({named / max(1, shown):.0%}); grey = NaN. "
                   "The cow of a track is the majority of its confirmed bursts (report.track_map).")
    print(f"[show-ids] {w.k} seconds, {shown} boxes ({named / max(1, shown):.0%} with a cow) -> {page}, "
          f"{os.path.join(args.out, 'ids.mp4')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--cam", required=True, help="the camera id in the config")
    p.add_argument("--video", required=True, help="the recorded file the run was made on")
    p.add_argument("--start", required=True, help="the --start the run was given (ISO): the file's first frame")
    p.add_argument("--from-min", type=float, default=0.0)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=4, help="video frames a second (one per second of the barn)")
    p.add_argument("--still-every", type=int, default=30, help="a still in index.html every N seconds")
    p.add_argument("--width", type=int, default=1920)
    cmd_barn(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
