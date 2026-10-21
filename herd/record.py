#!/usr/bin/env python3
"""Record the barn's cameras to files named as the dataset tools read them: every camera of the config
at once, for --minutes, into OUT/<camera>_<local start time>.mp4 (ffmpeg, the stream copied as it is).

    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 60
    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 30 --cameras cam1 cam2

Then: herd.py barn video / herd.py boxes frames --videos OUT (SETUP.md sections 11-12).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_config  # noqa: E402


def command(url, seconds, path):
    rtsp = ["-rtsp_transport", "tcp"] if url.startswith("rtsp") else []
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *rtsp, "-i", url, "-t", str(seconds),
            "-c", "copy", "-an", path]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--cameras", nargs="*", default=None, help="only these camera ids")
    args = p.parse_args(argv)
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not installed (the Docker image has it; on a server: apt install ffmpeg)")
    cfg = load_config(args.config)
    cams = [c for c in cfg["cameras"] if not args.cameras or c["id"] in args.cameras]
    if not cams:
        sys.exit(f"no camera {args.cameras} in {args.config}")
    os.makedirs(args.out, exist_ok=True)
    tz = dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))
    start = dt.datetime.now(tz)
    seconds = int(args.minutes * 60)
    procs = []
    for c in cams:
        path = os.path.join(args.out, f"{c['id']}_{start:%Y-%m-%d_%H-%M-%S}.mp4")
        procs.append((c["id"], path, subprocess.Popen(command(c["url"], seconds, path))))
    print(f"[record] {len(procs)} cameras for {args.minutes:g} min from {start:%Y-%m-%d %H:%M:%S} -> {args.out}",
          flush=True)
    t0 = time.time()
    bad = 0
    for cam, path, pr in procs:
        code = pr.wait()
        size = os.path.getsize(path) / 2 ** 20 if os.path.exists(path) else 0
        ok = code == 0 and size > 0
        bad += not ok
        print(f"[record] {cam}: {path} {size:.0f} MB" + ("" if ok else f"  !! ffmpeg exit {code} - check its url"),
              flush=True)
    print(f"[record] done in {(time.time() - t0) / 60:.1f} min")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
