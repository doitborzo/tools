#!/usr/bin/env python3
"""Recordings whose files are damaged (a recorder's broken metadata, time stamps, corrupt stretches) made
readable: each video encoded again by ffmpeg into a clean .mp4 - a steady frame rate, time stamps from
zero, broken frames dropped - under the same name, in another folder. They play in a player but the
tools read them wrong (one camera's clock standing still, frames missing, decoder errors).

    python herd/herd.py fix-videos --videos /srv/herd/rec --out /srv/herd/rec_fixed
    python herd/herd.py fix-videos --videos rec --out rec_fixed --fps 25 --copy    # only re-wrap, no encode

Then use --videos /srv/herd/rec_fixed with barn video / boxes frames / zones page.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import VIDEO_EXT  # noqa: E402


def stated_fps(path):
    """The video's frame rate as ffprobe reads it, or None when it makes no sense (a time base, 0)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=avg_frame_rate,r_frame_rate", "-of", "json", path],
                             capture_output=True, text=True, timeout=60).stdout
        s = json.loads(out)["streams"][0]
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = s.get(key, "0/0").partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if 1.0 <= fps <= 60.0:
            return round(fps, 3)
    return None


def command(src, dst, fps, copy=False, crf=20):
    head = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-err_detect", "ignore_err", "-fflags", "+genpts+discardcorrupt", "-i", src, "-map", "0:v:0", "-an"]
    if copy:                                     # the stream as it is, in a new container with new time stamps
        return head + ["-c:v", "copy", "-fps_mode", "passthrough", "-movflags", "+faststart", dst]
    return head + ["-vf", f"setpts=PTS-STARTPTS,fps={fps}", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]


def check(path):
    """(frames, seconds) as the tools will read the file."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n, (n / fps if fps else 0.0)


def one(src, dst, fps, copy, crf):
    t0 = time.time()
    rate = fps or stated_fps(src) or 25.0
    part = dst[:-4] + ".part.mp4"
    r = subprocess.run(command(src, part, rate, copy, crf), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(part) or os.path.getsize(part) == 0:
        if os.path.exists(part):
            os.remove(part)
        return f"!! {os.path.basename(src)}: ffmpeg failed ({(r.stderr or '').strip().splitlines()[-1:]})"
    os.replace(part, dst)
    n, secs = check(dst)
    errors = len([x for x in (r.stderr or "").splitlines() if x.strip()])
    return (f"{os.path.basename(src)} -> {os.path.basename(dst)}: {n} frames, {int(secs // 60)}:{int(secs % 60):02d} "
            f"at {rate:g} fps{' (copied)' if copy else ''}; {errors} damaged spots skipped; "
            f"{(time.time() - t0) / 60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="the folder of recordings (sub-folders too)")
    p.add_argument("--out", required=True, help="where the clean .mp4 go (the same names and sub-folders)")
    p.add_argument("--fps", type=float, default=None, help="the frame rate to write (default: the file's, else 25)")
    p.add_argument("--copy", action="store_true", help="no encoding: the stream re-wrapped with new time stamps "
                   "(fast; enough when only the container is broken)")
    p.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better, bigger; 18-23)")
    p.add_argument("--jobs", type=int, default=2, help="files at once")
    p.add_argument("--force", action="store_true", help="convert again files already in --out")
    args = p.parse_args(argv)
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not installed (the Docker image has it; on a server: apt install ffmpeg)")
    src_dir, out_dir = os.path.abspath(args.videos.rstrip("\\")), os.path.abspath(args.out.rstrip("\\"))
    todo = []
    for d, _, fs in os.walk(src_dir):
        if os.path.abspath(d).startswith(out_dir):
            continue
        for f in sorted(fs):
            if f.lower().endswith(VIDEO_EXT) and ".part." not in f:
                src = os.path.join(d, f)
                dst = os.path.join(out_dir, os.path.relpath(d, src_dir), os.path.splitext(f)[0] + ".mp4")
                if os.path.exists(dst) and not args.force:
                    print(f"[fix] {os.path.relpath(dst, out_dir)}: there already", flush=True)
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                todo.append((src, dst))
    if not todo:
        print(f"[fix] nothing to convert under {src_dir}")
        return 0
    print(f"[fix] {len(todo)} recordings -> {out_dir} ({'re-wrapped' if args.copy else 'encoded again, H.264'}, "
          f"{args.jobs} at once)", flush=True)
    bad = 0
    with cf.ThreadPoolExecutor(max(1, args.jobs)) as ex:
        for msg in ex.map(lambda t: one(t[0], t[1], args.fps, args.copy, args.crf), todo):
            bad += msg.startswith("!!")
            print(f"[fix] {msg}", flush=True)
    print(f"[fix] done -> use --videos {out_dir}" + (f"; {bad} failed" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
