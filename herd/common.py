"""Shared by every part of herd: configuration, geometry, small IO helpers.

herd is the barn system of the PDF brief: identify each of up to 60 cows by
appearance and gait, record lying / standing / feeding / drinking / idle time
every second and rumination every minute, and report and alert from that.
It reuses cowbench (CBVD-5 loading, the RT-DETRv2 detector) by path.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BENCH = os.path.join(REPO, "cowbench")
# Appended, not prepended: herd's own modules (report, train, ...) win over
# cowbench's namesakes; cowbench only lends cbvd, tracks, detect, detector.
for p in (BENCH, os.path.join(BENCH, "lora")):
    if p not in sys.path:
        sys.path.append(p)

POSTURES = ("standing", "lying")
ACTIVITIES = ("feeding", "drinking", "none")          # rumination is its own head
STATES = ("lying", "standing", "feeding", "drinking", "ruminating", "idle")


# ------------------------------------------------------------------ config

DEFAULTS = {
    "farm": {"timezone_offset_hours": 0, "migration_weekday": 1, "migration_hour": 10, "nightly_hour": 2},
    "sampling": {"fps": 1.0, "burst_every_s": 60, "burst_seconds": 7.0, "burst_fps": 25.0,
                 "burst_detect_fps": 5.0, "frame_width": 1920},   # Full HD: far cows keep their pixels
    "model": {"checkpoint": "", "encoder": "facebook/dinov2-small", "crop": 224, "grid": 2},
    "detector": {"weights": "", "threshold": None, "tiles": None},   # None: as the detector was trained
    "tracker": {"iou": 0.3, "max_age_s": 5},
    "identity": {"prototypes_per_cow": 6, "history_days": 7, "max_error": 0.01,
                 "new_cow_similarity": 0.55, "new_cow_min_bursts": 20, "retire_after_hours": 48},
    "lameness": {"enabled": False},
    "alerts": {"baseline_days": 7, "min_baseline_days": 3, "min_observed_minutes": 120,
               "mad_z": 3.0, "rumination_drop": 0.20, "lying_directions": ["low", "high"],
               "lying_change": 0.20},
    "report": {"period_days": 3},
    "store": {"path": "herd.sqlite"},
    # the barn's bursts kept for training identity on the barn's own cows (barn_dataset.py)
    "training_cache": {"enabled": False, "every_n": 10, "folder": "training_cache", "max_gb": 200},
    "cameras": [],
}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path=None):
    """The TOML config merged over DEFAULTS; no file means defaults only."""
    cfg = DEFAULTS
    if path:
        with open(path, "rb") as fh:
            cfg = _merge(DEFAULTS, tomllib.load(fh))
    for cam in cfg["cameras"]:
        cam.setdefault("mask", [])
        cam.setdefault("name", cam.get("id"))
    return cfg


# ---------------------------------------------------------------- geometry

def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def in_polygon(x, y, poly):
    """Ray casting; poly is [[x, y], ...] in 0-1 frame coordinates."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def in_mask(box, mask):
    """A cow belongs to a camera when its box centre lies in the camera's mask.

    Cameras that partly overlap get masks that do not: every patch of floor
    is drawn into exactly one camera's mask, so one cow is counted once. An
    empty mask means the whole frame."""
    if not mask:
        return True
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    polys = mask if isinstance(mask[0][0], (list, tuple)) else [mask]
    return any(in_polygon(cx, cy, p) for p in polys)


def interpolate_box(keys, t):
    """Box at time t from {time: box} keyframes: linear between, held outside."""
    times = sorted(keys)
    if t <= times[0]:
        return list(keys[times[0]])
    if t >= times[-1]:
        return list(keys[times[-1]])
    for a, b in zip(times, times[1:]):
        if a <= t <= b:
            w = (t - a) / (b - a) if b > a else 0.0
            return [ka + w * (kb - ka) for ka, kb in zip(keys[a], keys[b])]
    return list(keys[times[-1]])


def square_crop_box(box, w, h, margin=0.1):
    """Pixel box around a 0-1 box, grown by margin and made square (the encoder
    takes squares; squashing a lying cow into one would change its shape)."""
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = max(side, 8.0)
    return (int(math.floor(cx - side / 2)), int(math.floor(cy - side / 2)),
            int(math.ceil(cx + side / 2)), int(math.ceil(cy + side / 2)))


def crop(image_rgb, box, size, margin=0.1):
    """numpy HxWx3 frame -> size x size crop of the box (padded with grey at edges)."""
    import numpy as np
    from PIL import Image
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = square_crop_box(box, w, h, margin)
    out = np.full((y2 - y1, x2 - x1, 3), 114, dtype=np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = image_rgb[sy1:sy2, sx1:sx2]
    return np.asarray(Image.fromarray(out).resize((size, size), Image.BILINEAR))


# --------------------------------------------------------------------- io

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    os.replace(path + ".part", path)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
