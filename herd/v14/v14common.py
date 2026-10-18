"""Version 14 (A): cow identification only, trained on MultiCamCows2024 and MmCows.

Shared bits of the v14 scripts: paths onto herd/ and cowbench/, timestamps in file
names, MmCows label lines, the pen mask, small JSON helpers.

    MultiCamCows2024 (Bristol, UK Non-Commercial Government Licence): 90 cows, 7 days of
        lunchtime milking (14-20 Aug 2023), 3 ceiling cameras over a walkway; torso crops
        sorted as MultiCamCows2024Root/<day>/<cow>/... - the identity data.
    MmCows (Purdue, CC BY-NC-SA 4.0): one pen, 16 cows, 4 synchronised cameras over it;
        25 July: a frame every 15 s with every cow boxed WITH her id (visual_data.zip), and
        12:00-15:00 at 1 frame a second (1s_interval_images_3hr) - the multi-camera data.

Both non-commercial: research and personal use only.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HERD = os.path.dirname(HERE)
REPO = os.path.dirname(HERD)
for _p in (HERD, os.path.join(REPO, "cowbench"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MMCOWS_TZ = dt.timezone(dt.timedelta(hours=-5))      # the dataset's local time: CDT, UTC-5
MMCOWS_DAY = "0725"
MMCOWS_FULL = (4480, 2800)                           # labels are 0-1 of this frame


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_np_default)
    os.replace(path + ".part", path)


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def unix_from_name(name, day=MMCOWS_DAY, year=2023, tz=MMCOWS_TZ):
    """A frame's time from its file name -> unix seconds (float), or None.

    MmCows names its frames by unix time (10 digits, sometimes with a fraction);
    anything else that looks like HHMMSS is read as local time on `day`."""
    base = os.path.basename(name)
    m = re.search(r"(1[5-9]\d{8})(?:[._](\d{1,6}))?", base)
    if m:
        frac = float("0." + m.group(2)) if m.group(2) and len(m.group(2)) <= 3 else 0.0
        return float(m.group(1)) + frac
    m = re.search(r"(?<!\d)([01]\d|2[0-3])[-_:]?([0-5]\d)[-_:]?([0-5]\d)(?!\d)", base)
    if m:
        month, d = int(day[:2]), int(day[2:])
        t = dt.datetime(year, month, d, int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=tz)
        return t.timestamp()
    return None


def local_hour(ts, tz=MMCOWS_TZ):
    t = dt.datetime.fromtimestamp(ts, tz)
    return t.hour + t.minute / 60.0 + t.second / 3600.0


def parse_id_boxes(text):
    """MmCows label lines "cow_id cx cy w h" (0-1 of the 4480 x 2800 frame) -> [(cow, [x1, y1, x2, y2])]."""
    out = []
    for line in text.splitlines():
        p = line.replace(",", " ").split()
        if len(p) != 5:
            continue
        try:
            cow = int(float(p[0]))
            cx, cy, w, h = map(float, p[1:])
        except ValueError:
            continue
        if max(cx, cy, w, h) > 1.5:                      # pixels of the full frame
            W, H = MMCOWS_FULL
            cx, cy, w, h = cx / W, cy / H, w / W, h / H
        b = [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]
        if b[2] - b[0] > 1e-3 and b[3] - b[1] > 1e-3:
            out.append((cow, [round(v, 5) for v in b]))
    return out


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def pen_mask(images, thr=6, size=(560, 350)):
    """MmCows' labelled frames have the other pens painted black: the pixels black in
    (almost) all of them are outside the pen -> bool (H, W) at `size`, True = keep."""
    from PIL import Image
    acc = None
    for img in images:
        a = np.asarray(img.convert("L").resize(size, Image.NEAREST)) > thr
        acc = a.astype(np.int32) if acc is None else acc + a
    if acc is None:
        return None
    return acc >= max(1, int(0.1 * len(images)))


def apply_mask(img, mask, fill=0):
    """PIL image, bool mask (any size) -> the image with the masked-out part painted `fill`."""
    from PIL import Image
    if mask is None:
        return img
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) > 0
    a = np.asarray(img).copy()
    a[~m] = fill
    return Image.fromarray(a)
