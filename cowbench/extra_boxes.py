#!/usr/bin/env python3
"""Other datasets' cow boxes -> one list the detector can train on.

The detector (detector.py) learns from CBVD-5's keyframes. More barns, cameras
and angles are what it lacks most (it finds 93.6% of the cows on its held-out
train clips, 75% on val), so other public datasets' boxes are added with
`detector.py train --extra <list.jsonl>[:N]`. This file makes those lists:

    convert  a dataset's images + labels -> <name>.jsonl, one line per image:
             {"image": "/abs/path.jpg", "boxes": [[x1, y1, x2, y2], ...], "source": "<name>"}
             boxes on the 0-1 scale, one class (cow)
    show     draws the boxes of a few images of a list onto a sheet (PNG):
             LOOK AT IT before training - a wrong guess of the format (centre vs
             corner, pixels vs 0-1) gives boxes beside the cows
    stats    images, boxes, boxes an image, box sizes

Label formats, found by --format auto from the files themselves:
    yolo     .txt, a line "class cx cy w h" (0-1); Darknet as in OpenCows2020's labels-darknet
    voc      .xml, <bndbox> xmin ymin xmax ymax (pixels); <robndbox> cx cy w h angle
             (rotated, roLabelImg) becomes the upright box around it
    coco     one .json with "images" and "annotations" (bbox x, y, w, h in pixels)
    mmcows   .txt, a line "cow_id x y w h" (MmCows' labels/combined); values over 1 are
             pixels of --size (4480x2800 for MmCows), else 0-1; --xy centre|corner
             says what x, y are (the README does not; check with show)

Images and labels are paired by file name (the stem), anywhere under --images
and --labels; images without a label file are left out (an unlabelled image
is not "no cows"). Only box labels are read - ids, behaviours, keypoints are
not needed by the detector.

    python extra_boxes.py convert --name cows2021 --images Cows2021/Detection/Train/images \\
        --labels Cows2021/Detection/Train/annotations --out /workspace/extra/cows2021.jsonl
    python extra_boxes.py show  /workspace/extra/cows2021.jsonl --out /workspace/extra/cows2021_check.png
    python extra_boxes.py stats /workspace/extra/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def by_stem(root, exts):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in exts:
                out.setdefault(stem, os.path.join(d, f))
    return out


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def clean(boxes, min_side=0.002):
    """Clip to the frame, drop boxes with no area. [[x1, y1, x2, y2]] on 0-1."""
    out = []
    for b in boxes:
        x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in b)
        if x2 - x1 > min_side and y2 - y1 > min_side:
            out.append([round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)])
    return out


def from_centre(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ------------------------------------------------------------------ readers

def read_yolo(path, size=None):
    boxes = []
    for line in open(path, encoding="utf-8"):
        v = line.split()
        if len(v) >= 5:
            boxes.append(from_centre(*map(float, v[1:5])))
    return boxes


def read_mmcows(path, size, xy="centre"):
    """`cow_id x y w h`; pixels of `size` when any value is over 1, else 0-1."""
    rows = []
    for line in open(path, encoding="utf-8"):
        v = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
        if len(v) >= 5:
            rows.append([float(x) for x in v[1:5]])
    if not rows:
        return []
    W, H = size
    if max(max(r) for r in rows) > 1.0:
        rows = [[x / W, y / H, w / W, h / H] for x, y, w, h in rows]
    if xy == "centre":
        return [from_centre(*r) for r in rows]
    return [[x, y, x + w, y + h] for x, y, w, h in rows]


def read_voc(path, size=None):
    root = ET.parse(path).getroot()
    s = root.find("size")
    if s is not None and s.findtext("width"):
        W, H = float(s.findtext("width")), float(s.findtext("height"))
    else:
        W, H = size
    boxes = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        if b is not None:
            boxes.append([float(b.findtext("xmin")) / W, float(b.findtext("ymin")) / H,
                          float(b.findtext("xmax")) / W, float(b.findtext("ymax")) / H])
            continue
        r = obj.find("robndbox")
        if r is not None:              # rotated (roLabelImg): the upright box around it
            cx, cy, w, h, a = (float(r.findtext(k)) for k in ("cx", "cy", "w", "h", "angle"))
            c, s_ = abs(math.cos(a)), abs(math.sin(a))
            bw, bh = w * c + h * s_, w * s_ + h * c
            boxes.append([(cx - bw / 2) / W, (cy - bh / 2) / H, (cx + bw / 2) / W, (cy + bh / 2) / H])
    return boxes


def read_coco(path):
    """-> {image file name: [boxes 0-1]}"""
    d = json.load(open(path, encoding="utf-8"))
    imgs = {i["id"]: i for i in d.get("images", [])}
    out = {}
    for a in d.get("annotations", []):
        im = imgs.get(a.get("image_id"))
        if im is None or not a.get("bbox"):
            continue
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        out.setdefault(im["file_name"], []).append([x / W, y / H, (x + w) / W, (y + h) / H])
    for im in imgs.values():
        out.setdefault(im["file_name"], [])     # listed with no box: no cow in it
    return out


def guess_format(labels):
    """yolo / mmcows / voc / coco from what is in the labels folder."""
    exts = {}
    for d, _, files in os.walk(labels):
        for f in files:
            e = os.path.splitext(f)[1].lower()
            exts.setdefault(e, os.path.join(d, f))
    if ".xml" in exts:
        return "voc"
    if ".json" in exts and ".txt" not in exts:
        return "coco"
    if ".txt" in exts:
        for line in open(exts[".txt"], encoding="utf-8"):
            v = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
            if len(v) >= 5:
                vals = [float(x) for x in v[1:5]]
                # Darknet: a small class number and 0-1 values; else ids and/or pixels
                return "yolo" if float(v[0]) < 10 and max(vals) <= 1.0 else "mmcows"
        return "yolo"
    sys.exit(f"no .xml, .json or .txt labels found under {labels}")


# ------------------------------------------------------------------ commands

def cmd_convert(args):
    fmt = args.format if args.format != "auto" else guess_format(args.labels)
    images = by_stem(args.images, IMAGE_EXT)
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    rows, skipped = [], 0
    if fmt == "coco":
        jsons = [os.path.join(d, f) for d, _, fs in os.walk(args.labels) for f in fs if f.endswith(".json")]
        for j in jsons:
            for name, boxes in read_coco(j).items():
                path = images.get(os.path.splitext(os.path.basename(name))[0])
                if path is None:
                    skipped += 1
                    continue
                rows.append((path, boxes))
    else:
        ext = ".xml" if fmt == "voc" else ".txt"
        labels = by_stem(args.labels, (ext,))
        for stem, lpath in sorted(labels.items()):
            path = images.get(stem)
            if path is None:
                skipped += 1
                continue
            if fmt == "voc":
                boxes = read_voc(lpath, size or image_size(path))
            elif fmt == "mmcows":
                boxes = read_mmcows(lpath, size or image_size(path), args.xy)
            else:
                boxes = read_yolo(lpath)
            rows.append((path, boxes))
    rows.sort()
    if args.every > 1:            # near-duplicate frames (MmCows: every 15 s of one day)
        rows = rows[::args.every]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n_boxes = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for path, boxes in rows:
            b = clean(boxes)
            n_boxes += len(b)
            fh.write(json.dumps({"image": os.path.abspath(path), "boxes": b, "source": args.name}) + "\n")
    print(f"[extra] {args.name}: format {fmt}, {len(rows)} images, {n_boxes} boxes "
          f"({skipped} labels without an image) -> {args.out}")
    print(f"[extra] check the boxes before training: python {sys.argv[0]} show {args.out}")


def read_list(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cmd_show(args):
    from PIL import Image, ImageDraw
    rows = [r for r in read_list(args.list) if r["boxes"]]
    pick = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    tiles = []
    for r in pick:
        img = Image.open(r["image"]).convert("RGB")
        img.thumbnail((640, 640))
        d = ImageDraw.Draw(img)
        w, h = img.size
        for b in r["boxes"]:
            d.rectangle([b[0] * w, b[1] * h, b[2] * w, b[3] * h], outline=(0, 255, 0), width=3)
        tiles.append(img)
    if not tiles:
        sys.exit("no image with boxes in the list")
    cols = 3
    tw, th = max(t.size[0] for t in tiles), max(t.size[1] for t in tiles)
    sheet = Image.new("RGB", (cols * tw, math.ceil(len(tiles) / cols) * th), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    out = args.out or os.path.splitext(args.list)[0] + "_check.png"
    sheet.save(out)
    print(f"[extra] {len(tiles)} images with their boxes -> {out}  (every box should sit on a cow)")


def cmd_stats(args):
    for path in args.lists:
        rows = read_list(path)
        n = sum(len(r["boxes"]) for r in rows)
        areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for r in rows for b in r["boxes"])
        q = (lambda p: f"{100 * areas[min(len(areas) - 1, int(p * len(areas)))]:.2f}%") if areas else (lambda p: "-")
        print(f"{os.path.basename(path)}: {len(rows)} images, {n} boxes, "
              f"{n / max(1, len(rows)):.1f} a image; box area (of the frame) p10 {q(0.1)} p50 {q(0.5)} p90 {q(0.9)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--name", required=True, help="the source's name in the list (e.g. mmcows)")
    c.add_argument("--images", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--format", choices=("auto", "yolo", "voc", "coco", "mmcows"), default="auto")
    c.add_argument("--xy", choices=("centre", "corner"), default="centre",
                   help="mmcows: is x, y the box centre or its top-left corner")
    c.add_argument("--size", default=None, help="WxH the pixel labels refer to (MmCows: 4480x2800); "
                                                "default: each image's own size")
    c.add_argument("--every", type=int, default=1, help="keep every N-th image (near-duplicate frames)")
    s = sub.add_parser("show")
    s.add_argument("list")
    s.add_argument("--out", default=None)
    s.add_argument("--n", type=int, default=9)
    s.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("stats")
    t.add_argument("lists", nargs="+")
    args = p.parse_args(argv)
    {"convert": cmd_convert, "show": cmd_show, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
