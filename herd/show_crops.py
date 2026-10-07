#!/usr/bin/env python3
"""What the model sees: the frame with every cow's box and the square it is
cut to, the 224x224 crops at one or more margins, and (with a video) a strip
of one cow's 7 s burst.

    python herd/show_crops.py --root /workspace/cbvd5 --clip 371 --ts 5 --out crops.png
    python herd/show_crops.py --root /workspace/cbvd5 --clip 371 --ts 5 --burst 0 --out crops.png
    python herd/show_crops.py --image barn.jpg --boxes "0.1,0.2,0.3,0.5;0.5,0.2,0.7,0.6" --out crops.png

The crops are made by the same function (common.crop) as in training and in
the barn: the box grown by the margin on every side, made square, resized to
224; grey where the square runs off the frame.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import crop, interpolate_box, square_crop_box  # noqa: E402

COLOURS = [(50, 205, 50), (255, 140, 0), (30, 144, 255), (220, 20, 60)]   # box, then one per margin


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label(img, text, size=18):
    d = ImageDraw.Draw(img)
    f = font(size)
    l, t, r, b = d.textbbox((4, 4), text, font=f)
    d.rectangle([l - 3, t - 3, r + 3, b + 3], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 255), font=f)
    return img


def overview(img, boxes, margins, width=1280):
    """The frame: the annotated box in green, the square crop per margin in its colour."""
    h, w = img.shape[:2]
    pil = Image.fromarray(img).copy()
    d = ImageDraw.Draw(pil)
    lw = max(2, w // 500)
    for n, b in enumerate(boxes, 1):
        d.rectangle([b[0] * w, b[1] * h, b[2] * w, b[3] * h], outline=COLOURS[0], width=lw)
        for k, m in enumerate(margins):
            x1, y1, x2, y2 = square_crop_box(b, w, h, m)
            d.rectangle([x1, y1, x2, y2], outline=COLOURS[1 + k % 3], width=max(1, lw // 2))
        d.text((b[0] * w + lw, b[1] * h + lw), str(n), fill=(255, 255, 255), font=font(max(16, w // 60)))
    return pil.resize((width, int(h * width / w)))


def grid(tiles, cols, tile=224, pad=6, bg=(30, 30, 30)):
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * (tile + pad) + pad, rows * (tile + pad) + pad), bg)
    for i, t in enumerate(tiles):
        out.paste(t.resize((tile, tile)), (pad + (i % cols) * (tile + pad), pad + (i // cols) * (tile + pad)))
    return out


def stack(parts, bg=(30, 30, 30), pad=10):
    w = max(p.width for p in parts)
    out = Image.new("RGB", (w, sum(p.height for p in parts) + pad * (len(parts) + 1)), bg)
    y = pad
    for p in parts:
        out.paste(p, ((w - p.width) // 2, y))
        y += p.height + pad
    return out


def cbvd_boxes(root, clip, ts):
    import cbvd
    rows = cbvd.load_boxes(os.path.join(root, "annotations", "ava_val_v2.1.csv")) + \
        cbvd.load_boxes(os.path.join(root, "annotations", "ava_train_v2.1.csv"))
    boxes = [b for b in rows if b.video_id == str(clip) and b.timestamp == ts]
    if not boxes:
        sys.exit(f"no annotated cows in clip {clip} at {ts} s")
    return boxes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--clip", default=None)
    p.add_argument("--ts", type=int, default=5)
    p.add_argument("--image", default=None, help="any picture instead of a CBVD-5 keyframe")
    p.add_argument("--boxes", default=None, help='with --image: "x1,y1,x2,y2;..." as 0-1 fractions')
    p.add_argument("--margins", type=float, nargs="+", default=[0.1, 0.5])
    p.add_argument("--max-cows", type=int, default=8)
    p.add_argument("--burst", type=int, default=None, help="cow number (from the picture, 1-based) whose burst to show")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--out", default="crops.png")
    args = p.parse_args(argv)

    labels = []
    if args.image:
        img = np.asarray(Image.open(args.image).convert("RGB"))
        boxes = [[float(v) for v in s.split(",")] for s in args.boxes.split(";")]
        labels = ["" for _ in boxes]
    else:
        import cbvd
        found = cbvd_boxes(args.root, args.clip, args.ts)
        img = np.asarray(Image.open(cbvd.frame_path(args.root, found[0])).convert("RGB"))
        boxes = [list(b.xyxy) for b in found]
        labels = [f"{b.posture}/{b.activity}" for b in found]
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3], boxes[i][0]))[:args.max_cows]
    boxes, labels = [boxes[i] for i in order], [labels[i] for i in order]

    parts = [label(overview(img, boxes, args.margins), "green: box   " + "   ".join(
        f"{['orange', 'blue', 'red'][k % 3]}: crop at margin {m:g}" for k, m in enumerate(args.margins)))]
    tiles = []
    for n, b in enumerate(boxes, 1):
        for m in args.margins:
            tiles.append(label(Image.fromarray(crop(img, b, args.size, m)),
                               f"#{n} margin {m:g}" + (f"  {labels[n - 1]}" if labels[n - 1] else ""), 13))
    parts.append(grid(tiles, cols=len(args.margins) * 2))

    if args.burst is not None and not args.image:
        # one cow's burst as the model gets it: 7 s at 25 fps, every 25th frame shown
        import cbvd
        import tracks as tracks_mod
        import cbvd_bursts
        clip_boxes = [b for b in cbvd_boxes_all(args.root) if b.video_id == str(args.clip)]
        spec = cbvd_bursts.build_specs(clip_boxes)[str(args.clip)]
        target = boxes[args.burst - 1]
        best = max(spec["bursts"], key=lambda br: max(tracks_mod.iou(target, v) for v in br["keys"].values()))
        frames, fps = cbvd_bursts.read_video(cbvd.video_path(args.root, str(args.clip)))
        keys = {float(t): v for t, v in best["keys"].items()}
        strip = []
        for j in range(0, int(best["seconds"] * 25), 25):
            t = best["start"] + j / 25
            ix = min(len(frames) - 1, int(round(t * fps)))
            strip.append(label(Image.fromarray(crop(frames[ix], interpolate_box(keys, t), args.size, args.margins[0])),
                               f"{t:.1f} s", 13))
        parts.append(label(grid(strip, cols=len(strip)), f"cow #{args.burst}: burst, 1 of every 25 frames "
                                                        f"(margin {args.margins[0]:g})", 14))
    out = stack(parts)
    out.save(args.out)
    print(f"{args.out}  ({out.width}x{out.height}, {len(boxes)} cows)")


def cbvd_boxes_all(root):
    import cbvd
    return cbvd.load_boxes(os.path.join(root, "annotations", "ava_val_v2.1.csv")) + \
        cbvd.load_boxes(os.path.join(root, "annotations", "ava_train_v2.1.csv"))


if __name__ == "__main__":
    sys.exit(main())
