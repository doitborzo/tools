#!/usr/bin/env python3
"""What the working zone cuts away: a frame with every cow marked kept or dropped,
next to the frame the detector actually gets (the far zone greyed out).

    python herd/show_zone.py --root /workspace/cbvd5 --clip 371 --ts 5 --min-box-pct 0.8
    python herd/show_zone.py --root /workspace/cbvd5 --clip 371 --ts 5 --min-box-side-px 90
    python herd/show_zone.py --root /workspace/cbvd5 --clip 371 --ts 5 --detector /workspace/lora-runs/detector/best ...
    python herd/show_zone.py --image barn.jpg --boxes "0.1,0.2,0.3,0.5;..." --exclude "0,0 1,0 1,0.2 0,0.2" ...
    python herd/show_zone.py --config barn.toml --camera cam1 --image cam1.jpg --detector D   # a barn camera's own zone
    python herd/show_zone.py --root /workspace/cbvd5 --clip 371 --ts 5 --detector D --compare  # detector vs annotation

--compare (CBVD-5 frames with a detector): the annotation's boxes and the
detector's on one picture - what eval-det counts as found, missed and extra:
green  the detector's box on an annotated cow (IoU >= 0.5)
yellow found, but the boxes disagree (IoU 0.3-0.5): counted missed AND extra at 0.5
red    an annotated cow the detector has no box for (thin blue: the annotation)
magenta a detector box on no annotated cow (a cow the annotation left out, or a false box)

Colours: green - kept (tracked, identified); orange - dropped, too small
(min_box_pct: % of the frame / min_box_side_px: short side in pixels); red - dropped, centre in the excluded far zone.
Boxes are the annotation's, or the detector's with --detector. The same rule as
the barn (common.in_zone, common.blank_excluded).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import _polys, big_enough, blank_excluded, in_polygon, load_config, zone_min_area  # noqa: E402
from show_crops import font, label, stack  # noqa: E402

GREEN, ORANGE, RED, GREY = (40, 220, 60), (255, 150, 0), (235, 40, 40), (114, 114, 114)


def verdict(box, cam, w, h):
    """'kept', 'small' or 'far' - why a box is in or out of the camera's zone."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if any(in_polygon(cx, cy, p) for p in _polys(cam.get("exclude"))):
        return "far"
    if not big_enough(box, w, h, zone_min_area(cam), cam.get("min_box_side_px", 0)):
        return "small"
    return "kept"


def draw(img, boxes, cam, show_dropped=True, shade=True):
    h, w = img.shape[:2]
    pil = Image.fromarray(img).convert("RGB")
    if shade:
        over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        for p in _polys(cam.get("exclude")):
            d.polygon([(x * w, y * h) for x, y in p], fill=(235, 40, 40, 70), outline=(235, 40, 40, 255))
        pil = Image.alpha_composite(pil.convert("RGBA"), over).convert("RGB")
    d = ImageDraw.Draw(pil)
    lw = max(2, w // 400)
    f = font(max(14, w // 70))
    counts = {"kept": 0, "small": 0, "far": 0}
    for b in boxes:
        v = verdict(b, cam, w, h)
        counts[v] += 1
        if v != "kept" and not show_dropped:
            continue
        col = {"kept": GREEN, "small": ORANGE, "far": RED}[v]
        xy = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
        d.rectangle(xy, outline=col, width=lw if v == "kept" else max(1, lw - 1))
        if v != "kept":                                   # a cross: not tried
            d.line([xy[0], xy[1], xy[2], xy[3]], fill=col, width=max(1, lw - 1))
            d.line([xy[0], xy[3], xy[2], xy[1]], fill=col, width=max(1, lw - 1))
        side = min((b[2] - b[0]) * w, (b[3] - b[1]) * h)
        d.text((xy[0] + 2, xy[3] + 1), f"{(b[2] - b[0]) * (b[3] - b[1]) * 100:.2f}% {side:.0f}px", fill=col, font=f)
    return pil, counts


def compare(img, gts, args, name):
    import detect as detect_mod
    import detector as det_mod
    det = det_mod.Live(args.detector, args.threshold)
    found = det(Image.fromarray(img))
    dets = [c["bbox"] for c in found]
    good = detect_mod.match(dets, gts, 0.5)
    used_d, used_g = {i for i, _, _ in good}, {j for _, j, _ in good}
    rest_d = [i for i in range(len(dets)) if i not in used_d]
    rest_g = [j for j in range(len(gts)) if j not in used_g]
    loose = [(rest_d[i], rest_g[j], v) for i, j, v in
             detect_mod.match([dets[i] for i in rest_d], [gts[j] for j in rest_g], 0.3)]
    loose_d, loose_g = {i for i, _, _ in loose}, {j for _, j, _ in loose}
    h, w = img.shape[:2]
    pil = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(pil)
    lw, f = max(2, w // 450), font(max(14, w // 75))
    px = lambda b: [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    BLUE, YELLOW, MAGENTA = (60, 140, 255), (255, 220, 0), (230, 40, 230)
    for j, g in enumerate(gts):                                   # the annotation, thin blue
        d.rectangle(px(g), outline=RED if (j not in used_g and j not in loose_g) else BLUE, width=max(1, lw - 1))
    for i, gj, v in good:
        d.rectangle(px(dets[i]), outline=GREEN, width=lw)
    for i, gj, v in loose:
        d.rectangle(px(dets[i]), outline=YELLOW, width=lw)
        b = px(dets[i])
        d.text((b[0] + 3, b[1] + 3), f"IoU {v:.2f}", fill=YELLOW, font=f)
    for i in range(len(dets)):
        if i not in used_d and i not in loose_d:
            d.rectangle(px(dets[i]), outline=MAGENTA, width=lw)
    for j in range(len(gts)):
        if j not in used_g and j not in loose_g:
            b = px(gts[j])
            d.line([b[0], b[1], b[2], b[3]], fill=RED, width=lw)
            d.line([b[0], b[3], b[2], b[1]], fill=RED, width=lw)
    n_miss = len(gts) - len(good) - len(loose)
    n_extra = len(dets) - len(good) - len(loose)
    out = label(pil.resize((args.width, int(h * args.width / w))),
                f"{name}, threshold {det.threshold}: {len(gts)} annotated, {len(dets)} detected - found {len(good)} "
                f"(green), boxes disagree {len(loose)} (yellow), missed {n_miss} (red X), extra {n_extra} (magenta)", 15)
    out.save(args.out)
    print(f"{args.out}: annotated {len(gts)}, detected {len(dets)}, found {len(good)}, loose {len(loose)}, "
          f"missed {n_miss}, extra {n_extra}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--clip", default=None)
    p.add_argument("--ts", type=int, default=5)
    p.add_argument("--image", default=None)
    p.add_argument("--boxes", default=None, help='with --image: "x1,y1,x2,y2;..." as 0-1 fractions')
    p.add_argument("--detector", default=None, help="boxes from this detector (best/) instead of the annotation")
    p.add_argument("--config", default=None, help="take the zone from a barn config ...")
    p.add_argument("--camera", default=None, help="... for this camera id")
    p.add_argument("--exclude-above", type=float, default=None, help="far zone: everything above this height (0-1)")
    p.add_argument("--exclude", default=None, help='far zone polygon: "x,y x,y x,y ..." (0-1)')
    p.add_argument("--min-box-pct", type=float, default=None, help="drop boxes under this %% of the frame")
    p.add_argument("--min-box-area", type=float, default=None, help="the same as a share (0-1)")
    p.add_argument("--min-box-side-px", type=int, default=None)
    p.add_argument("--width", type=int, default=1280, help="width of each panel")
    p.add_argument("--out", default="zone.png")
    p.add_argument("--compare", action="store_true", help="annotation vs detector on one picture (needs --detector)")
    p.add_argument("--threshold", type=float, default=None, help="detector score cut-off (default: its own)")
    args = p.parse_args(argv)

    cam = {}
    if args.config:
        cfg = load_config(args.config)
        cam = dict(next((c for c in cfg["cameras"] if c["id"] == args.camera), cfg["cameras"][0]))
    if args.exclude_above is not None:
        cam["exclude"] = [[0, 0], [1, 0], [1, args.exclude_above], [0, args.exclude_above]]
    if args.exclude:
        cam["exclude"] = [[float(v) for v in pt.split(",")] for pt in args.exclude.split()]
    if args.min_box_pct is not None:
        cam["min_box_pct"] = args.min_box_pct
    if args.min_box_area is not None:
        cam["min_box_pct"] = args.min_box_area * 100
    if args.min_box_side_px is not None:
        cam["min_box_side_px"] = args.min_box_side_px

    if args.image:
        img = np.asarray(Image.open(args.image).convert("RGB"))
        boxes = [[float(v) for v in s.split(",")] for s in (args.boxes or "").split(";") if s.strip()]
        name = os.path.basename(args.image)
    else:
        import cbvd
        from show_crops import cbvd_boxes
        found = cbvd_boxes(args.root, args.clip, args.ts)
        img = np.asarray(Image.open(cbvd.frame_path(args.root, found[0])).convert("RGB"))
        boxes = [list(b.xyxy) for b in found]
        name = f"clip {args.clip} at {args.ts} s"
    src = "annotation"
    if args.compare:
        if not args.detector or args.image:
            sys.exit("--compare needs a CBVD-5 frame (--clip, --ts) and --detector")
        return compare(img, boxes, args, name)
    if args.detector:
        import detector as det_mod
        det = det_mod.Live(args.detector, args.threshold)
        boxes = [c["bbox"] for c in det(Image.fromarray(img))]           # all it finds, before the zone
        src = f"detector, threshold {det.threshold}"

    left, counts = draw(img, boxes, cam)
    right, _ = draw(blank_excluded(img, cam.get("exclude")), boxes, cam, show_dropped=False, shade=False)
    scale = lambda im: im.resize((args.width, int(im.height * args.width / im.width)))
    rule = []
    if cam.get("exclude"):
        rule.append("far zone (red) greyed out")
    if zone_min_area(cam):
        rule.append(f"boxes < {zone_min_area(cam) * 100:.2f}% of the frame dropped")
    if cam.get("min_box_side_px"):
        rule.append(f"boxes with a side < {cam['min_box_side_px']} px dropped")
    top = label(scale(left), f"{name} - boxes: {src}.  kept {counts['kept']} (green), too small {counts['small']} "
                             f"(orange), in the far zone {counts['far']} (red)", 16)
    bottom = label(scale(right), "what the detector gets and what is tracked: " + ("; ".join(rule) or "no zone set"), 16)
    out = stack([top, bottom])
    out.save(args.out)
    print(f"{args.out}: kept {counts['kept']}, too small {counts['small']}, far zone {counts['far']}")


if __name__ == "__main__":
    sys.exit(main())
