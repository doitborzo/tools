#!/usr/bin/env python3
"""Box calibration: the detector's boxes reshaped to the annotation's habit.

CBVD-5's annotators drew boxes with a margin; the detector draws them tight. For a
large cow the two still overlap well (IoU > 0.5), for a small far one the margin
eats the overlap and a cow the detector found counts as missed - and the frame
heads, trained on annotated crops, get tighter crops in the barn than they learnt
on. So: on keyframes the detector has not trained on (its own held-out clips of
CBVD-5 train, never val), match its boxes to the annotated ones (IoU >= 0.3) and
measure, per size of box (five size bands by log area), the median ratio of the
annotation's width and height to the detector's (sx, sy) and the shift of the
centre (dx, dy, in box widths / heights). Written to <detector>/box_calib.json;
detector.Live applies it to every box from then on (pipeline, keys, eval-det,
show_zone) - eval-det --calib 0 scores without it.

    python herd/herd.py calibrate --detector /workspace/herd/detector_rfdetr_large/best --root /workspace/cbvd5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_json  # noqa: E402

import detect as detect_mod  # noqa: E402  (cowbench)


def fit(pairs, bands=5):
    """pairs [(detector box, annotated box)] -> {"buckets": [{log_area, sx, sy, dx, dy, n}]}."""
    rows = []
    for d, g in pairs:
        dw, dh = d[2] - d[0], d[3] - d[1]
        gw, gh = g[2] - g[0], g[3] - g[1]
        if dw <= 0 or dh <= 0:
            continue
        rows.append((np.log(dw * dh), gw / dw, gh / dh, ((g[0] + g[2]) - (d[0] + d[2])) / 2 / dw,
                     ((g[1] + g[3]) - (d[1] + d[3])) / 2 / dh))
    rows = np.array(rows)
    order = np.argsort(rows[:, 0])
    out = []
    for part in np.array_split(rows[order], min(bands, max(1, len(rows) // 20))):
        out.append({"log_area": float(np.median(part[:, 0])), "sx": float(np.median(part[:, 1])),
                    "sy": float(np.median(part[:, 2])), "dx": float(np.median(part[:, 3])),
                    "dy": float(np.median(part[:, 4])), "n": int(len(part))})
    return {"buckets": out}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True, help="the detector's best/ folder (box_calib.json goes there)")
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--threshold", type=float, default=None, help="detector cut-off (default: its own)")
    p.add_argument("--holdout", type=float, default=0.1, help="the detector's held-out share of train clips")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args(argv)
    out_path = os.path.join(args.detector, "box_calib.json")
    if os.path.exists(out_path) and not args.fresh:
        print(f"[calibrate] done already: {out_path} (--fresh to redo)")
        return 0

    import detector as det_mod
    import train_lora
    det = det_mod.Live(args.detector, args.threshold, calib=False)
    rows, _ = train_lora.load_train(args.root)
    _, dev_rows = train_lora.split_dev(rows, args.holdout, args.seed)
    frames = det_mod.frames_of(dev_rows)
    print(f"[calibrate] {len(frames)} held-out train keyframes ({len(dev_rows)} cows), detector {det.name}, "
          f"threshold {det.threshold}", flush=True)
    pairs, found = [], {}
    for n, f in enumerate(frames, 1):
        boxes = [c["bbox"] for c in det(det_mod.load_image(args.root, f))]
        found[(f["video_id"], f["timestamp"])] = boxes
        for i, j, _ in detect_mod.match(boxes, f["boxes"], 0.3):
            pairs.append((boxes[i], f["boxes"][j]))
        print(f"\r  {n}/{len(frames)} frames, {len(pairs)} matched cows", end="", flush=True)
    print(flush=True)
    if len(pairs) < 20:
        sys.exit("too few matched cows to calibrate")
    calib = fit(pairs)

    def recall(cal, iou):
        tp = n = 0
        for f in frames:
            boxes = [det_mod.calibrate_box(b, cal) for b in found[(f["video_id"], f["timestamp"])]]
            tp += len(detect_mod.match(boxes, f["boxes"], iou))
            n += len(f["boxes"])
        return tp / max(1, n)

    mean_iou = lambda cal: float(np.mean([detect_mod.iou(det_mod.calibrate_box(d, cal), g) for d, g in pairs]))
    calib.update(detector=os.path.abspath(args.detector), threshold=det.threshold, frames=len(frames),
                 matched=len(pairs), mean_iou_before=mean_iou(None), mean_iou_after=mean_iou(calib),
                 recall_iou05_before=recall(None, 0.5), recall_iou05_after=recall(calib, 0.5))
    write_json(out_path, calib)
    for b in calib["buckets"]:
        print(f"  box area ~{np.exp(b['log_area']) * 100:.2f}% of frame: annotation {b['sx']:.2f}x wider, "
              f"{b['sy']:.2f}x taller, centre shifted {b['dx']:+.2f} w, {b['dy']:+.2f} h  ({b['n']} cows)")
    print(f"[calibrate] mean IoU with the annotation {calib['mean_iou_before']:.3f} -> {calib['mean_iou_after']:.3f}; "
          f"recall at IoU 0.5 {calib['recall_iou05_before']:.1%} -> {calib['recall_iou05_after']:.1%} "
          f"(held-out train clips) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
