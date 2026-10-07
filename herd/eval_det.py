"""herd on the detector's boxes: the whole once-a-second path, detector error included.

`herd.py eval` scores the heads on the annotated boxes - how good they are
when the cow is found. In the barn nobody draws the boxes: RT-DETRv2 does,
and a cow it misses has no posture or activity at all. Here the detector
runs on every val keyframe (the same 2532 cows), the frame heads answer
about its boxes, and the answers are matched to the annotated cows by
overlap (IoU >= 0.5, greedy, as cowbench does for the LoRA runs):

    a found cow     scored on the detector's box (its crop, its position)
    a missed cow    an error ("missed by detector")
    an extra box    a detection that is no annotated cow - counted on its own

    python herd/herd.py eval-det --run /workspace/herd/run5 --detector /workspace/lora-runs/detector/best
    python cowbench/cowbench.py --out /workspace/herd/run5/eval-val-det score   # then report, summary.py

Writes <run>/eval-val-det/results.jsonl + run_meta.json (cowbench format:
the "missed by detector" and "error on found cows" columns) and
<run>/eval_det.json: detector recall / precision at IoU 0.5 and 0.3, by cow
size, and the error with and without the detector.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ACTIVITIES, POSTURES, crop, device, write_json  # noqa: E402

import cbvd  # noqa: E402  (cowbench)
import detect as detect_mod  # noqa: E402  (cowbench: greedy IoU matching)


def val_keyframes(root):
    """{(clip, ts): [annotated cows]} of CBVD-5 val, as herd labels them."""
    from cbvd_bursts import labels_of, split_boxes
    frames = collections.defaultdict(list)
    for b in split_boxes(root, "val"):
        lab = labels_of(b)
        frames[(b.video_id, b.timestamp)].append({
            "id": b.uid, "video_id": b.video_id, "timestamp": b.timestamp, "bbox": list(b.xyxy),
            "gt_posture": POSTURES[lab["posture"]] if lab["posture"] >= 0 else None,
            "gt_activity": ACTIVITIES[lab["activity"]], "gt_rumination": bool(lab["rumination"])})
    return dict(sorted(frames.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])))


def detector_quality(frames, found):
    """frames {(clip, ts): [gt cows]}, found {(clip, ts): [boxes]} -> recall,
    precision at IoU 0.5 and 0.3, misses by cow size (quartiles of box area)."""
    out = {}
    for thr in (0.5, 0.3):
        tp = fp = fn = 0
        for key, gts in frames.items():
            pred = found.get(key, [])
            k = len(detect_mod.match(pred, [g["bbox"] for g in gts], thr))
            tp, fp, fn = tp + k, fp + len(pred) - k, fn + len(gts) - k
        out[f"iou{thr}"] = {"recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp),
                            "found": tp, "missed": fn, "extra": fp}
    areas, hit = [], []
    for key, gts in frames.items():
        pairs = detect_mod.match(found.get(key, []), [g["bbox"] for g in gts], 0.5)
        got = {j for _, j, _ in pairs}
        for j, g in enumerate(gts):
            b = g["bbox"]
            areas.append((b[2] - b[0]) * (b[3] - b[1]))
            hit.append(j in got)
    if areas:
        qs = np.quantile(areas, [0.25, 0.5, 0.75])
        bins = np.digitize(areas, qs)
        out["recall_by_size"] = {f"Q{q + 1} ({'smallest' if q == 0 else 'largest' if q == 3 else 'mid'})":
                                 float(np.mean([h for h, b in zip(hit, bins) if b == q]))
                                 for q in range(4) if any(b == q for b in bins)}
    out["frames"] = len(frames)
    out["cows"] = sum(len(v) for v in frames.values())
    return out


def errors(records):
    """exact / posture / activity error over records (missed = wrong), and on found cows only."""
    def rate(rows, f):
        return float(np.mean([f(r) for r in rows])) if rows else None
    exact = lambda r: r.get("posture") != r["gt_posture"] or r.get("activity") != r["gt_activity"]
    found = [r for r in records if not r.get("missed_by_detector")]
    return {"exact_error": rate(records, exact),
            "posture_error": rate(records, lambda r: r.get("posture") != r["gt_posture"]),
            "activity_error": rate(records, lambda r: r.get("activity") != r["gt_activity"]),
            "exact_error_found_cows": rate(found, exact),
            "n": len(records), "missed": len(records) - len(found)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the run folder (model.pt)")
    p.add_argument("--detector", default="/workspace/lora-runs/detector/best")
    p.add_argument("--threshold", type=float, default=None, help="detector score cut-off (default: its own)")
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", default=None, help="default <run>/eval-val-det")
    p.add_argument("--tiles", type=int, default=None,
                   help="1: whole frame + square tiles (far cows ~1.8x wider); default: as the detector was trained")
    p.add_argument("--sweep-from", type=float, default=0.15, help="lowest threshold of the sweep")
    args = p.parse_args(argv)

    import torch
    import detector as det_mod
    from model import FrameEncoder, HerdModel, box_pos
    from cbvd_bursts import cbvd_frame

    dev = device()
    model, ck = HerdModel.load(os.path.join(args.run, "model.pt"), map_location=dev)
    model.to(dev).eval()
    feat = ck.get("features", {})
    size, margin = feat.get("crop", 224), feat.get("margin", 0.1)
    enc = FrameEncoder(feat.get("encoder", "facebook/dinov2-small"), feat.get("grid", 2)).to(dev).eval()
    meta = json.load(open(os.path.join(args.detector, "det_train_meta.json"), encoding="utf-8"))
    threshold = meta["threshold"] if args.threshold is None else args.threshold
    sweep = sorted({round(t, 2) for t in np.arange(args.sweep_from, 0.651, 0.05)} | {round(threshold, 2)})
    # one pass at the lowest cut-off of the sweep: every box classified once, then cut per threshold
    det = det_mod.Live(args.detector, min(sweep), tiles=args.tiles)
    out_dir = args.out or os.path.join(args.run, "eval-val-det")
    os.makedirs(out_dir, exist_ok=True)

    frames = val_keyframes(args.root)
    print(f"[eval-det] {len(frames)} val keyframes, {sum(len(v) for v in frames.values())} annotated cows; "
          f"detector {det.name} on {det.device}, threshold {threshold}{', tiles' if det.tiles else ''}", flush=True)
    seen, det_ms = {}, []                                # (clip, ts) -> [(box, score, posture, activity)]
    for n, (clip, ts) in enumerate(frames, 1):
        img = cbvd_frame(args.root, clip, ts)
        t = time.perf_counter()
        dets = det(img)
        det_ms.append((time.perf_counter() - t) * 1000)
        boxes = [c["bbox"] for c in dets]
        answers = []
        if boxes:
            arr = np.asarray(img)
            x = torch.from_numpy(np.stack([crop(arr, b, size, margin) for b in boxes])).to(dev)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    f = enc(x).float()
                o = model.frame(f, torch.tensor([box_pos(b) for b in boxes], device=dev).float())
            answers = list(zip(o["posture"].argmax(-1).tolist(), o["activity"].argmax(-1).tolist()))
        seen[(clip, ts)] = [(c["bbox"], c["det_score"], POSTURES[p_], ACTIVITIES[a_])
                            for c, (p_, a_) in zip(dets, answers)]
        print(f"\r  {n}/{len(frames)} frames", end="", flush=True)
    print(flush=True)

    def at(thr):
        """records (one per annotated cow) and found boxes at a score cut-off"""
        records, found = [], {}
        for key, gts in frames.items():
            mine = [d for d in seen[key] if d[1] >= thr]
            found[key] = [d[0] for d in mine]
            pairs = detect_mod.match(found[key], [g["bbox"] for g in gts], args.iou)
            by_gt = {j: (i, v) for i, j, v in pairs}
            for j, g in enumerate(gts):
                if j in by_gt:
                    i, v = by_gt[j]
                    records.append(dict(g, posture=mine[i][2], activity=mine[i][3], det_bbox=mine[i][0],
                                        det_score=mine[i][1], det_iou=round(v, 3)))
                else:
                    records.append(dict(g, posture=None, activity=None, missed_by_detector=True,
                                        parse_error="not found by the detector"))
        return records, found

    table = []
    for thr in sweep:
        recs, fnd = at(thr)
        q = detector_quality(frames, fnd)["iou0.5"]
        table.append({"threshold": thr, **errors(recs), "recall": q["recall"], "precision": q["precision"],
                      "extra": q["extra"]})
    best = min(table, key=lambda r: (r["exact_error"], -r["threshold"]))
    records, found = at(threshold)

    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    write_json(os.path.join(out_dir, "run_meta.json"), {
        "model": "herd: DINOv2-S frame heads (1 fps) on RT-DETRv2 boxes",
        "engine": "herd, PyTorch", "reasoning": "n/a", "temperature": 0.0, "seed": 0,
        "render_mode": "224 px crop of the detected cow", "unit": "cow", "frames": 1, "max_width": 224,
        "annotations": "annotations/ava_val_v2.1.csv",
        "boxes": {"source": "detector", "detections": f"live: {os.path.abspath(args.detector)}",
                  "threshold": threshold, "tiles": det.tiles, "detector": det.name, "match_iou": args.iou,
                  "detector_ms_per_frame": round(float(np.median(det_ms)), 1) if det_ms else None},
        "prompt_sha": "herd", "videos": "", "excluded": []})
    quality = detector_quality(frames, found)
    quality["ms_per_frame_p50"] = round(float(np.median(det_ms)), 1) if det_ms else None
    anno_path = os.path.join(args.run, "eval-val", "results.jsonl")
    on_anno = None
    if os.path.exists(anno_path):
        with open(anno_path, encoding="utf-8") as fh:
            on_anno = errors([json.loads(line) for line in fh if line.strip()])
    res = {"detector": os.path.abspath(args.detector), "threshold": threshold, "tiles": det.tiles, "iou": args.iou,
           "detector_quality": quality, "on_detector_boxes": errors(records), "on_annotated_boxes": on_anno,
           "threshold_sweep": table, "best_threshold": best["threshold"]}
    write_json(os.path.join(out_dir, "eval_det.json"), res)
    if args.out is None:                       # the run's own detector figures, next to eval_val.json
        write_json(os.path.join(args.run, "eval_det.json"), res)

    q5, e = quality["iou0.5"], res["on_detector_boxes"]
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    print(f"[eval-det] detector: recall {pct(q5['recall'])}, precision {pct(q5['precision'])} at IoU 0.5 - "
          f"{q5['missed']} of {quality['cows']} cows missed, {q5['extra']} extra boxes; "
          f"{quality['ms_per_frame_p50']} ms a frame")
    print(f"[eval-det] on detector boxes: exact error {pct(e['exact_error'])} (posture {pct(e['posture_error'])}, "
          f"activity {pct(e['activity_error'])}); on the cows it found {pct(e['exact_error_found_cows'])}"
          + (f"; on annotated boxes {pct(on_anno['exact_error'])}" if on_anno else ""))
    print("[eval-det] threshold sweep (the whole 1 fps path: missed cows are errors):")
    print("   threshold  exact error  on found  recall  precision  extra boxes")
    for r in table:
        print(f"   {r['threshold']:9.2f}  {r['exact_error']:11.1%}  {r['exact_error_found_cows'] or 0:8.1%}  "
              f"{r['recall']:6.1%}  {r['precision']:9.1%}  {r['extra']:11d}"
              + ("   <- lowest error" if r is best else "") + ("   <- in use" if r["threshold"] == round(threshold, 2) else ""))
    print(f"[eval-det] cowbench format: python cowbench/cowbench.py --out {out_dir} score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
