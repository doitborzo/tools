#!/usr/bin/env python3
"""What the overlap rule does to the frame heads' scores: eval-det at IoU 0.5 vs 0.3.

eval-det matches the detector's boxes to the annotated cows at one overlap
(--iou). A cow found with a box that overlaps less than that counts as missed,
so as wrong. Scored again at a looser overlap, those cows are scored on what
the frame heads (DINOv2 + posture / activity heads) said about their box. This
reads the two eval-det folders and writes what changed:

    the error with missed = wrong, on found cows, posture / activity
    the cows found only at the looser overlap: how often the heads were right on them
    the threshold sweep at both

    python herd/compare_iou.py --strict R/eval-val-det-iou05 --loose R/eval-val-det-iou03 --out R/iou_effect
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def read(folder):
    res = json.load(open(os.path.join(folder, "eval_det.json"), encoding="utf-8"))
    with open(os.path.join(folder, "results.jsonl"), encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    return res, recs


def right(r):
    return {"posture": r.get("posture") == r["gt_posture"], "activity": r.get("activity") == r["gt_activity"],
            "exact": r.get("posture") == r["gt_posture"] and r.get("activity") == r["gt_activity"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strict", required=True, help="eval-det folder at IoU 0.5")
    p.add_argument("--loose", required=True, help="eval-det folder at IoU 0.3")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    (rs, recs_s), (rl, recs_l) = read(args.strict), read(args.loose)
    key = lambda r: (str(r["video_id"]), r["timestamp"], r["id"])
    strict, loose = {key(r): r for r in recs_s}, {key(r): r for r in recs_l}
    moved = [loose[k] for k, r in strict.items() if r.get("missed_by_detector") and k in loose
             and not loose[k].get("missed_by_detector")]
    lost = [k for k, r in strict.items() if not r.get("missed_by_detector") and k in loose
            and loose[k].get("missed_by_detector")]
    n = len(moved)
    share = lambda f: sum(right(r)[f] for r in moved) / n if n else None
    gained = {"cows": n, "posture_right": share("posture"), "activity_right": share("activity"),
              "exact_right": share("exact"),
              "det_iou_median": sorted(r["det_iou"] for r in moved)[n // 2] if n else None,
              "found_at_0.5_but_not_0.3": len(lost)}

    es, el = rs["on_detector_boxes"], rl["on_detector_boxes"]
    out = {"iou_strict": rs["iou"], "iou_loose": rl["iou"], "threshold": rs["threshold"],
           "detector": rs["detector"], "calibrated": rs.get("calibrated"),
           "detector_quality": rs["detector_quality"],
           "on_detector_boxes": {"strict": es, "loose": el},
           "on_annotated_boxes": rs.get("on_annotated_boxes"),
           "cows_found_only_at_loose": gained,
           "threshold_sweep": {"strict": rs["threshold_sweep"], "loose": rl["threshold_sweep"]}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "iou_effect.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"
    pts = lambda a, b: "-" if a is None or b is None else f"{100 * (b - a):+.1f} pts"
    q5, q3 = rs["detector_quality"]["iou0.5"], rs["detector_quality"]["iou0.3"]
    L = [f"# Overlap rule: IoU {rs['iou']} vs {rl['iou']} (frame heads on the detector's boxes)", "",
         f"Detector `{rs['detector']}`, threshold {rs['threshold']}"
         f"{', boxes calibrated' if rs.get('calibrated') else ''}; "
         f"{rs['detector_quality']['frames']} val frames, {rs['detector_quality']['cows']} annotated cows. "
         "No training: the same detector and model scored twice.", "",
         "## Detector", "",
         "| | IoU 0.5 | IoU 0.3 |", "|---|---|---|",
         f"| Found | {q5['found']} | {q3['found']} |", f"| Missed | {q5['missed']} | {q3['missed']} |",
         f"| Extra boxes | {q5['extra']} | {q3['extra']} |",
         f"| Recall | {pct(q5['recall'])} | {pct(q3['recall'])} |",
         f"| Precision | {pct(q5['precision'])} | {pct(q3['precision'])} |", "",
         "## Frame heads (DINOv2 + posture / activity) on the detector's boxes", "",
         f"| | IoU {rs['iou']} | IoU {rl['iou']} | change |", "|---|---|---|---|"]
    for k, name in (("exact_error", "Error, missed cows = wrong"), ("posture_error", "  posture"),
                    ("activity_error", "  activity"), ("exact_error_found_cows", "Error on found cows")):
        L.append(f"| {name} | {pct(es[k])} | {pct(el[k])} | {pts(es[k], el[k])} |")
    L.append(f"| Missed (counted wrong) | {es['missed']} | {el['missed']} | {el['missed'] - es['missed']:+d} |")
    if rs.get("on_annotated_boxes"):
        L.append(f"| Error on annotated boxes (no detector) | {pct(rs['on_annotated_boxes']['exact_error'])} | "
                 f"{pct(rs['on_annotated_boxes']['exact_error'])} | same |")
    L += ["", f"## The {n} cows found only at IoU {rl['iou']}", "",
          "Missed at the strict overlap (counted wrong there); at the loose one the heads' answer on the "
          "detector's box is scored.", "",
          f"- right on both posture and activity: {pct(gained['exact_right'])}",
          f"- posture right: {pct(gained['posture_right'])}, activity right: {pct(gained['activity_right'])}",
          f"- median overlap of their boxes: {gained['det_iou_median']}",
          f"- found at {rs['iou']} but not at {rl['iou']} (normally 0: the looser rule can re-pair boxes): {len(lost)}", "",
          "## Threshold sweep", "",
          f"| threshold | error @{rs['iou']} | error @{rl['iou']} | on found @{rs['iou']} | on found @{rl['iou']} |",
          "|---|---|---|---|---|"]
    for a, b in zip(rs["threshold_sweep"], rl["threshold_sweep"]):
        L.append(f"| {a['threshold']} | {pct(a['exact_error'])} | {pct(b['exact_error'])} | "
                 f"{pct(a['exact_error_found_cows'])} | {pct(b['exact_error_found_cows'])} |")
    with open(os.path.join(args.out, "iou_effect.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
