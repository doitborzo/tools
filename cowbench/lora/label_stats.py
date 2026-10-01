"""Compare how CBVD-5 train and val are annotated.

    python lora/label_stats.py /workspace/cbvd5

The LoRA trained on ava_train learned "standing at the feed barrier ->
ruminating" and "lying -> none", while in ava_val rumination is almost
always a lying cow. This prints, for each split:

  - posture x activity, over the boxes training and scoring use
  - the boxes cbvd.partition() drops, by label set
  - dropped boxes that sit on top of a kept box on the same keyframe: the
    sign of one cow's rows failing to group into one box (coordinates or
    entity id written differently), which leaves its posture half as
    activity "none"
  - raw action-id counts and how many CSV rows each box has
"""

from __future__ import annotations

import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cbvd  # noqa: E402


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def table(counter, rows, cols):
    w = max(len(r) for r in rows) + 2
    print(" " * w + "".join(f"{c:>12}" for c in cols) + f"{'total':>9}")
    for r in rows:
        n = [counter[(r, c)] for c in cols]
        print(f"{r:<{w}}" + "".join(f"{x:>12}" for x in n) + f"{sum(n):>9}")


def split_stats(path):
    print(f"\n=== {os.path.basename(path)}")
    raw = collections.Counter()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and not row[0].startswith("#"):
                raw[row[6]] += 1
    print("CSV rows per action id:",
          ", ".join(f"{k} {cbvd.LABEL_NAMES.get(int(k), '?')}: {v}"
                    for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))))

    boxes = cbvd.load_boxes(path)
    usable, rejected = cbvd.partition(boxes)
    rejected = [b for b, _reason in rejected]
    print(f"boxes: {len(boxes)}  usable: {len(usable)}  dropped: {len(rejected)}")
    print("label ids per box:", dict(sorted(collections.Counter(len(b.labels) for b in boxes).items())))

    print("\nposture x activity (usable boxes):")
    table(collections.Counter((b.posture, b.activity) for b in usable),
          list(cbvd.POSTURES), list(cbvd.ACTIVITIES))

    if rejected:
        print("\ndropped boxes by label set:")
        for labels, n in collections.Counter(b.labels for b in rejected).most_common(10):
            print(f"  {n:6}  {labels}  " + ", ".join(cbvd.LABEL_NAMES.get(l, '?') for l in labels))
        by_frame = collections.defaultdict(list)
        for b in usable:
            by_frame[(b.video_id, b.timestamp)].append(b)
        overlap = collections.Counter()
        for b in rejected:
            best = max(by_frame.get((b.video_id, b.timestamp), []),
                       key=lambda u: iou(u.xyxy, b.xyxy), default=None)
            if best is not None and iou(best.xyxy, b.xyxy) > 0.8:
                overlap[(b.labels, best.labels)] += 1
        if overlap:
            print("\ndropped boxes lying on a kept box (IoU > 0.8) - rows that did not group:")
            for (lab, kept), n in overlap.most_common(10):
                print(f"  {n:6}  dropped {lab} on kept {kept}")
        else:
            print("\nno dropped box lies on a kept box")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/cbvd5"
    for name in ("ava_train_v2.1.csv", "ava_val_v2.1.csv"):
        split_stats(os.path.join(root, "annotations", name))
    labelmap = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(root, "annotations"))
                for f in fs if "labelmap" in f.lower() or f.endswith(".pbtxt")]
    for p in labelmap:
        print(f"\n=== {p}")
        with open(p, encoding="utf-8", errors="replace") as fh:
            print(fh.read().strip()[:1500])


if __name__ == "__main__":
    main()
