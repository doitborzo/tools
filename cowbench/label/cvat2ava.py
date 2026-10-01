#!/usr/bin/env python3
"""A CVAT export of a task pre-labelled by prelabel.py -> CBVD-5's own format,
so cowbench, detector.py and lora/train_lora.py read the new video as they
read CBVD-5.

    python label/cvat2ava.py export.zip --root newfarm            # -> ava_val_v2.1.csv
    python label/cvat2ava.py export.zip --root newfarm --split train

export.zip is CVAT's "Export task dataset" (or annotations) in the format
"CVAT for video 1.1"; the annotations.xml inside it may be given directly.
--root is prelabel.py's --out: its keyframes.json maps CVAT frame numbers to
clips and seconds, and its keyframes are already in labelframes/.

Rows are AVA, as CBVD-5's: clip, second, x1, y1, x2, y2 (0-1), action id,
entity id - one row per label, posture (1 stand, 2 lying) always, activity
(3 feeding, 4 drinking, 5 ruminating) when there is one. The entity id is the
CVAT track: the same cow keeps it from clip to clip (in CBVD-5 it is always 1).

A cow whose posture or activity is still "?" has not been looked at. Its whole
keyframe is left out - a keyframe with an unlabelled cow would teach the
frame question to skip cows - and the keyframes left out are listed.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

POSTURE_ID = {"standing": 1, "lying": 2}
ACTIVITY_ID = {"feeding": 3, "drinking": 4, "ruminating": 5, "none": None}
LABELMAP = "".join(f'item {{\n name: "{n}"\n id: {i}\n}}\n' for i, n in
                   enumerate(("stand", "lying down", "foraging", "drinking water", "rumination"), 1))


def read_xml(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("annotations.xml")] or \
                    [n for n in z.namelist() if n.endswith(".xml")]
            if not names:
                sys.exit(f"no annotations.xml in {path}")
            return ET.parse(io.BytesIO(z.read(names[0]))).getroot()
    return ET.parse(path).getroot()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", help="CVAT export (.zip or annotations.xml), format CVAT for video 1.1")
    p.add_argument("--root", required=True, help="prelabel.py's --out")
    p.add_argument("--split", default="val", choices=("val", "train"),
                   help="val: a test set the bench reads as is; train: for training")
    args = p.parse_args(argv)

    with open(os.path.join(args.root, "keyframes.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    kf = {k["frame"]: k for k in meta["keyframes"]}
    w, h = meta["width"], meta["height"]
    root = read_xml(args.export)
    if root.find("track") is None and root.find("image") is not None:
        sys.exit("this export has <image> elements, not <track>: export as 'CVAT for video 1.1'")

    cows = collections.defaultdict(list)    # keyframe -> [(track, box, posture, activity)]
    for t_index, track in enumerate(root.iter("track"), 1):
        if track.get("label") != "cow":
            continue
        for b in track.iter("box"):
            fr = int(b.get("frame"))
            if fr not in kf or b.get("outside") == "1":
                continue
            attrs = {a.get("name"): (a.text or "").strip() for a in b.iter("attribute")}
            box = (max(0.0, float(b.get("xtl")) / w), max(0.0, float(b.get("ytl")) / h),
                   min(1.0, float(b.get("xbr")) / w), min(1.0, float(b.get("ybr")) / h))
            cows[fr].append((t_index, box, attrs.get("posture", "?"), attrs.get("activity", "?")))

    rows, skipped, counts = [], [], collections.Counter()
    for fr in sorted(kf):
        entries = cows.get(fr, [])
        unseen = [(t, po, ac) for t, _, po, ac in entries
                  if po not in POSTURE_ID or ac not in ACTIVITY_ID]
        if unseen:
            skipped.append((kf[fr], unseen))
            continue
        k = kf[fr]
        for t, (x1, y1, x2, y2), po, ac in entries:
            base = [k["video_id"], str(k["timestamp"]), f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}"]
            rows.append(base + [str(POSTURE_ID[po]), str(t)])
            if ACTIVITY_ID[ac]:
                rows.append(base + [str(ACTIVITY_ID[ac]), str(t)])
            counts[po] += 1
            counts[ac] += 1
            counts["cows"] += 1

    ann = os.path.join(args.root, "annotations")
    os.makedirs(ann, exist_ok=True)
    out = os.path.join(ann, f"ava_{args.split}_v2.1.csv")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        for r in rows:
            fh.write(",".join(r) + "\n")
    with open(os.path.join(ann, "labelmap.txt"), "w", encoding="utf-8") as fh:
        fh.write(LABELMAP)

    n_kf = len(kf) - len(skipped)
    print(f"{counts['cows']} cows on {n_kf} of {len(kf)} keyframes -> {out}")
    print("  posture : " + ", ".join(f"{v} {counts[v]}" for v in ("standing", "lying")))
    print("  activity: " + ", ".join(f"{v} {counts[v]}" for v in ("none", "feeding", "drinking", "ruminating")))
    if skipped:
        print(f"!! {len(skipped)} keyframes left out: a cow there still has posture or activity '?'")
        for k, unseen in skipped[:15]:
            print(f"   frame {k['frame']} (clip {k['video_id']}, {k['second']} s): "
                  + "; ".join(f"track {t}: {po}/{ac}" for t, po, ac in unseen[:4]))
        if len(skipped) > 15:
            print(f"   ... and {len(skipped) - 15} more")


if __name__ == "__main__":
    main()
