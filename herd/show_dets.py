#!/usr/bin/env python3
"""Every box a detector draws on a set of pictures, as one HTML gallery.

    python herd/herd.py show-dets --src /workspace/mydata --detector /workspace/herd/detector_rfdetr_large/best
    python herd/herd.py show-dets --src /workspace/extra_det/mmcows/valid --detector OLD/best --detector NEW/best
    python herd/herd.py show-dets --src barn_cam1.mp4 --every-sec 2 --detector D
    python herd/herd.py show-dets --cbvd /workspace/cbvd5 --detector D --pick worst     # CBVD-5 val keyframes
    python herd/herd.py show-dets --cbvd /workspace/cbvd5 --cbvd-split train --detector D --max 200

--src: a folder (searched recursively for .jpg/.jpeg/.png/.bmp/.webp pictures and
.mp4/.avi/.mkv/.mov videos), or single files; a video gives a frame every
--every-sec seconds. Annotation found next to the pictures is used:
  COCO  _annotations.coco.json in the pictures' folder (Roboflow, extra_det.py);
  YOLO  <name>.txt beside the picture or in a sibling labels/ folder (images/ -> labels/).
With an annotation every picture shows what eval-det counts:
  green    the detector's box on an annotated cow (IoU >= 0.5)
  yellow   found, but the boxes disagree (IoU 0.3-0.5)
  red X    an annotated cow with no box (thin red: the annotation)
  magenta  a box on no annotated cow (a cow the annotation left out, or a false box)
Without one: every box in cyan with its score.

Several --detector: one panel per detector, one above the other, on each picture.
Out: <out>/index.html (open in a browser; the pictures are in <out>/img/),
<out>/summary.json (per picture and in total: boxes, and with an annotation found,
missed, extra, recall, precision). --calib 0 by default: the boxes as the detector
draws them (box_calib.json is fitted to CBVD-5's annotation, not to other sets).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402,F401  (puts cowbench on the path)
from show_crops import font, label, stack  # noqa: E402

PICS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEOS = (".mp4", ".avi", ".mkv", ".mov")
GREEN, YELLOW, RED, MAGENTA, CYAN = (40, 220, 60), (255, 220, 0), (235, 40, 40), (230, 40, 230), (0, 220, 255)


# ------------------------------------------------------------- the pictures

def coco_boxes(folder):
    """folder/_annotations.coco.json -> {file name: [[x1, y1, x2, y2] 0-1]} (every category: cow)."""
    path = os.path.join(folder, "_annotations.coco.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    ims = {im["id"]: im for im in d["images"]}
    out = {im["file_name"]: [] for im in d["images"]}
    for a in d["annotations"]:
        im = ims[a["image_id"]]
        x, y, w, h = a["bbox"]
        out[im["file_name"]].append([x / im["width"], y / im["height"], (x + w) / im["width"], (y + h) / im["height"]])
    return out


def yolo_boxes(path):
    """The YOLO label of a picture, if there is one: beside it, or images/ -> labels/."""
    stem = os.path.splitext(path)[0]
    cands = [stem + ".txt"]
    parts = stem.split(os.sep)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        cands.append(os.sep.join(parts[:i] + ["labels"] + parts[i + 1:]) + ".txt")
    for c in cands:
        if os.path.exists(c):
            out = []
            for line in open(c, encoding="utf-8", errors="replace"):
                p = line.split()
                if len(p) == 5:
                    cx, cy, w, h = map(float, p[1:])
                    out.append([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)])
            return out
    return None


def find_sources(srcs):
    """--src paths -> (pictures [(path, annotation or None)], videos [path])."""
    pics, vids = [], []
    for s in srcs:
        if os.path.isfile(s):
            files = [s]
        else:
            files = sorted(os.path.join(r, f) for r, _, fs in os.walk(s) for f in fs)
        coco = {}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEOS:
                vids.append(f)
            elif ext in PICS:
                d = os.path.dirname(f)
                if d not in coco:
                    coco[d] = coco_boxes(d)
                ann = coco[d].get(os.path.basename(f)) if coco[d] is not None else yolo_boxes(f)
                pics.append((f, ann))
    return pics, vids


def video_frames(path, every_sec, limit):
    """[(name, PIL image)] - a frame every every_sec seconds."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps * every_sec))
    out = []
    for i in range(0, n if n > 0 else 10 ** 9, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok or len(out) >= limit:
            break
        out.append((f"{os.path.basename(path)}@{i / fps:.1f}s", Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def cbvd_items(root, split="val"):
    """CBVD-5 keyframes with their annotated boxes: val (never trained on), or train
    (the detector's own training frames, the val clips left out as in training)."""
    import cbvd
    import train_lora
    rows = cbvd.load_boxes(os.path.join(root, "annotations", f"ava_{split}_v2.1.csv"))
    if split == "train":
        val = {b.video_id for b in cbvd.load_boxes(os.path.join(root, "annotations", "ava_val_v2.1.csv"))}
        rows = [b for b in rows if b.video_id not in val]
    by = {}
    for b in rows:
        by.setdefault((b.video_id, b.timestamp), []).append(list(b.xyxy))
    out = []
    for (v, t), boxes in sorted(by.items(), key=lambda x: (int(x[0][0]) if str(x[0][0]).isdigit() else 0, x[0])):
        load = lambda v=v, t=t: train_lora.read_frame(root, cbvd.Box(v, t, 0, 0, 0, 0, "1", ())).convert("RGB")
        out.append((f"cbvd_{v}_{t:05d}", load, boxes))
    return out


# --------------------------------------------------------------- scoring

def score(dets, gts):
    """dets [[x1, y1, x2, y2, score]], gts [[...]] -> what each box is (eval-det's matching)."""
    import detect as detect_mod
    boxes = [d[:4] for d in dets]
    good = detect_mod.match(boxes, gts, 0.5)
    used_d, used_g = {i for i, _, _ in good}, {j for _, j, _ in good}
    rest_d = [i for i in range(len(boxes)) if i not in used_d]
    rest_g = [j for j in range(len(gts)) if j not in used_g]
    loose = [(rest_d[i], rest_g[j], v) for i, j, v in
             detect_mod.match([boxes[i] for i in rest_d], [gts[j] for j in rest_g], 0.3)]
    loose_d, loose_g = {i for i, _, _ in loose}, {j for _, j, _ in loose}
    return {"good": good, "loose": loose,
            "extra": [i for i in rest_d if i not in loose_d],
            "missed": [j for j in rest_g if j not in loose_g]}


def paint(img, dets, gts, title, width):
    """One panel: the picture with the detector's boxes (and the annotation), titled."""
    w, h = img.size
    pil = img.copy()
    d = ImageDraw.Draw(pil)
    lw, f = max(2, w // 450), font(max(12, w // 90))
    px = lambda b: [b[0] * w, b[1] * h, b[2] * w, b[3] * h]

    def tag(b, text, colour):
        th = d.textbbox((0, 0), text, font=f)[3] + 4
        x, y = b[0] + 2, (b[1] - th if b[1] >= th else b[1] + 2)
        l, t, r, bt = d.textbbox((x, y), text, font=f)
        d.rectangle([l - 1, t - 1, r + 1, bt + 1], fill=(0, 0, 0))
        d.text((x, y), text, fill=colour, font=f)

    stats = {"boxes": len(dets)}
    if gts is None:
        for b in dets:
            d.rectangle(px(b), outline=CYAN, width=lw)
            tag(px(b), f"{b[4]:.2f}", CYAN)
        line = f"{title}: {len(dets)} boxes"
    else:
        s = score(dets, gts)
        missed = set(s["missed"])
        for j, g in enumerate(gts):
            d.rectangle(px(g), outline=RED if j in missed else (60, 140, 255), width=max(1, lw - 1))
        for i, _, _ in s["good"]:
            d.rectangle(px(dets[i]), outline=GREEN, width=lw)
            tag(px(dets[i]), f"{dets[i][4]:.2f}", GREEN)
        for i, _, v in s["loose"]:
            d.rectangle(px(dets[i]), outline=YELLOW, width=lw)
            tag(px(dets[i]), f"{dets[i][4]:.2f} IoU {v:.2f}", YELLOW)
        for i in s["extra"]:
            d.rectangle(px(dets[i]), outline=MAGENTA, width=lw)
            tag(px(dets[i]), f"{dets[i][4]:.2f}", MAGENTA)
        for j in missed:
            b = px(gts[j])
            d.line([b[0], b[1], b[2], b[3]], fill=RED, width=lw)
            d.line([b[0], b[3], b[2], b[1]], fill=RED, width=lw)
        stats.update(annotated=len(gts), found=len(s["good"]), loose=len(s["loose"]),
                     missed=len(s["missed"]), extra=len(s["extra"]))
        line = (f"{title}: {len(gts)} annotated, {len(dets)} boxes - found {stats['found']} (green), "
                f"boxes disagree {stats['loose']} (yellow), missed {stats['missed']} (red X), "
                f"extra {stats['extra']} (magenta)")
    if w > width:
        pil = pil.resize((width, round(h * width / w)), Image.BILINEAR)
    return label(pil, line, 15), stats


# ------------------------------------------------------------------ html

def write_html(out, cards, totals, title):
    rows = []
    for c in cards:
        meta = "<br>".join(
            f"{html.escape(k)}: found {s['found']}, disagree {s['loose']}, missed {s['missed']}, extra {s['extra']}"
            if "annotated" in s else f"{html.escape(k)}: {s['boxes']} boxes" for k, s in c["stats"].items())
        rows.append(f'<a class="card" href="img/{html.escape(c["file"])}" target="_blank">'
                    f'<img loading="lazy" src="img/{html.escape(c["file"])}" alt="">'
                    f'<div class="name">{html.escape(c["name"])}</div><div class="meta">{meta}</div></a>')
    tot = "".join(f"<li><b>{html.escape(k)}</b>: " + ", ".join(f"{a} {v:.1%}" if isinstance(v, float) else f"{a} {v}"
                                                           for a, v in t.items()) + "</li>" for k, t in totals.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Detector boxes</title>
<style>
:root {{ --bg:#f6f6f4; --fg:#1d1d1b; --card:#fff; --muted:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161616; --fg:#eee; --card:#222; --muted:#aaa; --line:#333; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }}
h1 {{ font-size:20px; margin:0 0 8px; }} ul {{ margin:4px 0 12px; padding-left:18px; }}
.key span {{ display:inline-block; margin-right:14px; }} .key i {{ display:inline-block; width:12px; height:12px;
  vertical-align:middle; margin-right:4px; border-radius:2px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 420px), 1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden;
  color:inherit; text-decoration:none; }}
.card img {{ width:100%; display:block; }} .name {{ padding:6px 8px 0; font-weight:600; word-break:break-all; }}
.meta {{ padding:2px 8px 8px; color:var(--muted); font-size:12px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<ul>{tot}</ul>
<div class="key"><span><i style="background:#28dc3c"></i>found (IoU ≥ 0.5)</span><span><i style="background:#ffdc00"></i>boxes
disagree (IoU 0.3–0.5)</span><span><i style="background:#eb2828"></i>missed</span><span><i style="background:#e628e6"></i>extra</span>
<span><i style="background:#00dcff"></i>box (no annotation)</span></div>
<p style="color:var(--muted)">{len(cards)} pictures; click one for full size.</p>
<div class="grid">{''.join(rows)}</div></body></html>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)


# ------------------------------------------------------------------ main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="*", default=[], help="folders / pictures / videos")
    p.add_argument("--cbvd", default=None, help="a CBVD-5 root: its keyframes with the annotation")
    p.add_argument("--cbvd-split", choices=("val", "train"), default="val",
                   help="val: 292 keyframes no detector trained on; train: ~2.9k the detectors learnt from")
    p.add_argument("--detector", action="append", required=True, help="a best/ folder (several: one panel each)")
    p.add_argument("--threshold", type=float, default=None, help="score cut-off (default: each detector's own)")
    p.add_argument("--calib", type=int, default=0, help="1: apply the detector's box_calib.json (CBVD-5's habit)")
    p.add_argument("--max", type=int, default=300, help="at most this many pictures")
    p.add_argument("--pick", choices=("first", "random", "worst"), default="random",
                   help="which pictures when there are more than --max; worst: sorted by missed + extra (annotated only)")
    p.add_argument("--every-sec", type=float, default=2.0, help="videos: a frame every N seconds")
    p.add_argument("--width", type=int, default=1280, help="panel width in the pictures written")
    p.add_argument("--out", default="/workspace/herd/show_dets")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.src and not args.cbvd:
        sys.exit("give --src <folder / files> or --cbvd <CBVD-5 root>")

    # every picture as (name, loader, annotation or None)
    items = []
    pics, vids = find_sources(args.src)
    base = os.path.commonpath([os.path.dirname(os.path.abspath(f)) for f, _ in pics]) if pics else ""
    items += [(os.path.relpath(os.path.abspath(f), base), (lambda f=f: Image.open(f).convert("RGB")), ann)
              for f, ann in pics]
    for v in vids:
        items += [(n, (lambda im=im: im), None) for n, im in video_frames(v, args.every_sec, args.max)]
    if args.cbvd:
        items += cbvd_items(args.cbvd, args.cbvd_split)
    if not items:
        sys.exit("no pictures or videos found")
    annotated = sum(a is not None for _, _, a in items)
    pick = args.pick if (args.pick != "worst" or annotated) else "random"
    if pick == "random" and len(items) > args.max:
        items = sorted(random.Random(args.seed).sample(items, args.max), key=lambda x: x[0])
    elif pick == "first":
        items = items[:args.max]
    print(f"[show-dets] {len(items)} pictures ({annotated} with an annotation), {len(args.detector)} detector(s)",
          flush=True)

    import detector as det_mod
    dets = []
    for b in args.detector:
        live = det_mod.Live(b, args.threshold, calib=bool(args.calib))
        tag = os.path.basename(os.path.dirname(os.path.abspath(b)))
        dets.append((tag, live))

    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
    cards, totals = [], {t: {} for t, _ in dets}
    for n, (name, load, gts) in enumerate(items, 1):
        img = load()
        panels, stats = [], {}
        for tag, live in dets:
            found = [c["bbox"] + [c["det_score"]] for c in live(img)]
            panel, s = paint(img, found, gts, f"{tag} (thr {live.threshold})", args.width)
            panels.append(panel)
            stats[tag] = s
            for k, v in s.items():
                totals[tag][k] = totals[tag].get(k, 0) + v
        fname = f"{n:05d}.jpg"
        (stack(panels) if len(panels) > 1 else panels[0]).save(os.path.join(args.out, "img", fname), quality=88)
        cards.append({"name": name, "file": fname, "stats": stats})
        print(f"\r  {n}/{len(items)}", end="", flush=True)
    print(flush=True)

    for t in totals.values():
        if t.get("annotated"):
            t["recall"] = t["found"] / t["annotated"]
            t["precision"] = t["found"] / max(1, t["boxes"])
            t["recall_iou0.3"] = (t["found"] + t["loose"]) / t["annotated"]
    if pick == "worst":
        bad = lambda c: max(s.get("missed", 0) + s.get("extra", 0) for s in c["stats"].values())
        cards.sort(key=lambda c: -bad(c))
        for c in cards[args.max:]:                 # all were drawn to find the worst; only these are kept
            os.remove(os.path.join(args.out, "img", c["file"]))
        cards = cards[:args.max]
    title = "Detector boxes - " + ", ".join(args.src + ([f"CBVD-5 {args.cbvd_split} ({args.cbvd})"] if args.cbvd else []))
    write_html(args.out, cards, totals, title)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"sources": args.src, "cbvd": args.cbvd, "detectors": args.detector, "threshold": args.threshold,
                   "calib": args.calib, "totals": totals, "pictures": cards}, fh, indent=1)
    for tag, t in totals.items():
        if t.get("annotated"):
            print(f"  {tag}: {t['annotated']} annotated, {t['boxes']} boxes - recall {t['recall']:.1%}, precision "
                  f"{t['precision']:.1%} (IoU 0.5), recall {t['recall_iou0.3']:.1%} at IoU 0.3")
        else:
            print(f"  {tag}: {t.get('boxes', 0)} boxes")
    print(f"-> {os.path.join(args.out, 'index.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
