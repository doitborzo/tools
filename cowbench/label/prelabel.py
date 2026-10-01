#!/usr/bin/env python3
"""Pre-label a new, unannotated video for CVAT, so a person corrects instead
of drawing from nothing.

    video -> one keyframe a second -> cows found by a detector -> linked into
    tracks (one id per cow) -> optionally posture / activity asked of Muse ->
    a CVAT import file

    python label/prelabel.py farm.mp4 --out newfarm
    python label/prelabel.py farm.mp4 --out newfarm --detector /workspace/lora-runs/detector/best
    python label/prelabel.py farm.mp4 --out newfarm --base-url http://<pod>:8000 --model muse-glimmer

Writes into --out:
    labelframes/labelframes/<clip>_<second>.jpg   the keyframes, named as in CBVD-5
    keyframes.json         video frame index <-> clip and second; cvat2ava.py needs it
    prelabels.jsonl        every box found, with its track and any labels
    cvat_labels.json       the label set: paste into the CVAT task's "Raw" label editor
    cvat_prelabels.xml     import into that task as "CVAT 1.1"

The CVAT task must be made from the same video file: boxes are placed on its
frame numbers (one keyframe every fps frames; CVAT interpolates between).

Detectors:
  owlv2   (default) google/owlv2-base-patch16-ensemble, Apache-2.0. Finds
          "a cow" without training on this barn - the choice for a camera
          our RT-DETRv2 never saw (it was trained on CBVD-5's side views).
          On 51 barn photos of another dataset, threshold 0.3: 67% of the
          cows found, 65% of its boxes right; expect to add and delete boxes.
  <dir>   a detector.py best/ folder (RT-DETRv2 trained on CBVD-5).
  none    keyframes and an empty CVAT file only: draw every box by hand.

Faster after a first pass: correct the first minute or two in CVAT, export,
train the detector on it, and pre-label only the rest - the corrected part is
kept as it is (an import replaces everything in the CVAT task):

    python label/cvat2ava.py export.zip --root newfarm --split train
    python detector.py train --root newfarm --out newfarm_det --model <CBVD-5 detector/best>
    python label/prelabel.py farm.mp4 --out newfarm --detector newfarm_det/best \
        --keep export.zip --keep-until 120

Labels a model did not give are "?" in CVAT, and cvat2ava.py refuses a "?",
so nothing unreviewed slips into the dataset as a label.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

from PIL import Image  # noqa: E402

UNSET = "?"
POSTURE_VALUES = [UNSET, "standing", "lying"]
ACTIVITY_VALUES = [UNSET, "none", "feeding", "drinking", "ruminating"]
CVAT_LABELS = [{
    "name": "cow", "color": "#32cd32", "type": "rectangle",
    "attributes": [
        {"name": "posture", "mutable": True, "input_type": "radio",
         "default_value": UNSET, "values": POSTURE_VALUES},
        {"name": "activity", "mutable": True, "input_type": "radio",
         "default_value": UNSET, "values": ACTIVITY_VALUES},
    ],
}]
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OWL_QUERIES = ["a photo of a cow"]   # more queries changed nothing on a barn test set


# ------------------------------------------------------------------ keyframes

def keyframes(video, every):
    """Yield (frame_index, second, PIL image, fps, frames in video) once every
    `every` seconds.

    Frames are read one after another, as CVAT decodes them, so frame N here
    is frame N in the CVAT task."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    i, s = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i == round(s * every * fps):
            yield i, round(s * every, 3), Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), fps, n
            s += 1
        i += 1
    cap.release()


# ------------------------------------------------------------------ detection

def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _nms(boxes, iou, contain):
    """Best first; drop a box overlapping a kept one by IoU >= iou, or lying
    mostly (>= contain of the smaller) inside or around it - OWLv2 also boxes
    a cow's front half, or two cows together."""
    import detect as detect_mod
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        ok = True
        for k in keep:
            inter = _area([max(b[0], k[0]), max(b[1], k[1]), min(b[2], k[2]), min(b[3], k[3])])
            if detect_mod.iou(b[:4], k[:4]) >= iou or inter >= contain * min(_area(b), _area(k)):
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


class Owl:
    def __init__(self, threshold, queries):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(self.dev).eval()
        self.threshold = threshold
        self.queries = queries
        self.name = OWL_MODEL

    def __call__(self, img):
        w, h = img.size
        inputs = self.processor(text=[self.queries], images=img, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model(**inputs)
        # OWLv2 pads the image to a square at the bottom / right: boxes come
        # back relative to that square.
        side = max(w, h)
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=self.threshold, target_sizes=[(side, side)])[0]
        boxes = []
        for (x1, y1, x2, y2), score in zip(res["boxes"].tolist(), res["scores"].tolist()):
            x1, x2 = max(0.0, x1) / w, min(w, x2) / w
            y1, y2 = max(0.0, y1) / h, min(h, y2) / h
            # A box over most of the frame is "the herd", not a cow.
            if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) < 0.4:
                boxes.append([x1, y1, x2, y2, score])
        return _nms(boxes, 0.5, 0.8)


class RtDetr:
    def __init__(self, best_dir, threshold):
        import detector as detector_mod
        self.live = detector_mod.Live(best_dir, threshold)
        self.name = f"{self.live.name} ({best_dir})"

    def __call__(self, img):
        return [c["bbox"] + [c["det_score"]] for c in self.live(img)]


# ------------------------------------------------------------------- tracking

def link(frames, min_iou, max_gap):
    """Greedy IoU linking, keyframe to keyframe: a cow barely moves in a
    second, least of all seen from above. A track waits up to max_gap
    keyframes for its cow to reappear (a missed detection, an occlusion).
    Sets "track" on every box; returns the number of tracks."""
    import detect as detect_mod
    open_tracks = []   # [track_id, last box, keyframes since seen]
    next_id = 1
    for f in frames:
        pairs = sorted(((detect_mod.iou(t[1]["bbox"], b["bbox"]), ti, bi)
                        for ti, t in enumerate(open_tracks) for bi, b in enumerate(f["boxes"])),
                       reverse=True)
        used_t, used_b = set(), set()
        for v, ti, bi in pairs:
            if v < min_iou or ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            f["boxes"][bi]["track"] = open_tracks[ti][0]
            open_tracks[ti][1] = f["boxes"][bi]
            open_tracks[ti][2] = 0
        for ti, t in enumerate(open_tracks):
            if ti not in used_t:
                t[2] += 1
        for bi, b in enumerate(f["boxes"]):
            if bi not in used_b:
                b["track"] = next_id
                open_tracks.append([next_id, b, 0])
                next_id += 1
        open_tracks = [t for t in open_tracks if t[2] <= max_gap]
    return next_id - 1


# ----------------------------------------------------------------- behaviour

def ask_muse(frames, args, path_of):
    """posture / activity for every box, one frame question per keyframe."""
    import client as client_mod
    import frame as frame_mod
    import render as render_mod
    cli = client_mod.MuseClient(base_url=args.base_url, model=args.model, api_key=args.api_key,
                                max_tokens=8192, timeout=600, answer_now=args.answer_now)

    def one(f):
        cows = frame_mod.order([b for b in f["boxes"]])
        if not cows:
            return 0
        img = frame_mod.render(path_of(f), cows, args.max_width)
        url = render_mod.to_data_url(img, 90)
        text = frame_mod.prompt(cows)
        if args.answer_now:
            out = cli.complete([url], text, prefill=frame_mod.ANSWER_PREFILL,
                               max_tokens=24 * len(cows) + 64)
        else:
            out = cli.complete([url], text, schema=frame_mod.SCHEMA, name="cows_in_frame")
        got = frame_mod.parse(out.get("raw") or "", len(cows))
        for i, c in enumerate(cows, 1):
            if i in got:
                c["posture"], c["activity"] = got[i]
        return len(got)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for k, n in enumerate(pool.map(one, frames), 1):
            done += n
            print(f"\r  Muse: {k}/{len(frames)} keyframes, {done} cows labelled", end="", flush=True)
    print()


# ----------------------------------------------------------------------- CVAT

def kept_tracks(export, boundary):
    """The tracks of a CVAT export (CVAT for video 1.1), cut at frame
    `boundary`: their keyframes before it, the exact box at the last frame
    before it, and outside="1" at it. XML lines for cvat_xml."""
    import xml.etree.ElementTree as ET
    sys.path.insert(0, HERE)
    import cvat2ava
    root = cvat2ava.read_xml(export)
    out = []
    for track in root.iter("track"):
        boxes = [b for b in track.iter("box") if int(b.get("frame")) < boundary]
        if not boxes:
            continue
        last = max(boxes, key=lambda b: int(b.get("frame")))
        keep = [b for b in boxes if b.get("keyframe") == "1" or b is last]
        lines = []
        for b in sorted(keep, key=lambda b: int(b.get("frame"))):
            b.set("keyframe", "1")
            lines.append("    " + ET.tostring(b, encoding="unicode").strip())
        if last.get("outside") != "1":
            end = ET.fromstring(ET.tostring(last))
            end.set("frame", str(boundary))
            end.set("outside", "1")
            lines.append("    " + ET.tostring(end, encoding="unicode").strip())
        out.append((track.get("label", "cow"), lines))
    return out


def cvat_xml(frames, size, kept=(), still_iou=0.8):
    """CVAT for video 1.1: one <track> per cow, outside="1" at the first
    keyframe it was not seen on, CVAT interpolating between.

    A CVAT keyframe holds its own posture and activity, so a keyframe every
    second would make a person set them every second. A box becomes a CVAT
    keyframe only where the cow moved (IoU < still_iou with the last one
    kept), its labels changed, or it is first or last of a run: a cow lying
    still for a minute is two keyframes."""
    import detect as detect_mod
    w, h = size

    def box(fr, b, outside):
        x1, y1, x2, y2 = b["bbox"]
        return [f'    <box frame="{fr}" keyframe="1" outside="{outside}" occluded="0" '
                f'xtl="{x1 * w:.2f}" ytl="{y1 * h:.2f}" xbr="{x2 * w:.2f}" ybr="{y2 * h:.2f}" z_order="0">',
                f'      <attribute name="posture">{b.get("posture") or UNSET}</attribute>',
                f'      <attribute name="activity">{b.get("activity") or UNSET}</attribute>',
                "    </box>"]

    by_track = {}
    for f in frames:
        for b in f["boxes"]:
            by_track.setdefault(b["track"], []).append((f["frame"], b))
    order = [f["frame"] for f in frames]
    nxt = dict(zip(order, order[1:]))
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    for k, (label, boxes) in enumerate(kept):
        lines.append(f'  <track id="{k}" label="{label}" source="manual">')
        lines += boxes
        lines.append("  </track>")
    labels = lambda b: (b.get("posture"), b.get("activity"))
    for k, (_tid, seq) in enumerate(sorted(by_track.items()), len(kept)):
        lines.append(f'  <track id="{k}" label="cow" source="auto">')
        last = None
        for i, (fr, b) in enumerate(seq):
            gone = nxt.get(fr)
            run_ends = i + 1 == len(seq) or seq[i + 1][0] != gone
            if (last is None or run_ends or labels(b) != labels(last)
                    or detect_mod.iou(b["bbox"], last["bbox"]) < still_iou):
                lines += box(fr, b, 0)
                last = b
            if run_ends:
                if gone is not None:
                    lines += box(gone, b, 1)
                last = None
        lines.append("  </track>")
    lines.append("</annotations>")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--video-id", type=int, default=1001,
                   help="first clip id; CBVD-5 uses 1-394, so new ones start above")
    p.add_argument("--clip-seconds", type=int, default=10,
                   help="split into clips this long (ids video-id, video-id+1, ...); 0 = one clip")
    p.add_argument("--every", type=float, default=1.0, help="seconds between keyframes (CBVD-5: 1)")
    p.add_argument("--detector", default="owlv2", help="owlv2 | <detector.py best/ dir> | none")
    p.add_argument("--det-threshold", type=float, default=None,
                   help="score threshold (owlv2: 0.3; RT-DETRv2: its own)")
    p.add_argument("--queries", nargs="+", default=OWL_QUERIES, help="owlv2 text queries")
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument("--track-gap", type=int, default=2, help="keyframes a track waits for its cow")
    p.add_argument("--still-iou", type=float, default=0.8,
                   help="a box overlapping the track's last CVAT keyframe this much adds none")
    p.add_argument("--base-url", default=None, help="vLLM server: ask Muse for posture/activity")
    p.add_argument("--model", default="muse-glimmer")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--answer-now", action="store_true", help="for a LoRA: prefilled answer, no reasoning")
    p.add_argument("--max-width", type=int, default=1920)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--keep", default=None, metavar="EXPORT",
                   help="a CVAT export (CVAT for video 1.1) whose tracks are kept before --keep-until")
    p.add_argument("--keep-until", type=float, default=0, metavar="SECONDS",
                   help="with --keep: keep the export before this second, pre-label from it on")
    args = p.parse_args(argv)
    if args.keep and not args.keep_until:
        sys.exit("--keep needs --keep-until: the second up to which the export is kept")

    frame_dir = os.path.join(args.out, "labelframes", "labelframes")
    os.makedirs(frame_dir, exist_ok=True)
    if args.detector == "owlv2":
        det = Owl(0.3 if args.det_threshold is None else args.det_threshold, args.queries)
    elif args.detector == "none":
        det = None
    else:
        det = RtDetr(args.detector, args.det_threshold)

    def clip_of(sec):
        return args.video_id + (int(sec) // args.clip_seconds if args.clip_seconds else 0)

    def path_of(f):
        return os.path.join(frame_dir, f"{f['video_id']}_{f['timestamp']:05d}.jpg")

    frames, size, fps, n_frames = [], None, None, 0
    for fr, sec, img, fps, n_frames in keyframes(args.video, args.every):
        size = img.size
        f = {"frame": fr, "second": sec, "video_id": str(clip_of(sec)), "timestamp": int(round(sec))}
        img.save(path_of(f), quality=95)
        fresh = det is not None and not (args.keep and sec < args.keep_until)
        f["boxes"] = ([{"bbox": [round(v, 4) for v in b[:4]], "det_score": round(b[4], 3)}
                       for b in det(img)] if fresh else [])
        frames.append(f)
        print(f"\r  {len(frames)} keyframes, {sum(len(x['boxes']) for x in frames)} cows found",
              end="", flush=True)
    print()
    if not frames:
        sys.exit("no frames read from the video")
    n_tracks = link(frames, args.track_iou, args.track_gap)
    if args.base_url and det:
        ask_muse(frames, args, path_of)

    step = frames[1]["frame"] - frames[0]["frame"] if len(frames) > 1 else 1
    with open(os.path.join(args.out, "keyframes.json"), "w", encoding="utf-8") as fh:
        json.dump({"video": os.path.abspath(args.video), "fps": fps, "frames_in_video": n_frames,
                   "width": size[0], "height": size[1], "step": step,
                   "detector": det.name if det else None,
                   "behaviour_from": (f"{args.model} at {args.base_url}" if args.base_url and det else None),
                   "keyframes": [{k: f[k] for k in ("frame", "second", "video_id", "timestamp")}
                                 for f in frames]}, fh, indent=1)
    with open(os.path.join(args.out, "prelabels.jsonl"), "w", encoding="utf-8") as fh:
        for f in frames:
            fh.write(json.dumps(f) + "\n")
    with open(os.path.join(args.out, "cvat_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(CVAT_LABELS, fh, indent=2)
    with open(os.path.join(args.out, "cvat_prelabels.xml"), "w", encoding="utf-8") as fh:
        kept = []
        if args.keep:
            boundary = next((f["frame"] for f in frames if f["second"] >= args.keep_until),
                            frames[-1]["frame"] + 1)
            kept = kept_tracks(args.keep, boundary)
            print(f"  kept {len(kept)} tracks of {args.keep} before frame {boundary} "
                  f"({args.keep_until:g} s)")
        fh.write(cvat_xml(frames, size, kept, args.still_iou))
    n_boxes = sum(len(f["boxes"]) for f in frames)
    print(f"{len(frames)} keyframes ({size[0]}x{size[1]}, every {step} frames of {fps:.2f} fps), "
          f"{n_boxes} boxes, {n_tracks} tracks -> {args.out}")
    if det:
        print(f"  {n_boxes / len(frames):.1f} cows a keyframe on average; look at a few keyframes before "
              f"trusting it (--det-threshold to change)")


if __name__ == "__main__":
    main()
