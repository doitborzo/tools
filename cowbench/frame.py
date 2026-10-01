"""Every cow of a keyframe in one question.

The per-cow question sends the whole frame once per cow: ~15 times the same
image for a busy keyframe, which on one A100 held 12 cameras to one update
every ~77 s. Here the frame goes once. A detector (RT-DETRv2 on a farm, the
annotation here) supplies the boxes; they are drawn on the frame, lime as in
the per-cow question and numbered, and listed in the text with their
coordinates on the 0-1000 scale the model grounds in. The answer is one list:

    {"cows": [{"id": 1, "posture": "standing", "activity": "feeding"}, ...]}

Shared by lora/train_lora.py (training and eval), cowbench.py run --unit
frame and stress --task frame, so all three ask exactly the same question.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re

from PIL import ImageDraw, ImageFont

import render as render_mod
from cbvd import ACTIVITIES, POSTURES

_PROMPT = """You are looking at a single still frame from a fixed surveillance camera in a dairy barn.

{n} cows are outlined with bright green rectangles, each labelled with its number. Their positions, as [x1, y1, x2, y2] on a 0-1000 scale (x from the left edge, y from the top edge):
{boxes}

For each numbered cow, report two things about that cow only.

posture - exactly one of:
  standing  the cow carries its weight on its legs, body upright
  lying     the cow's body rests on the stall bed or the floor

activity - exactly one of:
  feeding     the head is down at the feed barrier or in the feed alley, eating or reaching for feed
  drinking    the head is over or inside a water trough
  ruminating  chewing cud while not at feed and not at water; the jaw works sideways, head up or resting
  none        none of the three above can be seen

Both fields are independent: a lying cow can be ruminating, a standing cow can be feeding.

Answer with JSON only: {{"cows": [{{"id": 1, "posture": ..., "activity": ...}}, ...]}}, one entry per numbered cow, in number order."""

PROMPT_SHA = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()[:12]

# Prefilled answer start, as client.ANSWER_PREFILL for the per-cow question.
ANSWER_PREFILL = '{"cows": [{"id": 1, "posture": "'

SCHEMA = {
    "type": "object",
    "properties": {"cows": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"},
                       "posture": {"type": "string", "enum": list(POSTURES)},
                       "activity": {"type": "string", "enum": list(ACTIVITIES)}},
        "required": ["id", "posture", "activity"], "additionalProperties": False}}},
    "required": ["cows"], "additionalProperties": False,
}


def order(cows):
    """Reading order - top to bottom, then left to right, by box centre - so
    the numbering does not depend on how the boxes happened to arrive."""
    def key(c):
        x1, y1, x2, y2 = c["bbox"]
        return (round((y1 + y2) / 2 * 20), (x1 + x2) / 2)
    return sorted(cows, key=key)


def group(rows):
    """Per-cow rows -> keyframes [{"video_id", "timestamp", "cows": [...]}],
    the cows of each in reading order."""
    frames = collections.OrderedDict()
    for r in rows:
        frames.setdefault((r["video_id"], r["timestamp"]), []).append(r)
    return [{"video_id": v, "timestamp": t, "cows": order(cs)} for (v, t), cs in frames.items()]


def prompt(cows) -> str:
    boxes = "\n".join(f"  {i}: [{', '.join(str(round(v * 1000)) for v in c['bbox'])}]"
                      for i, c in enumerate(cows, 1))
    return _PROMPT.format(n=len(cows), boxes=boxes)


def _font(size):
    try:
        return ImageFont.load_default(size=size)    # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw(img, cows):
    """The frame with every cow outlined and numbered at its top-left corner,
    white on black so the number reads on any coat. In place; returns img."""
    d = ImageDraw.Draw(img)
    w, h = img.size
    stroke = max(3, round(w / 320))
    font = _font(max(14, round(w / 55)))
    for i, c in enumerate(cows, 1):
        x1, y1, x2, y2 = (c["bbox"][0] * w, c["bbox"][1] * h, c["bbox"][2] * w, c["bbox"][3] * h)
        d.rectangle([x1, y1, x2, y2], outline=render_mod.OUTLINE_RGB, width=stroke)
        tx, ty = x1 + stroke, y1 + stroke
        l, t, r, b = d.textbbox((tx, ty), str(i), font=font)
        d.rectangle([l - 2, t - 2, r + 2, b + 2], fill=(0, 0, 0))
        d.text((tx, ty), str(i), fill=(255, 255, 255), font=font)
    return img


def render(frame, cows, max_width):
    """Plain frame -> numbered boxes -> downscaled like every other mode."""
    img = render_mod.render(frame, None, mode="plain", max_width=10 ** 6)
    return render_mod.render(draw(img, cows), None, mode="plain", max_width=max_width)


def answer_json(cows) -> str:
    return json.dumps({"cows": [{"id": i, "posture": c["gt_posture"], "activity": c["gt_activity"]}
                                for i, c in enumerate(cows, 1)]})


def activity_spans(cows):
    """(start, end, activity) of every activity value inside answer_json(cows)."""
    text, spans, pos = answer_json(cows), [], 0
    for c in cows:
        key = '"activity": "'
        start = text.index(key, pos) + len(key)
        spans.append((start, start + len(c["gt_activity"]), c["gt_activity"]))
        pos = start
    return spans


def parse(text, n):
    """{id: (posture, activity)} from an answer; JSON first, then the entries
    by pattern so one broken entry does not lose the rest of the frame."""
    out = {}
    try:
        for e in json.loads(text).get("cows", []):
            if e.get("posture") in POSTURES and e.get("activity") in ACTIVITIES:
                out[int(e["id"])] = (e["posture"], e["activity"])
    except Exception:
        for i, p, a in re.findall(r'"id"\s*:\s*(\d+)\s*,\s*"posture"\s*:\s*"(\w+)"\s*,'
                                  r'\s*"activity"\s*:\s*"(\w+)"', text):
            if p in POSTURES and a in ACTIVITIES:
                out[int(i)] = (p, a)
    return {k: v for k, v in out.items() if 1 <= k <= n}


def records(cows, text, extra=None):
    """One results.jsonl record per cow, as the per-cow bench writes them, so
    score / report / compare work unchanged. A cow the answer skipped gets
    no posture and no activity - scored wrong, flagged."""
    got = parse(text, len(cows))
    out = []
    for i, c in enumerate(cows, 1):
        p, a = got.get(i, (None, None))
        rec = dict(c, posture=p, activity=a, frame_answer=text, frame_index=i, **(extra or {}))
        if i not in got:
            rec["parse_error"] = f"cow {i} of {len(cows)} missing from the frame answer"
        out.append(rec)
    return out


# ------------------------------------------------------ boxes from a detector
# Tests take their boxes from the detector (detector.py, RT-DETRv2), never from
# the annotation: the model answers about what the detector found, and the
# annotation is used only to score. An annotated cow the detector missed is an
# error; a detection that is no annotated cow is counted on its own.

def detected_cows(det, threshold):
    """A detections.jsonl row -> pseudo-cows to ask about, in reading order."""
    cows = [{"id": "{}_{:05d}_det{}".format(det["video_id"], det["timestamp"], k),
             "video_id": det["video_id"], "timestamp": det["timestamp"],
             "bbox": b[:4], "det_score": b[4], "gt_posture": None, "gt_activity": None}
            for k, b in enumerate(det["boxes"]) if b[4] >= threshold]
    return order(cows)


def to_gt(gt_cows, answered, iou=0.5):
    """Answers about detected boxes -> one record per annotated cow, matched by
    box overlap (greedy, highest IoU first). Returns (records, extras): extras
    are the answered detections that matched no annotated cow."""
    import detect as detect_mod   # here, not at the top: detect imports nothing of ours
    pairs = detect_mod.match([a["bbox"] for a in answered], [g["bbox"] for g in gt_cows], iou)
    by_gt = {j: (i, v) for i, j, v in pairs}
    records = []
    for j, g in enumerate(gt_cows):
        if j in by_gt:
            i, v = by_gt[j]
            a = answered[i]
            rec = dict(g, posture=a.get("posture"), activity=a.get("activity"),
                       det_bbox=a["bbox"], det_score=a.get("det_score"), det_iou=round(v, 3))
            for k in ("parse_error", "error", "frame_answer", "raw"):
                if k in a:
                    rec[k] = a[k]
        else:
            rec = dict(g, posture=None, activity=None, missed_by_detector=True,
                       parse_error="not found by the detector")
        records.append(rec)
    used = {i for i, _, _ in pairs}
    extras = [a for i, a in enumerate(answered) if i not in used]
    return records, extras


def load_detections(path):
    """detections.jsonl -> ({(video_id, timestamp): row}, meta of the detector run)."""
    import os
    dets = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                dets[(r["video_id"], r["timestamp"])] = r
    meta_path = os.path.join(os.path.dirname(os.path.abspath(path)), "det_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return dets, meta
