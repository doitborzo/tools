"""Finding the cows: the whole keyframe, no box drawn, "list every cow".

The bench proper hands the model a box from the annotation and asks about
that one cow. On a farm nobody hands out boxes, so this asks the other
question: given a bare frame, does the model find every cow, and where?

    detect        -> detect_results.jsonl  one answer per keyframe (resumable)
    detect-score  -> detect_metrics.json, detect_report.md

A found cow is matched to an annotated one by box overlap (IoU), greedily,
highest overlap first - the usual detection scoring. Two thresholds: 0.5,
the common bar, and 0.3, because the annotators' boxes and the model's may
simply be drawn differently around the same animal.

What this cannot see: whether every visible cow was annotated. A cow the
model finds and the annotators skipped counts as a false positive, so
precision is a lower bound.
"""

from __future__ import annotations

import collections
import concurrent.futures
import datetime
import json
import os
import threading

import cbvd
import render as render_mod
from cbvd import ACTIVITIES, POSTURES

PROMPT = """You are looking at a single still frame from a fixed surveillance camera in a dairy barn. The image is {w} x {h} pixels.

Find every cow in the frame: standing or lying, near or far from the camera, and cows partly hidden behind other cows or the barn structure. Report each cow once.

For each cow give:

box - [x1, y1, x2, y2], the tightest rectangle around the cow's visible body, on a 0-1000 scale: x from 0 at the left edge to 1000 at the right edge, y from 0 at the top edge to 1000 at the bottom edge

posture - exactly one of:
  standing  the cow carries its weight on its legs, body upright
  lying     the cow's body rests on the stall bed or the floor

activity - exactly one of:
  feeding     the head is down at the feed barrier or in the feed alley, eating or reaching for feed
  drinking    the head is over or inside a water trough
  ruminating  chewing cud while not at feed and not at water; the jaw works sideways, head up or resting
  none        none of the three above can be seen

Answer with JSON only: {{"cows": [...]}}."""

SCHEMA = {
    "type": "object",
    "properties": {
        "cows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box": {"type": "array", "items": {"type": "number"},
                            "minItems": 4, "maxItems": 4},
                    "posture": {"type": "string", "enum": list(POSTURES)},
                    "activity": {"type": "string", "enum": list(ACTIVITIES)},
                },
                "required": ["box", "posture", "activity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["cows"],
    "additionalProperties": False,
}

# The prompt asks for 0-1000: asked for pixels on the full-val run, the model
# answered 0-1000 anyway (recall at IoU 0.5: 53.2% read as 0-1000, 4.3% as
# pixels) - its own grounding convention.
BOX_FORMATS = ("norm1000", "pixel", "norm1")


def frames_of(manifest):
    """Keyframes with every annotated cow on them, in a stable order."""
    frames = collections.OrderedDict()
    for r in sorted(manifest, key=lambda r: (int(r["video_id"]), r["timestamp"])):
        frames.setdefault((r["video_id"], r["timestamp"]), []).append(
            {"bbox": r["bbox"], "posture": r["gt_posture"], "activity": r["gt_activity"]})
    return frames


# ---------------------------------------------------------------------- run

def run(args, client_mod, jsonl_read):
    manifest = jsonl_read(os.path.join(args.out, "manifest.jsonl"))
    if not manifest:
        raise SystemExit(f"no manifest in {args.out} - run `plan` first")
    with open(os.path.join(args.out, "plan_meta.json"), encoding="utf-8") as fh:
        plan_meta = json.load(fh)
    root = args.root or plan_meta["root"]
    frames = frames_of(manifest)

    path = os.path.join(args.out, "detect_results.jsonl")
    if args.fresh and os.path.exists(path):
        os.remove(path)
    done = {(r["video_id"], r["timestamp"]) for r in jsonl_read(path)}
    todo = [k for k in frames if k not in done]
    print(f"{len(frames)} keyframes ({sum(len(v) for v in frames.values())} annotated cows), "
          f"{len(done)} done, {len(todo)} to go")
    if not todo:
        return

    cli = client_mod.MuseClient(base_url=args.base_url, model=args.model, api_key=args.api_key,
                                temperature=args.temperature, max_tokens=args.max_tokens,
                                timeout=args.timeout, retries=args.retries)
    info = cli.server_info()
    meta = {
        "run_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "detect: every cow on a bare keyframe, box + posture + activity",
        "model": args.model, "vllm_version": info.get("vllm_version"), "server": info,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "max_width": args.max_width, "jpeg_quality": args.jpeg_quality,
        "prompt": PROMPT, "annotations": plan_meta["annotations"],
        "videos": ", ".join(plan_meta["videos"]),
    }
    with open(os.path.join(args.out, "detect_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    lock = threading.Lock()
    counter = {"n": 0}
    out_fh = open(path, "a", encoding="utf-8")

    def work(key):
        vid, ts = key
        rec = {"video_id": vid, "timestamp": ts, "gt": frames[key]}
        try:
            box = cbvd.Box(vid, ts, 0, 0, 0, 0, "1", ())
            img = render_mod.render(cbvd.frame_path(root, box), None, mode="plain",
                                    max_width=args.max_width)
            rec["width"], rec["height"] = img.size
            url = render_mod.to_data_url(img, quality=args.jpeg_quality)
            rec.update(cli.ask([url], PROMPT.format(w=img.width, h=img.height),
                               SCHEMA, "cows_in_frame"))
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        with lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            counter["n"] += 1
            found = len((rec.get("parsed") or {}).get("cows") or [])
            print(f"\r  {counter['n']}/{len(todo)}  (last: {found} found, "
                  f"{len(rec['gt'])} annotated)   ", end="", flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))
    finally:
        out_fh.close()
        print()
    print(f"-> {path}")


# -------------------------------------------------------------------- score

def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def normalise(box, fmt, w, h):
    """The model's box as fractions of the frame, corners in order, clipped."""
    x1, y1, x2, y2 = (float(v) for v in box)
    sx, sy = {"pixel": (w, h), "norm1000": (1000.0, 1000.0), "norm1": (1.0, 1.0)}[fmt]
    x1, x2 = sorted((x1 / sx, x2 / sx))
    y1, y2 = sorted((y1 / sy, y2 / sy))
    clip = lambda v: min(max(v, 0.0), 1.0)
    return [clip(x1), clip(y1), clip(x2), clip(y2)]


def match(preds, gts, threshold):
    """Greedy one-to-one matching, highest IoU first: [(pred_i, gt_i, iou)]."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
                   reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < threshold:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j, v))
    return out


def _predicted(rec, fmt):
    cows = (rec.get("parsed") or {}).get("cows") or []
    out = []
    for c in cows:
        try:
            out.append(dict(c, nbox=normalise(c["box"], fmt, rec["width"], rec["height"])))
        except Exception:
            continue    # a malformed box is a cow not found, not a crash
    return out


def score(results, fmt="norm1000"):
    ok = [r for r in results if "width" in r and "error" not in r]
    m = {"n_keyframes": len(results), "n_failed": len(results) - len(ok),
         "n_annotated": sum(len(r["gt"]) for r in ok), "box_format": fmt}
    for thr in (0.5, 0.3):
        tp = fp = fn = 0
        for r in ok:
            preds = _predicted(r, fmt)
            k = len(match([p["nbox"] for p in preds], [g["bbox"] for g in r["gt"]], thr))
            tp += k
            fp += len(preds) - k
            fn += len(r["gt"]) - k
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        m[f"iou{thr}"] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec_,
                          "f1": 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0}

    diffs = [len(_predicted(r, fmt)) - len(r["gt"]) for r in ok]
    m["count"] = {
        "annotated_mean": sum(len(r["gt"]) for r in ok) / max(len(ok), 1),
        "found_mean": sum(len(_predicted(r, fmt)) for r in ok) / max(len(ok), 1),
        "mean_abs_error": sum(abs(d) for d in diffs) / max(len(diffs), 1),
        "bias": sum(diffs) / max(len(diffs), 1),
        "exact_share": sum(d == 0 for d in diffs) / max(len(diffs), 1),
    }

    # What the model says about the cows it did find, at IoU 0.5, and the
    # whole pipeline end to end: found AND both answers right, over every
    # annotated cow.
    found = post_ok = act_ok = both_ok = 0
    by_size = collections.defaultdict(lambda: [0, 0])
    areas = sorted((g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1])
                   for r in ok for g in r["gt"])
    cuts = [areas[len(areas) * q // 4] for q in (1, 2, 3)] if areas else [0, 0, 0]
    for r in ok:
        preds = _predicted(r, fmt)
        pairs = {j: i for i, j, _ in match([p["nbox"] for p in preds],
                                           [g["bbox"] for g in r["gt"]], 0.5)}
        for j, g in enumerate(r["gt"]):
            a = (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1])
            q = sum(a >= c for c in cuts)
            by_size[q][0] += 1
            if j not in pairs:
                continue
            by_size[q][1] += 1
            p = preds[pairs[j]]
            found += 1
            post_ok += p["posture"] == g["posture"]
            act_ok += p["activity"] == g["activity"]
            both_ok += p["posture"] == g["posture"] and p["activity"] == g["activity"]
    m["matched_labels"] = {
        "matched": found,
        "posture_error": 1 - post_ok / found if found else None,
        "activity_error": 1 - act_ok / found if found else None,
        "exact_error": 1 - both_ok / found if found else None,
    }
    m["end_to_end_error"] = 1 - both_ok / m["n_annotated"] if m["n_annotated"] else None
    names = ["smallest", "smaller", "larger", "largest"]
    m["recall_by_size"] = {names[q]: {"annotated": n, "recall": k / n if n else None}
                           for q, (n, k) in sorted(by_size.items())}
    # If the model reads coordinates another way, its recall collapses under
    # the assumed format and not under the right one; show all three.
    m["recall_iou0.5_by_format"] = {}
    for f in BOX_FORMATS:
        tp = sum(len(match([p["nbox"] for p in _predicted(r, f)], [g["bbox"] for g in r["gt"]], 0.5))
                 for r in ok)
        m["recall_iou0.5_by_format"][f] = tp / m["n_annotated"] if m["n_annotated"] else 0.0
    return m


def render_report(meta, m) -> str:
    pct = lambda x: "—" if x is None else f"{x:.1%}"
    lines = [
        "# Finding the cows: bare keyframe, no box given", "",
        f"Model `{meta.get('model')}` on vLLM {meta.get('vllm_version') or '?'}, "
        f"{meta.get('run_date', '?')}. Frames at max width {meta.get('max_width')} px, "
        f"temperature {meta.get('temperature')}, structured output (json_schema). "
        f"Boxes read as `{m['box_format']}` coordinates.", "",
        f"{m['n_keyframes']} keyframes, {m['n_annotated']} annotated cows"
        + (f", {m['n_failed']} failed requests" if m["n_failed"] else "") + ".", "",
        "## Found vs annotated", "",
        "| IoU needed | Found (TP) | Extra (FP) | Missed (FN) | Recall | Precision | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for thr in (0.5, 0.3):
        d = m[f"iou{thr}"]
        lines.append(f"| {thr} | {d['tp']} | {d['fp']} | {d['fn']} | {pct(d['recall'])} | "
                     f"{pct(d['precision'])} | {pct(d['f1'])} |")
    c = m["count"]
    lines += [
        "", "Precision is a lower bound: a cow the annotators skipped counts as an extra.", "",
        "## Counting", "",
        f"Per keyframe: {c['annotated_mean']:.1f} cows annotated, {c['found_mean']:.1f} found; "
        f"off by {c['mean_abs_error']:.2f} on average (bias {c['bias']:+.2f}), "
        f"exactly right on {pct(c['exact_share'])} of keyframes.", "",
        "## Recall by cow size (IoU 0.5)", "",
        "| Size quarter | Annotated | Recall |", "|---|---|---|",
    ]
    for k, v in m["recall_by_size"].items():
        lines.append(f"| {k} | {v['annotated']} | {pct(v['recall'])} |")
    ml = m["matched_labels"]
    lines += [
        "", "## Behaviour of the cows it found (IoU 0.5)", "",
        f"{ml['matched']} cows matched. Error on them: posture {pct(ml['posture_error'])}, "
        f"activity {pct(ml['activity_error'])}, both {pct(ml['exact_error'])}.", "",
        f"End to end - found **and** both answers right, over every annotated cow: "
        f"error **{pct(m['end_to_end_error'])}**.", "",
        "## Coordinate convention check", "",
        "Recall at IoU 0.5 if the boxes are read as 0-1000, as pixels, or as 0-1. "
        "The prompt asks for 0-1000; a much higher number under another convention "
        "means the model uses that one, and `detect-score --box-format` should follow it.", "",
        "| Read as | Recall |", "|---|---|",
    ]
    for f, v in m["recall_iou0.5_by_format"].items():
        lines.append(f"| {f} | {pct(v)} |")
    return "\n".join(lines) + "\n"
