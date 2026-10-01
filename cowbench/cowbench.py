#!/usr/bin/env python3
"""cowbench - measure how well a VLM reads cow behaviour on CBVD-5.

Four steps, four files. The split exists because only `run` costs anything:
the sample must be fixed before results are seen, and scoring must be
repeatable without asking the model 120 more questions.

    plan   -> manifest.jsonl   which boxes are being tested
    run    -> results.jsonl    one raw model answer per box (resumable)
    score  -> metrics.json     error rates, per-class, confusion
    report -> report.md        the document to hand over

Two more tests on the same plan, for what a farm needs beyond the bench:

    detect / detect-score      no box given: does the model find every cow?
    stress                     N cameras at once: does the server keep up?

Typical session, model served on a pod and reached through an SSH tunnel:

    python cowbench.py plan   --video 371
    python cowbench.py run
    python cowbench.py score
    python cowbench.py report
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import json
import os
import random
import sys
import threading

import cbvd
import client as client_mod
import compare as compare_mod
import detect as detect_mod
import frame as frame_mod
import render as render_mod
import report as report_mod
import scoring
import stress as stress_mod
import tracks as tracks_mod

DEFAULT_ROOT = r"C:\Users\Work\Downloads\archive"
DEFAULT_OUT = "out"
# val, not test: ava_test_v2.1.csv is ava_val_v2.1.csv with every row duplicated,
# same 2533 boxes. Using it would not add a single new example.
DEFAULT_ANN = os.path.join("annotations", "ava_val_v2.1.csv")


def _jsonl_write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonl_read(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------- plan

def cmd_plan(args):
    ann_path = os.path.join(args.root, args.annotations)
    boxes = cbvd.load_boxes(ann_path)
    usable, rejected = cbvd.partition(boxes)

    wanted = set(args.video or [])
    if args.clips:
        # Sample whole clips, not boxes. Two reasons, both measured: per-clip
        # exact-match error runs 0-100%, so the clip is the unit the variance
        # lives in; and a random subset of boxes shreds the tracks that
        # `score --vote` needs - 4% coverage on a 300-box sample against 93%
        # on a full split.
        pool = sorted({b.video_id for b in usable}, key=int)
        random.Random(args.seed).shuffle(pool)
        wanted = set(pool[:args.clips])
    selected = [b for b in usable if not wanted or b.video_id in wanted]
    excluded = [{"id": b.uid, "reason": why} for b, why in rejected
                if not wanted or b.video_id in wanted]

    if not selected:
        sys.exit("no boxes for video(s) {} in {}".format(sorted(wanted), ann_path))

    # Deterministic order, then an optional head. Sampling is seeded and the
    # seed is written into the manifest so the same sample can be rebuilt.
    selected.sort(key=lambda b: (b.video_id, b.timestamp, b.entity_id, b.xyxy))
    if args.limit and len(selected) > args.limit:
        random.Random(args.seed).shuffle(selected)
        selected = selected[:args.limit]
        selected.sort(key=lambda b: (b.video_id, b.timestamp, b.entity_id, b.xyxy))

    # Fail now, not forty requests into the run.
    missing = set()
    for b in selected:
        try:
            cbvd.frame_path(args.root, b)
        except FileNotFoundError as exc:
            missing.add(str(exc))
    if missing:
        sys.exit("missing keyframes:\n  " + "\n  ".join(sorted(missing)[:10]))

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i, b in enumerate(selected):
        rows.append({
            "id": "{}_{:04d}".format(b.uid, i),
            "video_id": b.video_id,
            "timestamp": b.timestamp,
            "bbox": list(b.xyxy),
            "entity_id": b.entity_id,
            "labels": list(b.labels),
            "gt_posture": b.posture,
            "gt_activity": b.activity,
        })
    manifest = os.path.join(args.out, "manifest.jsonl")
    _jsonl_write(manifest, rows)

    meta = {
        "root": args.root,
        "annotations": args.annotations,
        "videos": sorted({r["video_id"] for r in rows}),
        "seed": args.seed,
        "limit": args.limit,
        "clips": args.clips,
        "excluded": excluded,
    }
    with open(os.path.join(args.out, "plan_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    postures = collections.Counter(r["gt_posture"] for r in rows)
    activities = collections.Counter(r["gt_activity"] for r in rows)
    keyframes = len({(r["video_id"], r["timestamp"]) for r in rows})
    print("{} examples from {} clip(s) -> {}".format(
        len(rows), len(meta["videos"]), manifest))
    print("  posture  : " + ", ".join(
        "{}={}".format(k, v) for k, v in postures.most_common()))
    print("  activity : " + ", ".join(
        "{}={}".format(k, v) for k, v in activities.most_common()))
    print("  keyframes: {}".format(keyframes))
    if excluded:
        print("  excluded {} box(es) with contradictory annotations".format(len(excluded)))


# ---------------------------------------------------------------------- run

def cmd_run(args):
    manifest = _jsonl_read(os.path.join(args.out, "manifest.jsonl"))
    if not manifest:
        sys.exit("no manifest in {} - run `plan` first".format(args.out))
    with open(os.path.join(args.out, "plan_meta.json"), encoding="utf-8") as fh:
        plan_meta = json.load(fh)
    root = args.root or plan_meta["root"]

    results_path = os.path.join(args.out, "results.jsonl")
    # Resume rather than restart: vLLM on this hardware is not guaranteed to
    # survive a whole run, and re-asking questions already answered costs time
    # for nothing.
    if args.fresh and os.path.exists(results_path):
        os.remove(results_path)
    done = {r["id"] for r in _jsonl_read(results_path)}
    todo = [r for r in manifest if r["id"] not in done]
    print("{} examples, {} already done, {} to go".format(
        len(manifest), len(done), len(todo)))
    if not todo:
        return

    cli = client_mod.MuseClient(
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
        timeout=args.timeout, retries=args.retries, answer_now=args.answer_now)

    info = cli.server_info()
    meta = {
        "run_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "model_repo": info.get("model_root") or args.model_repo or args.model,
        "quantization": args.quantization,
        "vllm_version": info.get("vllm_version"),
        "server": info,
        "temperature": args.temperature,
        "seed": 0,
        "unit": args.unit,
        "max_tokens": ("per frame" if args.unit == "frame" else 64) if args.answer_now
                      else args.max_tokens,
        "reasoning": ("off: the answer is prefilled with "
                      + (frame_mod.ANSWER_PREFILL if args.unit == "frame" else client_mod.ANSWER_PREFILL)
                      if args.answer_now else "on"),
        "render_mode": args.mode,
        "min_width": args.min_width,
        "frames": args.frames,
        "span": args.span if args.frames > 1 else 0.0,
        "max_width": args.max_width,
        "prompt_sha": (frame_mod.PROMPT_SHA if args.unit == "frame"
                       else client_mod.prompt_sha(args.frames, args.span)),
        "prompt": (frame_mod._PROMPT if args.unit == "frame"
                   else client_mod.build_prompt(args.frames, args.span)),
        "annotations": plan_meta["annotations"],
        "videos": ", ".join(plan_meta["videos"]),
        "excluded": plan_meta.get("excluded", []),
    }
    if args.unit == "frame":
        meta["render_mode"] = "numbered (all cows of the frame)"
    dets = None
    if args.boxes:
        dets, det_meta = frame_mod.load_detections(args.boxes)
        if args.det_threshold is None:
            args.det_threshold = det_meta.get("threshold", 0.5)
        meta["boxes"] = {"source": "detector", "detections": os.path.abspath(args.boxes),
                         "threshold": args.det_threshold, "detector": det_meta.get("base_model"),
                         "detector_ms_per_frame": det_meta.get("ms_per_frame"), "match_iou": 0.5}
    else:
        meta["boxes"] = {"source": "annotation"}
    # Resuming adds to the answers already there: they must be about the same boxes.
    old_meta = os.path.join(args.out, "run_meta.json")
    if done and os.path.exists(old_meta):
        with open(old_meta, encoding="utf-8") as fh:
            old = json.load(fh).get("boxes", {"source": "annotation"})
        if (old.get("source"), old.get("detections")) != (meta["boxes"]["source"], meta["boxes"].get("detections")):
            sys.exit(f"{args.out} already holds {len(done)} answers about boxes from the "
                     f"{old.get('source')} ({old.get('detections', 'annotation')}); "
                     f"use another --out for boxes from the {meta['boxes']['source']}")
    with open(os.path.join(args.out, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    if dets is not None:
        return _run_detected(args, cli, root, manifest, done, results_path, dets)
    if args.unit == "frame":
        return _run_frames(args, cli, root, manifest, done, results_path)

    modes = ["marked", "crop"] if args.mode == "both" else [args.mode]
    lock = threading.Lock()
    counter = {"n": 0}
    out_fh = open(results_path, "a", encoding="utf-8")

    def sources(item, box):
        """The frame(s) this example is built from.

        One frame comes from labelframes/, which is what the annotation was
        drawn on. Several come from the mp4 - verified to be the same pixels at
        the keyframe, so the two paths are interchangeable at frames=1.
        """
        if args.frames <= 1:
            return [cbvd.frame_path(root, box)]
        return render_mod.load_frames(cbvd.video_path(root, box.video_id),
                                      box.timestamp, args.frames, args.span)

    def work(item):
        record = dict(item)
        try:
            box = cbvd.Box(item["video_id"], item["timestamp"], *item["bbox"],
                           item["entity_id"], tuple(item["labels"]))
            urls = []
            for frame in sources(item, box):
                for mode in modes:
                    img = render_mod.render(frame, item["bbox"], mode=mode,
                                            max_width=args.max_width,
                                            min_width=args.min_width)
                    urls.append(render_mod.to_data_url(img, quality=args.jpeg_quality))
            record.update(cli.classify(urls, n_frames=args.frames, span=args.span))
        except Exception as exc:
            record["error"] = "{}: {}".format(type(exc).__name__, exc)
        with lock:
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            counter["n"] += 1
            print("\r  {}/{}".format(counter["n"], len(todo)), end="", flush=True)
        return record

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))
    finally:
        out_fh.close()
        print()
    print("-> {}".format(results_path))


# -------------------------------------------------------------------- score

def cmd_score(args):
    results = _jsonl_read(os.path.join(args.out, "results.jsonl"))
    if not results:
        sys.exit("no results in {} - run `run` first".format(args.out))
    if args.vote:
        results, info = tracks_mod.vote(results)
        print("track vote: {} tracks covering {} of {} boxes".format(
            info["tracks"], info["covered"], info["total"]))
    metrics = scoring.score(results)
    metrics["voted"] = bool(args.vote)
    path = os.path.join(args.out, "metrics-voted.json" if args.vote else "metrics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print("examples          : {}".format(metrics["n_examples"]))
    print("exact-match error : {:.1%}".format(metrics["exact_match"]["error_rate"]))
    print("posture error     : {:.1%}   (baseline {:.1%})".format(
        metrics["posture"]["error_rate"], metrics["baseline"]["posture"]["error_rate"]))
    print("activity error    : {:.1%}   (baseline {:.1%})".format(
        metrics["activity"]["error_rate"], metrics["baseline"]["activity"]["error_rate"]))
    missed = sum(bool(r.get("missed_by_detector")) for r in results)
    if missed:
        print("missed by detector: {}   (counted as errors)".format(missed))
    if metrics["n_failed"] - missed:
        print("failed requests   : {}".format(metrics["n_failed"] - missed))
    print("-> {}".format(path))


# ------------------------------------------------------------------- report

def cmd_report(args):
    with open(os.path.join(args.out, "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    meta_path = os.path.join(args.out, "run_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    # The per-cow rows come from results.jsonl, not from metrics.json: the
    # metrics are aggregates and cannot be un-summed back into examples.
    results = _jsonl_read(os.path.join(args.out, "results.jsonl"))
    text = report_mod.render(meta, metrics, results if not args.no_rows else None)
    path = args.output or os.path.join(args.out, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("-> {}".format(path))


# ------------------------------------------------------------------ compare

def _load_meta(run_dir):
    path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_compare(args):
    a = _jsonl_read(os.path.join(args.a, "results.jsonl"))
    b = _jsonl_read(os.path.join(args.b, "results.jsonl"))
    for path, rows in ((args.a, a), (args.b, b)):
        if not rows:
            sys.exit("no results.jsonl in {}".format(path))
    label_a = args.label_a or os.path.basename(os.path.normpath(args.a))
    label_b = args.label_b or os.path.basename(os.path.normpath(args.b))
    result = compare_mod.compare(a, b, label_a, label_b)
    text = compare_mod.render(result, _load_meta(args.a), _load_meta(args.b))
    path = args.output or os.path.join(args.b, "compare.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print("paired on {} examples".format(result["n_paired"]))
    for axis in ("exact", "posture", "activity"):
        m = result["axes"][axis]
        print("  {:9s} {:>6.1%} -> {:>6.1%}   +{} fixed / -{} broken   p={:.3f}".format(
            axis, m["error_a"], m["error_b"], m["fixed_by_b"], m["broken_by_b"],
            m["p_value"]))
    print("-> {}".format(path))


# ---------------------------------------------------------------------- cli

def _run_frames(args, cli, root, manifest, done, results_path):
    """--unit frame: one question per keyframe listing every cow on it
    (frame.py); one record per cow written, as the per-cow run writes them."""
    frames = [f for f in frame_mod.group(manifest) if any(c["id"] not in done for c in f["cows"])]
    lock = threading.Lock()
    counter = {"n": 0}
    out_fh = open(results_path, "a", encoding="utf-8")

    def work(f):
        cows = f["cows"]
        try:
            box = cbvd.Box(f["video_id"], f["timestamp"], 0, 0, 0, 0, "1", ())
            img = frame_mod.render(cbvd.frame_path(root, box), cows, args.max_width)
            url = render_mod.to_data_url(img, quality=args.jpeg_quality)
            text = frame_mod.prompt(cows)
            if args.answer_now:
                out = cli.complete([url], text, prefill=frame_mod.ANSWER_PREFILL,
                                   max_tokens=24 * len(cows) + 64)
            else:
                out = cli.complete([url], text, schema=frame_mod.SCHEMA, name="cows_in_frame")
            extra = {"frame_seconds": out["seconds"], "frame_usage": out["usage"],
                     "finish_reason": out["finish_reason"]}
            recs = frame_mod.records(cows, out["raw"], extra)
        except Exception as exc:
            recs = [dict(c, error="{}: {}".format(type(exc).__name__, exc)) for c in cows]
        with lock:
            for rec in recs:
                if rec["id"] not in done:
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            counter["n"] += 1
            print("\r  {}/{} frames".format(counter["n"], len(frames)), end="", flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, frames))
    finally:
        out_fh.close()
        print()
    print("-> {}".format(results_path))


def _run_detected(args, cli, root, manifest, done, results_path, dets):
    """--boxes: the model is asked about the detector's boxes, per frame
    (--unit frame) or per box; its answers are matched to the annotated cows
    by overlap. One record per annotated cow - a cow the detector missed is
    an error - and the answers about detections that matched no annotated cow
    go to detections_answered.jsonl."""
    frames = [f for f in frame_mod.group(manifest) if any(c["id"] not in done for c in f["cows"])]
    lock = threading.Lock()
    counter = {"n": 0, "missed": 0, "extra": 0}
    out_fh = open(results_path, "a", encoding="utf-8")
    extra_fh = open(os.path.join(args.out, "detections_answered.jsonl"), "a", encoding="utf-8")

    def ask_cow(c, path):
        img = render_mod.render(path, c["bbox"], mode="marked", max_width=args.max_width)
        out = cli.classify([render_mod.to_data_url(img, quality=args.jpeg_quality)])
        return dict(c, posture=out.get("posture"), activity=out.get("activity"), raw=out.get("raw"),
                    **({"parse_error": out["parse_error"]} if "parse_error" in out else {}))

    def work(f):
        det = dets.get((f["video_id"], f["timestamp"]), {"boxes": []})
        cows = frame_mod.detected_cows(dict(det, video_id=f["video_id"], timestamp=f["timestamp"]),
                                       args.det_threshold)
        answered = []
        try:
            path = cbvd.frame_path(root, cbvd.Box(f["video_id"], f["timestamp"], 0, 0, 0, 0, "1", ()))
            if cows and args.unit == "frame":
                img = frame_mod.render(path, cows, args.max_width)
                url = render_mod.to_data_url(img, quality=args.jpeg_quality)
                if args.answer_now:
                    out = cli.complete([url], frame_mod.prompt(cows), prefill=frame_mod.ANSWER_PREFILL,
                                       max_tokens=24 * len(cows) + 64)
                else:
                    out = cli.complete([url], frame_mod.prompt(cows), schema=frame_mod.SCHEMA,
                                       name="cows_in_frame")
                answered = frame_mod.records(cows, out["raw"])
            elif cows:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    answered = list(pool.map(lambda c: ask_cow(c, path), cows))
            recs, extras = frame_mod.to_gt(f["cows"], answered)
        except Exception as exc:
            recs = [dict(c, error="{}: {}".format(type(exc).__name__, exc)) for c in f["cows"]]
            extras = []
        with lock:
            for rec in recs:
                if rec["id"] not in done:
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for e in extras:
                extra_fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            out_fh.flush()
            extra_fh.flush()
            counter["n"] += 1
            counter["missed"] += sum(bool(r.get("missed_by_detector")) for r in recs)
            counter["extra"] += len(extras)
            print("\r  {}/{} frames  ({} annotated cows missed by the detector, {} extra boxes)".format(
                counter["n"], len(frames), counter["missed"], counter["extra"]), end="", flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, frames))
    finally:
        out_fh.close()
        extra_fh.close()
        print()
    print("-> {}".format(results_path))


# ------------------------------------------------------- detect and stress

def cmd_detect(args):
    detect_mod.run(args, client_mod, _jsonl_read)


def cmd_detect_score(args):
    results = _jsonl_read(os.path.join(args.out, "detect_results.jsonl"))
    if not results:
        sys.exit("no detect results in {} - run `detect` first".format(args.out))
    m = detect_mod.score(results, args.box_format)
    with open(os.path.join(args.out, "detect_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2, ensure_ascii=False)
    meta = _load_meta_file(os.path.join(args.out, "detect_meta.json"))
    path = os.path.join(args.out, "detect_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(detect_mod.render_report(meta, m))
    d5, d3 = m["iou0.5"], m["iou0.3"]
    print("keyframes         : {} ({} annotated cows)".format(m["n_keyframes"], m["n_annotated"]))
    print("recall / precision: {:.1%} / {:.1%} at IoU 0.5,  {:.1%} / {:.1%} at IoU 0.3".format(
        d5["recall"], d5["precision"], d3["recall"], d3["precision"]))
    print("count per frame   : {:.1f} annotated, {:.1f} found, off by {:.2f}".format(
        m["count"]["annotated_mean"], m["count"]["found_mean"], m["count"]["mean_abs_error"]))
    if m["end_to_end_error"] is not None:
        print("end to end error  : {:.1%}  (found and both answers right)".format(
            m["end_to_end_error"]))
    print("recall by format  : " + ", ".join(
        "{} {:.1%}".format(k, v) for k, v in m["recall_iou0.5_by_format"].items()))
    print("-> {}".format(path))


def _load_meta_file(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_stress(args):
    stress_mod.run(args, client_mod, _jsonl_read)


def _server_args(sp, max_tokens):
    sp.add_argument("--root", default=None, help="override the root recorded by plan")
    sp.add_argument("--base-url", default="http://127.0.0.1:8000")
    sp.add_argument("--model", default="muse-glimmer")
    sp.add_argument("--api-key", default="EMPTY")
    sp.add_argument("--max-width", type=int, default=1920)
    sp.add_argument("--jpeg-quality", type=int, default=90)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--max-tokens", type=int, default=max_tokens)
    sp.add_argument("--timeout", type=float, default=600.0)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="cowbench", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="run directory (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="choose the examples")
    sp.add_argument("--root", default=DEFAULT_ROOT)
    sp.add_argument("--annotations", default=DEFAULT_ANN)
    sp.add_argument("--video", action="append",
                    help="clip id, repeatable; omit for the whole split")
    sp.add_argument("--clips", type=int, default=0,
                    help="sample this many whole clips (keeps tracks intact)")
    sp.add_argument("--limit", type=int, default=0,
                    help="cap on individual boxes; breaks tracks, use --clips "
                         "unless the run is only comparing prompts")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_plan)

    sr = sub.add_parser("run", help="query the model")
    sr.add_argument("--root", default=None, help="override the root recorded by plan")
    sr.add_argument("--base-url", default="http://127.0.0.1:8000")
    sr.add_argument("--model", default="muse-glimmer")
    sr.add_argument("--model-repo", default="RedHatAI/Muse-Glimmer-30B-FP8-block")
    sr.add_argument("--quantization", default=None,
                    help="override the quantization string in the report")
    sr.add_argument("--api-key", default="EMPTY")
    sr.add_argument("--mode", choices=("marked", "crop", "both"), default="marked")
    sr.add_argument("--frames", type=int, default=1,
                    help="frames per example: 1 uses the annotated keyframe, "
                         ">1 decodes that many from the clip (needs opencv)")
    sr.add_argument("--span", type=float, default=2.0,
                    help="seconds spanned by --frames, centred on the keyframe")
    # 1920 is the keyframes' native width, i.e. no downscale at all. Measured
    # against 1280 on 300 paired examples: 21 fixed, 10 broken, and the gain
    # falls off monotonically with box size (10 fixed in the smallest quartile,
    # 2 in the largest) - the model was simply short of pixels on distant cows.
    # Costs ~2x the image tokens (1448 -> 2943 per request).
    sr.add_argument("--max-width", type=int, default=1920)
    sr.add_argument("--min-width", type=int, default=0,
                    help="enlarge images narrower than this; a crop of a distant "
                         "cow is otherwise too few patches to read")
    sr.add_argument("--jpeg-quality", type=int, default=90)
    sr.add_argument("--temperature", type=float, default=0.0)
    # The model reasons before answering and the reasoning scales with the
    # number of images; 2048 is enough for one frame and not for five.
    sr.add_argument("--max-tokens", type=int, default=4096)
    sr.add_argument("--concurrency", type=int, default=4)
    sr.add_argument("--timeout", type=float, default=300.0)
    sr.add_argument("--retries", type=int, default=3)
    sr.add_argument("--fresh", action="store_true", help="discard previous results")
    sr.add_argument("--boxes", default=None,
                    help="detections.jsonl from detector.py: ask about the detector's boxes, not "
                         "the annotation's; the annotation only scores (missed cows count as errors)")
    sr.add_argument("--det-threshold", type=float, default=None,
                    help="detector score threshold (default: the one in det_meta.json)")
    sr.add_argument("--unit", choices=("cow", "frame"), default="cow",
                    help="cow: one question per cow; frame: one per keyframe, every cow on it "
                         "numbered (frame.py) - results are still one record per cow")
    sr.add_argument("--answer-now", action="store_true",
                    help="no reasoning, no json_schema: start the answer for the model, as a "
                         "LoRA from lora/train_lora.py was trained (use with --max-width 896)")
    sr.set_defaults(func=cmd_run)

    ss = sub.add_parser("score", help="compute metrics")
    # Free accuracy: the same cow is answered on up to six keyframes, so the
    # per-frame answers are repeated measurements and a majority over the track
    # throws away the ones a passing animal or an awkward moment spoiled.
    ss.add_argument("--vote", action="store_true",
                    help="replace each answer with the majority over its track "
                         "before scoring (writes metrics-voted.json)")
    ss.set_defaults(func=cmd_score)

    srp = sub.add_parser("report", help="render report.md")
    srp.add_argument("--output", default=None)
    srp.add_argument("--no-rows", action="store_true",
                     help="omit the per-cow table (it is one line per example)")
    srp.set_defaults(func=cmd_report)

    sc = sub.add_parser("compare", help="paired comparison of two runs")
    sc.add_argument("--a", required=True, help="baseline run directory")
    sc.add_argument("--b", required=True, help="run directory to compare against it")
    sc.add_argument("--label-a", default=None)
    sc.add_argument("--label-b", default=None)
    sc.add_argument("--output", default=None)
    sc.set_defaults(func=cmd_compare)

    sd = sub.add_parser("detect", help="find every cow on bare keyframes (no box given)")
    _server_args(sd, 8192)   # reasoning plus a list of ~8 cows
    sd.add_argument("--concurrency", type=int, default=4)
    sd.add_argument("--retries", type=int, default=3)
    sd.add_argument("--fresh", action="store_true", help="discard previous detect results")
    sd.set_defaults(func=cmd_detect)

    sds = sub.add_parser("detect-score", help="score detect results against the annotation")
    sds.add_argument("--box-format", choices=detect_mod.BOX_FORMATS, default="norm1000",
                     help="how the model's box numbers are read (the prompt asks for 0-1000)")
    sds.set_defaults(func=cmd_detect_score)

    sst = sub.add_parser("stress", help="N cameras at once: throughput and latency")
    _server_args(sst, 4096)
    sst.add_argument("--task", choices=("classify", "frame", "detect"), default="classify",
                     help="classify: one request per annotated cow; frame: one per frame about "
                          "every cow on it; detect: one per frame, finding the cows")
    sst.add_argument("--streams", type=int, default=12, help="cameras, one val clip each")
    sst.add_argument("--duration", type=float, default=300, help="seconds measured")
    sst.add_argument("--warmup", type=float, default=30, help="seconds run first, not counted")
    sst.add_argument("--interval", type=float, default=0,
                     help="seconds between frames per camera; 0 = as fast as possible")
    sst.add_argument("--answer-now", action="store_true",
                     help="classify as a LoRA from lora/train_lora.py was trained (see run)")
    sst.add_argument("--boxes", default=None,
                     help="detections.jsonl from detector.py: the detector's boxes, not the annotation's")
    sst.add_argument("--det-threshold", type=float, default=None)
    sst.set_defaults(func=cmd_stress)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
