#!/usr/bin/env python3
"""One table of every result under the given folders: bench errors, the
detector, the camera tests. Pure Python, runs anywhere - on the pod or on a
PC over an unpacked export.

    python cowbench/summary.py                         # the pod: /workspace/lora-runs + cowbench/runs
    python cowbench/summary.py C:\\Users\\Work\\Downloads   # wherever an export was unpacked
    python cowbench/summary.py <dir> [<dir> ...] -o summary.md

Searched recursively:
    metrics.json / metrics-voted.json   a bench run (cowbench.py score, train_lora.py eval)
    det_metrics.json                    the detector (detector.py score)
    stress_*.json                       a camera test (cowbench.py stress)

"excl. ruminating" leaves out the cows annotated as ruminating - a class the
adapters are not trained on and a still frame cannot show - recomputed from
results.jsonl next to metrics.json (not for the voted rows). "found cows" is
the error on the cows the detector found, for runs on its boxes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception:
        pass
    return rows


def walk(roots, names=None, prefix=None):
    """Files under roots matching a name or a prefix + .json, sorted; symlinked
    folders are not followed (an export holds no links, a pod might)."""
    out = []
    for root in roots:
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in ("node_modules", ".git", "__pycache__", "checkpoints"))
            for f in sorted(files):
                if (names and f in names) or (prefix and f.startswith(prefix) and f.endswith(".json")):
                    out.append(os.path.join(d, f))
    return sorted(set(out))


def label(path, roots):
    for r in sorted(roots, key=len, reverse=True):
        rp = os.path.abspath(r)
        if os.path.abspath(path).startswith(rp + os.sep):
            rel = os.path.relpath(path, rp)
            return rel.replace(os.sep, "/") if len(roots) == 1 else f"{os.path.basename(rp)}/{rel.replace(os.sep, '/')}"
    return path


def no_rumination(results):
    """(exact error, activity error, n) without the cows annotated ruminating."""
    rows = [r for r in results if r.get("gt_activity") != "ruminating" and not r.get("gt_rumination")]
    if not rows:
        return None
    exact = sum(1 for r in rows if r.get("posture") != r.get("gt_posture") or r.get("activity") != r.get("gt_activity"))
    act = sum(1 for r in rows if r.get("activity") != r.get("gt_activity"))
    return exact / len(rows), act / len(rows), len(rows)


def pct(x):
    return "-" if x is None else f"{x:.1%}"


def bench_table(roots):
    lines = ["| run | boxes | cows | exact error | posture error | activity error | "
             "exact / activity excl. ruminating | missed by detector | exact error on found cows |",
             "|---|---|---|---|---|---|---|---|---|"]
    n_rows = 0
    for path in walk(roots, names={"metrics.json", "metrics-voted.json"}):
        m = load(path)
        if not m or "exact_match" not in m:
            continue
        d = os.path.dirname(path)
        meta = load(os.path.join(d, "run_meta.json")) or {}
        boxes = (meta.get("boxes") or {}).get("source", "annotation")
        voted = os.path.basename(path) == "metrics-voted.json"
        n = m.get("n_examples") or 0
        miss = m.get("n_missed_by_detector", 0) or 0
        found = ((m["exact_match"]["error_rate"] * n - miss) / (n - miss)) if miss and n > miss else None
        nr = None if voted else no_rumination(jsonl(os.path.join(d, "results.jsonl")))
        nr_txt = f"{pct(nr[0])} / {pct(nr[1])}" if nr else "-"
        name = label(d, roots) + (" (track vote)" if voted else "")
        lines.append(f"| {name} | {boxes} | {n} | {pct(m['exact_match']['error_rate'])} | "
                     f"{pct(m['posture']['error_rate'])} | {pct(m['activity']['error_rate'])} | {nr_txt} | "
                     f"{miss} | {pct(found)} |")
        n_rows += 1
    return lines if n_rows else []


def detector_lines(roots):
    out = []
    for path in walk(roots, names={"det_metrics.json"}):
        d = load(path)
        if not d:
            continue
        i5, i3 = d.get("iou0.5", {}), d.get("iou0.3", {})
        size = d.get("recall_by_size") or {}
        out.append(f"- `{label(os.path.dirname(path), roots)}`: recall {pct(i5.get('recall'))}, precision "
                   f"{pct(i5.get('precision'))} at IoU 0.5 (at 0.3: {pct(i3.get('recall'))} / "
                   f"{pct(i3.get('precision'))}); threshold {d.get('threshold')}; "
                   f"{d.get('ms_per_frame')} ms a frame"
                   + (f"; recall by size: " + ", ".join(f"{k} {pct(v)}" for k, v in size.items()) if size else ""))
    return out


def stress_table(roots):
    lines = ["| camera test | model / checkpoint / GPU | frames/s, all cameras | update per camera, s | "
             "frame latency p50 / p90, s | detector p50, ms | keeps up |",
             "|---|---|---|---|---|---|---|"]
    n_rows = 0
    f = lambda v, dd=1: "-" if v is None else f"{v:.{dd}f}"
    for path in walk(roots, prefix="stress_"):
        s = load(path)
        if not s or "frames_per_s_total" not in s:
            continue
        fps = s.get("frames_per_s_total") or 0
        lat = s.get("frame_latency_s") or {}
        det = ((s.get("live_detector") or {}).get("detect_ms") or {}).get("p50")
        keeps = (s.get("paced") or {}).get("keeps_up")
        mode = f"one frame per {s['interval']:g} s" if s.get("interval") else "as fast as possible"
        who = " / ".join(x for x in (s.get("model"), s.get("base_model"), s.get("gpu")) if x)
        lines.append(f"| {label(path, roots)}: {s.get('streams')} cameras, {s.get('task')}, {mode} | {who} | "
                     f"{fps:.2f} | {f(s['streams'] / fps if fps else None)} | "
                     f"{f(lat.get('p50'))} / {f(lat.get('p90'))} | {f(det, 0)} | "
                     f"{'-' if keeps is None else ('yes' if keeps else 'no')} |")
        n_rows += 1
    return lines if n_rows else []


def build(roots):
    out = ["# Results", ""]
    bench = bench_table(roots)
    if bench:
        out += ["## Bench", "", *bench, "",
                "Missed cows count as errors in every column; \"found cows\" leaves them out.", ""]
    det = detector_lines(roots)
    if det:
        out += ["## Detector", "", *det, ""]
    stress = stress_table(roots)
    if stress:
        out += ["## Cameras at once", "", *stress, ""]
    if len(out) == 2:
        out.append("No results found under: " + ", ".join(roots))
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="*", help="folders to search (default: /workspace/lora-runs and cowbench/runs)")
    p.add_argument("-o", "--output", default=None, help="write here as well as to the screen")
    args = p.parse_args(argv)
    roots = args.roots or [d for d in ("/workspace/lora-runs", os.path.join(HERE, "runs")) if os.path.isdir(d)]
    roots = [r for r in roots if os.path.isdir(r)] or sys.exit("no such folder: " + ", ".join(args.roots))
    text = build(roots)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:   # an old Windows console
        sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
