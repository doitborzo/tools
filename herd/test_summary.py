#!/usr/bin/env python3
"""test_pod.sh's summary: the detectors on the public sets' valid frames and the stress test, as Markdown.

    python herd/test_summary.py <gpu name> <detector> <eval_extra.json> <stress folder> <out.md>
"""

import glob
import json
import os
import sys


def main(argv=None):
    gpu, det, ev, sout, out = (argv or sys.argv[1:])[:5]
    lines = [f"# herd test on {gpu}", "", f"stress test and galleries with `{det}`", ""]
    if os.path.exists(ev):
        with open(ev, encoding="utf-8") as fh:
            e = json.load(fh)
        lines += ["## Usual test: the detectors on fully boxed frames (valid, IoU 0.5, each at its own threshold)", "",
                  "| detector | set | frames | recall | precision | recall at IoU 0.3 | best F1 (threshold) |",
                  "|---|---|---|---|---|---|---|"]
        for tag, r in e.items():
            for name, m in r.items():
                if isinstance(m, dict) and "iou0.5" in m:
                    a, b = m["iou0.5"], m["iou0.3"]
                    lines.append(f"| {tag} (thr {r['threshold']}) | {name} | {m['frames']} | {a['recall']:.1%} | "
                                 f"{a['precision']:.1%} | {b['recall']:.1%} | "
                                 f"{m['best_f1']['f1']:.3f} ({m['best_f1_threshold']}) |")
        lines.append("")
    runs = []
    for f in glob.glob(os.path.join(sout, "stress_herd_*cam.json")):
        with open(f, encoding="utf-8") as fh:
            L = json.load(fh).get("live")
        if L:
            runs.append(L)
    if runs:
        # every camera count side by side: where this GPU stops keeping up
        lines += ["## Stress test: cameras side by side", "",
                  "| cameras | keeps up | ticks done | tick lag p99, s | bursts done | burst ready -> done max, s "
                  "| GPU busy | cameras estimate (20% free) |", "|---|---|---|---|---|---|---|---|"]
        for L in sorted(runs, key=lambda r: r["cameras"]):
            lines.append(f"| {L['cameras']} | {'yes' if L['keeps_up'] else 'NO'} | {L['ticks_done']} of "
                         f"{L['ticks_expected']} | {L['tick_lag_s'].get('p99', '-')} | {L['bursts_done']} of "
                         f"{L['bursts_due']} | {L['burst_late_s'].get('max', '-')} | {L['gpu_busy_share']:.0%} | "
                         f"{L['cameras_max_estimate']} |")
        lines.append("")
    for f in sorted(glob.glob(os.path.join(sout, "*.md")),
                    key=lambda f: int("".join(c for c in os.path.basename(f) if c.isdigit()) or 0)):
        with open(f, encoding="utf-8") as fh:
            lines += [f"## Stress test ({os.path.basename(f)})", "", fh.read(), ""]
    text = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
