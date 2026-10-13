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
    for f in sorted(glob.glob(os.path.join(sout, "*.md"))):
        with open(f, encoding="utf-8") as fh:
            lines += [f"## Stress test ({os.path.basename(f)})", "", fh.read(), ""]
    text = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
