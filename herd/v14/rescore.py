#!/usr/bin/env python3
"""Score a saved onboarding test (system.py's tracks_<tag>.jsonl + names_<tag>.json) again, on
chosen cows - nothing runs again, no model is loaded. E.g. version 14 (D) on version 14 (A)'s
five test cows, to compare like with like:

    python herd/v14/rescore.py --data /workspace/v14/mmcows --system /workspace/v14/system_V14D \\
        --cows 2,6,10,11,15 --out /workspace/v14/system_V14D/rescore_A_cows.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import v14common as C  # noqa: F401  (paths)
from v14common import read_jsonl, write_json


def load_run(folder, tag):
    names = json.load(open(os.path.join(folder, f"names_{tag}.json"), encoding="utf-8"))
    det_log = [(r["t"], r["cam"], r["box"], r["track"], r["conf"])
               for r in read_jsonl(os.path.join(folder, f"tracks_{tag}.jsonl"))]
    said = {}
    for tid, m, n, always in names["names"]:
        said[(tid, m)] = n
        said[("always", tid, m)] = always
    track_album = {int(k): v for k, v in names["track_album"].items()}
    return names, det_log, said, track_album


def rescore(data, folder, tag, groups, start=12.0, end=15.0, match_iou=0.3):
    import metrics
    import system
    names, det_log, said, track_album = load_run(folder, tag)
    a = argparse.Namespace(data=data, start=start, end=end, step=names["step"], match_iou=match_iou)
    cams = {d[1] for d in det_log}
    return system.score(a, det_log, track_album, said, names["t_start"], names["t_setup"], cams, metrics, groups)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data_mmcows.py's folder (labels.jsonl)")
    p.add_argument("--system", required=True, help="system.py's --out")
    p.add_argument("--tag", default="assoc")
    p.add_argument("--cows", required=True, help="comma list of MmCows cow ids")
    p.add_argument("--start", type=float, default=12.0)
    p.add_argument("--end", type=float, default=15.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cows = [int(c) for c in a.cows.split(",")]
    res = rescore(a.data, a.system, a.tag, {"chosen cows": cows}, a.start, a.end)
    g = res["identification"]["by_cow_group"]["chosen cows"]
    out = {"cows": cows, "with_not_sure": g["with_not_sure"], "always_answer": g["always_answer"],
           "all_cows": res["identification"], "album": res["album"], "tracking": res["tracking"]}
    write_json(a.out, out)
    p_ = lambda v: "-" if v is None else f"{v:.1%}"
    print(f"[rescore] cows {cows}: ID F1 {p_(g['with_not_sure']['f1'])} (precision {p_(g['with_not_sure']['precision'])}, "
          f"recall {p_(g['with_not_sure']['recall'])}); always answering {p_(g['always_answer']['f1'])} -> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
