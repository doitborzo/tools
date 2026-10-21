#!/usr/bin/env python3
"""Version 14 (E) scored again over time: identification minute by minute after the setup hour,
from what system_e.py saved (tracks_e_<way>.jsonl, names_e_<way>.json) - nothing runs again.

    python herd/v14/rescore_e.py --data /workspace/v14/mmcows --system /workspace/v14/system_V14E --way built

Prints and writes <system>/by_minute_<way>.json: per minute (and for the first 1, 5, 10, 30 minutes
together) ID F1 with 'not sure', the share answered, and ID F1 always answering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import v14common as C  # noqa: F401  (paths)
from v14common import local_hour, read_jsonl, write_json


def load(folder, way):
    names = json.load(open(os.path.join(folder, f"names_e_{way}.json"), encoding="utf-8"))
    det_log = [(r["t"], r["cam"], r["box"], r["track"], r["conf"]) for r in read_jsonl(os.path.join(folder, f"tracks_e_{way}.jsonl"))]
    said = {(cid, k): (n, a) for cid, k, n, a in names["names"]}
    return names, det_log, said


def by_minute(data, folder, way, start=12.0, end=15.0, minutes=30, match_iou=0.3):
    import metrics
    import system_e as E
    names, det_log, said = load(folder, way)
    t_setup = names["t_setup"]
    labels = [r for r in read_jsonl(os.path.join(data, "labels.jsonl")) if start <= local_hour(r["ts"]) < end]
    a = argparse.Namespace(step=names["step"], match_iou=match_iou)
    to_cow = names["album_to_cow"]

    def window(m0, m1):
        sel = [r for r in labels if t_setup + 60 * m0 <= r["ts"] < t_setup + 60 * m1]
        r = E.score_e(a, det_log, sel, said, to_cow, names["t_start"], t_setup, metrics)
        w, al = r["with_not_sure"], r["always_answer"]
        return {"sightings": w["cases"], "f1": w["f1"], "answered": w["answered_share"], "precision": w["precision"],
                "f1_always": al["f1"]}

    out = {"way": way, "minutes": {m + 1: window(m, m + 1) for m in range(minutes)},
           "first": {n: window(0, n) for n in (1, 5, 10, 30) if n <= minutes}}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--way", default="built", choices=("built", "confirmed", "oracle"))
    p.add_argument("--minutes", type=int, default=30)
    a = p.parse_args(argv)
    out = by_minute(a.data, a.system, a.way, minutes=a.minutes)
    write_json(os.path.join(a.system, f"by_minute_{a.way}.json"), out)
    pc = lambda v: "   -  " if v is None else f"{v:6.1%}"
    print(f"album: {a.way}   (minute 1 = the first minute after the setup hour)")
    print("minute  sightings  ID F1  answered  F1 always")
    for m, r in out["minutes"].items():
        print(f"{m:6d}  {r['sightings']:9d} {pc(r['f1'])}   {pc(r['answered'])}    {pc(r['f1_always'])}")
    for n, r in out["first"].items():
        print(f"first {n:2d} min: {r['sightings']} sightings, ID F1 {pc(r['f1'])}, answered {pc(r['answered'])}, "
              f"always answering {pc(r['f1_always'])}")


if __name__ == "__main__":
    sys.exit(main())
