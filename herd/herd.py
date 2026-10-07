#!/usr/bin/env python3
"""herd - one entry point for the barn system.

Training (on CBVD-5 today, on the barn's own footage later):
    herd.py extract  --split train|val      frozen DINOv2 vectors of every crop (Stage A input)
    herd.py train                           temporal transformer + all heads on those vectors
    herd.py eval                            val scores + cowbench-format results
    herd.py abstain                         the NaN model and its cut-off

Running:
    herd.py run --config barn.toml [--start ISO]   cameras (RTSP) or files -> herd.sqlite
    herd.py stress --model M --detector D          one GPU: a burst alone, then 5 cameras live

Reports:
    herd.py report --config barn.toml csv [--end DAY]       the 3-day CSV
    herd.py report --config barn.toml alerts [--day DAY]    alerts for the vet
    herd.py report --config barn.toml verdict --alert N --confirmed yes|no [--diagnosis ...]
    herd.py report --config barn.toml alert-stats           how often alerts were right
    herd.py report --config barn.toml daily                 per cow per day
    herd.py gallery --config barn.toml                      the cows the system knows
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def gallery_cmd(argv):
    import argparse
    from common import load_config
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    path = os.path.join(os.path.dirname(os.path.abspath(cfg["store"]["path"])), "gallery", "gallery.json")
    if not os.path.exists(path):
        print("no gallery yet - it is created by the first run")
        return
    meta = json.load(open(path, encoding="utf-8"))
    import datetime as dt
    for cid, c in sorted(meta["cows"].items()):
        seen = dt.datetime.fromtimestamp(c["last_seen"]).isoformat(timespec="minutes")
        print(f"{cid:24s} prototypes {c['prototypes']:2d}  last seen {seen}")
    print(f"{len(meta['cows'])} cows, {len(meta['pool_tracks'])} unknown bursts waiting to be grouped")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "extract":
        import cbvd_bursts
        return cbvd_bursts.main(argv)
    if cmd in ("train", "eval"):
        import train
        return train.main([cmd] + argv)
    if cmd == "abstain":
        import abstain
        return abstain.main(argv)
    if cmd == "run":
        import pipeline
        return pipeline.main(argv)
    if cmd == "stress":
        import stress
        return stress.main(argv)
    if cmd == "report":
        import report
        return report.main(argv)
    if cmd == "gallery":
        return gallery_cmd(argv)
    print(f"unknown command {cmd}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
