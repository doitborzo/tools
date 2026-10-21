#!/usr/bin/env python3
"""herd - one entry point for the barn system.

Training (on CBVD-5 today, on the barn's own footage later):
    herd.py extract  --split train|val      frozen DINOv2 vectors of every crop (Stage A input)
    herd.py motion   --split train|val      the rhythm of every burst (chewing), CPU only
    herd.py keys     --split train          keyframe crops from the detector's boxes (+ jittered), for --det-keys
    herd.py degrade  --split train|val      spoilt burst frames (occlusion, mud, blur, dark), for --quality
    herd.py train                           temporal transformer + all heads on those vectors
    herd.py eval                            val scores + cowbench-format results
    herd.py frame-id --out R                one frame -> its burst's fingerprint (mct.py tracks by it)
    herd.py reid fetch|extract|train        identity taught on public sets with cow ids (reid_pod.sh)
    herd.py eval-det --run R                the same on the detector's boxes: detector misses count
    herd.py calibrate --detector D          the detector's boxes reshaped to the annotation's habit
    herd.py show-dets --src DIR --detector D  every box a detector draws on a folder / video, as an HTML gallery
    herd.py abstain                         the NaN model and its cut-off

Running:
    herd.py run --config barn.toml [--start ISO]   cameras (RTSP) or files -> herd.sqlite
    herd.py stress --model M --detector D          one GPU: a burst alone, then 5 cameras live
    herd.py show-ids --config barn.toml --cam cam1 --video F --start ISO   who is who, drawn on a recorded file

Barn data for identity (once the barn runs with [training_cache] enabled):
    herd.py barn sheet --cache C --out tracks.html     the labelling page: name each track's cow -> merges.csv
    herd.py barn export --cache C --merges merges.csv   -> herd_barn_feat_*.tgz for the identity model (reid_pod.sh)
    herd.py barn video --config barn.toml --videos DIR --out D   recorded camera files -> D/training_cache
    herd.py barn build --cache C --out F [--merges merges.csv] [--val-from DAY]
    then train --extra-train F/train, eval --extra-val F/val
Cameras' zones, drawn on their pictures:
    herd.py zones page --config barn.toml --videos DIR --out D    D/zones.html -> cameras.toml
    herd.py zones apply --config barn.toml --cameras cameras.toml
Recordings for them:
    herd.py record --config barn.toml --out DIR --minutes 60   every camera at once -> DIR/<cam>_<time>.mp4
    herd.py fix-videos --videos DIR --out DIR2             damaged recordings -> clean .mp4 the tools read
A dress rehearsal on recordings (two passes: the gallery fills, then who is who, drawn per camera):
    herd.py trial --config barn.toml --videos DIR --out D [--boxes B --labels L]   (L: your ids -> IDF1)
Cameras sharing floor calibrated by the cows moving in both (nothing measured; then [mct] link):
    herd.py autocalib --config barn.toml --store D/pass2/herd.sqlite --videos DIR --out C [--apply]
The models on the barn's own labels (nothing trained on them):
    herd.py check --config barn.toml [--det herd_det_barn_X.tgz] [--ids herd_barn_ids_X.tgz]
Cow boxes on the barn's cameras (detector training):
    herd.py boxes frames --config barn.toml --videos DIR --out D   frames + the detector's guess + D/boxes.html
    herd.py boxes export --data D --labels labels.json            -> herd_det_barn_*.tgz (run_pod.sh detector)

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
    if cmd in ("motion", "keys", "degrade"):
        import cbvd_bursts
        return cbvd_bursts.main([cmd] + argv)
    if cmd in ("train", "eval", "frame-id"):
        import train
        return train.main([cmd] + argv)
    if cmd == "calibrate":
        import calib_boxes
        return calib_boxes.main(argv)
    if cmd == "reid":
        import reid
        return reid.main(argv)
    if cmd == "show-ids":
        import show_ids
        return show_ids.main(argv)
    if cmd == "show-dets":
        import show_dets
        return show_dets.main(argv)
    if cmd == "eval-det":
        import eval_det
        return eval_det.main(argv)
    if cmd == "abstain":
        import abstain
        return abstain.main(argv)
    if cmd == "run":
        import pipeline
        return pipeline.main(argv)
    if cmd == "stress":
        import stress
        return stress.main(argv)
    if cmd == "barn":
        import barn_dataset
        return barn_dataset.main(argv)
    if cmd == "boxes":
        import box_dataset
        return box_dataset.main(argv)
    if cmd == "zones":
        import zone_page
        return zone_page.main(argv)
    if cmd == "autocalib":
        import autocalib
        return autocalib.main(argv)
    if cmd == "trial":
        import trial
        return trial.main(argv)
    if cmd == "check":
        import check
        return check.main(argv)
    if cmd == "fix-videos":
        import fix_videos
        return fix_videos.main(argv)
    if cmd == "record":
        import record
        return record.main(argv)
    if cmd == "report":
        import report
        return report.main(argv)
    if cmd == "gallery":
        return gallery_cmd(argv)
    print(f"unknown command {cmd}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
