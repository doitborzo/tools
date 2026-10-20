#!/usr/bin/env python3
"""How the models do on the barn's own labelled frames - before anything is trained on them.

  detector   every frame of herd_det_barn_*.tgz (train and valid: the detector has seen none of
             them): cows found and boxes right (IoU 0.5) at the config's threshold, at the detector's
             own, and the threshold with the best F1 on this barn - the one for [detector] threshold.
  identity   herd_barn_ids_*.tgz: DINOv2 on the named crops as the barn does, then each identity
             model (the config's [model] identity, and the herd model's own fingerprint to compare)
             on all its cows: a cow's group of crops from one camera against the other cameras' at
             the same time (other_camera_top1) and from one crop (frame_other_camera_top1); the
             thresholds this model would get calibrated on these cows.

    python herd/herd.py check --config barn.toml --det herd_det_barn_X.tgz --ids herd_barn_ids_X.tgz
    (default: the newest archives next to the config) -> <work>/check.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402,F401  (puts cowbench on the path)
from common import load_config  # noqa: E402


def newest(folder, pattern):
    found = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getmtime)
    return found[-1] if found else None


def unpack(archive, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    with tarfile.open(archive) as tar:
        tar.extractall(dest)
    return dest


def check_detector(cfg, archive, work, extra):
    from PIL import Image
    import detector as det_mod
    import extra_det
    root = os.path.join(unpack(archive, os.path.join(work, "det")), "extra_det", "barn")
    frames = []
    for part in ("train", "valid"):
        folder = os.path.join(root, part)
        if os.path.exists(os.path.join(folder, "_annotations.coco.json")):
            frames += [dict(f, folder=folder) for f in extra_det.load_coco(folder)]
    print(f"\n[check] detector on {len(frames)} labelled frames, {sum(len(f['boxes']) for f in frames)} cows",
          flush=True)
    out = {}
    for best in [cfg["detector"]["weights"]] + list(extra):
        if not best or not os.path.exists(os.path.join(best, "det_train_meta.json")):
            continue
        live = det_mod.Live(best, threshold=det_mod.KEEP_SCORE, tiles=cfg["detector"].get("tiles"))
        own = json.load(open(os.path.join(best, "det_train_meta.json")))["threshold"]
        dets = {}
        for i in range(0, len(frames), 8):
            chunk = frames[i:i + 8]
            imgs = [Image.open(os.path.join(f["folder"], f["video_id"])).convert("RGB") for f in chunk]
            for f, found in zip(chunk, live.many(imgs)):
                dets[(f["video_id"], 0)] = [c["bbox"] + [c["det_score"]] for c in found]
            print(f"\r  {min(i + 8, len(frames))}/{len(frames)} frames", end="", flush=True)
        print(flush=True)
        thr_cfg = cfg["detector"]["threshold"] or own
        at_cfg, at_own = det_mod.prf(dets, frames, thr_cfg, 0.5), det_mod.prf(dets, frames, own, 0.5)
        thr, m = det_mod.best_threshold(dets, frames, "f1")
        b = os.path.abspath(best).rstrip("/")
        name = os.path.basename(os.path.dirname(b)) if os.path.basename(b) == "best" else os.path.basename(b)
        out[name] = {"threshold_config": thr_cfg, "at_config": at_cfg, "threshold_own": own, "at_own": at_own,
                     "best_f1_threshold": thr, "at_best": m}
        print(f"  {name}")
        for label, t, r in (("config", thr_cfg, at_cfg), ("its own", own, at_own), ("best F1", thr, m)):
            print(f"    at {label:8s} threshold {t:<5g}: found {r['recall']:.1%} of the cows, {r['precision']:.1%} "
                  f"of its boxes right, F1 {r['f1']:.3f}")
    return out


def check_identity(cfg, archive, work):
    import torch
    import reid
    from common import device
    from model import HerdModel
    data = os.path.join(unpack(archive, os.path.join(work, "ids")), "reid")
    rows = [json.loads(x) for x in open(os.path.join(data, "raw", "barn_ids", "index.jsonl"), encoding="utf-8")]
    for r in rows:                     # nothing here is trained on: every cow is a test cow
        r["split"] = "val"
    with open(os.path.join(data, "raw", "barn_ids", "index.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    ckpt = cfg["model"]["checkpoint"]
    print(f"\n[check] identity on {len(rows)} named crops of {len({r['cow'] for r in rows})} cows "
          f"(DINOv2 on them first)", flush=True)
    reid.main(["extract", "--data", data, "--sources", "barn_ids", "--model", ckpt])
    split = reid.Bursts(data, ["barn_ids"], "val")
    dev = device()
    base, ck = HerdModel.load(ckpt, map_location=dev)
    models = {"herd model (no identity model)": base.temporal.to(dev).eval()}
    ip = cfg["model"].get("identity")
    if ip and os.path.exists(ip):
        models[f"identity model {os.path.basename(os.path.dirname(ip))}"] = reid.load_identity(
            ip, ck.get("features", {}), dev)[0]
    cams = {}
    for b in split.bursts:
        cams.setdefault(b["identity"], set()).add(b["cam"])
    multi = sum(1 for v in cams.values() if len(v) > 1)
    out = {}
    for name, model in models.items():
        with torch.no_grad():
            m, _ = reid.evaluate(model, split)
            th = reid.calibrate(model, split, None)
        out[name] = {"metrics": m, "thresholds": th}
        print(f"  {name}:")
        if "other_camera_top1" in m:
            print(f"    a cow from another camera: the nearest of all the cows is her in {m['other_camera_top1']:.1%} "
                  f"(groups of crops), {m.get('frame_other_camera_top1', 0):.1%} (one crop); "
                  f"{m['other_camera_queries']} questions, {multi} cows seen by 2+ cameras, chance ~{1 / max(1, len(cams)):.1%}")
        else:
            print("    no cow named on two cameras: nothing to ask across cameras")
        for k in ("new_cow_similarity", "merge_similarity"):
            if k in th:
                print(f"    calibrated on these cows: {k} {th[k]:.3f} (the same cow at or above: {th['recall'][k]:.1%})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--det", default=None, help="herd_det_barn_*.tgz (default: the newest next to the config)")
    p.add_argument("--ids", default=None, help="herd_barn_ids_*.tgz (default: the newest next to the config)")
    p.add_argument("--detector", action="append", default=[], help="another best/ folder to compare")
    p.add_argument("--work", default=None, help="default: check/ next to the config")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    here = os.path.dirname(os.path.abspath(args.config))
    det = args.det or newest(here, "herd_det_barn_*.tgz")
    ids = args.ids or newest(here, "herd_barn_ids_*.tgz")
    work = os.path.abspath(args.work or os.path.join(here, "check"))
    os.makedirs(work, exist_ok=True)
    report = {"det_archive": det, "ids_archive": ids}
    if det:
        report["detector"] = check_detector(cfg, det, work, args.detector)
    else:
        print("[check] no herd_det_barn_*.tgz: the detector is not checked (herd.py boxes export makes it)")
    if ids:
        report["identity"] = check_identity(cfg, ids, work)
    else:
        print("[check] no herd_barn_ids_*.tgz: identity is not checked (herd.py boxes ids makes it)")
    with open(os.path.join(work, "check.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\n-> {os.path.join(work, 'check.json')}")


if __name__ == "__main__":
    sys.exit(main())
