#!/usr/bin/env python3
"""RF-DETR (Roboflow, Apache-2.0 for N/S/M/L) as the cow detector, in place of RT-DETRv2.

RF-DETR is a DETR whose backbone is DINOv2 (windowed): a detector built on the
same family of features herd's frame encoder uses. Trained on the same CBVD-5
boxes, held-out clips and threshold choice as detector.py, and saved as a best/
folder that detector.Live loads like any other - so herd's pipeline, eval-det,
stress and show_zone take it with no change.

    python cowbench/rfdetr_det.py train --root /workspace/cbvd5 --out /workspace/herd/detector_rfdetr_l --size large
    python herd/herd.py eval-det --run /workspace/herd/run7 --detector /workspace/herd/detector_rfdetr_l/best

    export  CBVD-5 train keyframes -> COCO folders (train/, valid/, test/; images linked, not copied)
            --extra <folder>...: public sets fetched by extra_det.py added (one class: cow)
            --init <best/>: continue from a detector trained before (same size; fewer epochs, lower lr)
    train   export + rfdetr training + the threshold on the held-out clips -> <out>/best/

Sizes: nano (384 px), small (512), medium (576), large (704) - all Apache-2.0;
--resolution sets another input side (a multiple of 32 for large, of the model's
patch x windows otherwise). Needs: pip install "rfdetr[train]".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "lora")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cbvd  # noqa: E402
import train_lora  # noqa: E402  (load_train, split_dev)

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def frames_of(rows):
    out = {}
    for r in rows:
        out.setdefault((r["video_id"], r["timestamp"]), []).append(r["bbox"])
    return [{"video_id": v, "timestamp": t, "boxes": b} for (v, t), b in out.items()]


def write_coco(root, frames, folder, extra=(), repeat=None):
    """Keyframes -> folder/_annotations.coco.json + the images linked in (one class: cow);
    extra: [(source name, its COCO folder of this split, at most this many images or 0)]
    appended (extra_det.py's sets); repeat {source name: k}: its images k times (a small set - the
    barn's own few hundred frames - weighs more against CBVD-5's thousands)."""
    from PIL import Image
    import random
    os.makedirs(folder, exist_ok=True)
    images, anns = [], []
    for i, f in enumerate(frames, 1):
        src = cbvd.frame_path(root, cbvd.Box(f["video_id"], f["timestamp"], 0, 0, 0, 0, "1", ()))
        name = f"{f['video_id']}_{f['timestamp']:05d}.jpg"
        dst = os.path.join(folder, name)
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy(src, dst)
        with Image.open(src) as im:
            w, h = im.size
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for b in f["boxes"]:
            x, y, bw, bh = b[0] * w, b[1] * h, (b[2] - b[0]) * w, (b[3] - b[1]) * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [x, y, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
    for name, src, cap in extra:
        with open(os.path.join(src, "_annotations.coco.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        ims = d["images"]
        if cap and len(ims) > cap:
            ims = random.Random(0).sample(ims, cap)
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        for r in range((repeat or {}).get(name, 1)):
            for im in ims:
                i = len(images) + 1
                fname = f"x_{name}{f'_r{r}' if r else ''}_{im['file_name']}"
                dst = os.path.join(folder, fname)
                if not os.path.lexists(dst):
                    try:
                        os.symlink(os.path.abspath(os.path.join(src, im["file_name"])), dst)
                    except OSError:
                        shutil.copy(os.path.join(src, im["file_name"]), dst)
                images.append({"id": i, "file_name": fname, "width": im["width"], "height": im["height"]})
                for a in by.get(im["id"], []):
                    anns.append({**a, "id": len(anns) + 1, "image_id": i, "category_id": 1})
    with open(os.path.join(folder, "_annotations.coco.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "cow", "supercategory": "none"}]}, fh)
    return len(images), len(anns)


def extra_sources(args):
    """--extra folders (extra_det.py fetch) -> [(name, folder)]; a missing one is an error, not skipped."""
    out = []
    for d in args.extra:
        if not os.path.exists(os.path.join(d, "train", "_annotations.coco.json")):
            sys.exit(f"--extra {d}: no train/_annotations.coco.json - fetch it first (cowbench/extra_det.py fetch)")
        out.append((os.path.basename(os.path.normpath(d)), d))
    return out


def repeat(args):
    """--repeat barn=3 -> {"barn": 3}."""
    out = {}
    for r in args.repeat:
        name, _, k = r.partition("=")
        if not k.isdigit() or int(k) < 1:
            sys.exit(f"--repeat {r}: want <set>=<times>, e.g. barn=3")
        out[name] = int(k)
    return out


def cmd_export(args):
    """The same split as detector.py: CBVD-5 train minus the val clips, 10% of clips held out.
    --extra sets join train (each at most --extra-max images) and valid (the checkpoint is chosen on
    CBVD-5's held-out clips + their valid); the threshold is still chosen on CBVD-5 alone."""
    rows, _ = train_lora.load_train(args.root)
    rows, dev_rows = train_lora.split_dev(rows, args.holdout, args.seed)
    tr, dv = frames_of(rows), frames_of(dev_rows)
    ds = os.path.join(args.out, "dataset")
    xs = extra_sources(args)
    for name, fr in (("train", tr), ("valid", dv), ("test", dv)):
        extra = [(x, os.path.join(d, "train" if name == "train" else "valid"), args.extra_max if name == "train" else 0)
                 for x, d in xs]
        n, a = write_coco(args.root, fr, os.path.join(ds, name), extra, repeat(args) if name == "train" else None)
        print(f"[rfdetr] {name}: {n} images ({len(fr)} CBVD-5 keyframes), {a} cows")
    return ds, dv


def load(best_dir, meta):
    """best/ -> (model, predict(images, keep) -> [[x1, y1, x2, y2, score] 0-1 ...] per image, device)."""
    from rfdetr import RFDETR
    model = RFDETR.from_checkpoint(os.path.join(best_dir, meta["checkpoint"]), trust_checkpoint=True)
    # Inference mode: fp16 on a GPU, no compile (the batch size varies: tiles, bursts).
    # rfdetr >= 1.11 calls it inference(); older versions optimize_for_inference().
    import torch
    try:
        if hasattr(model, "inference"):
            model.inference(compile=False, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        else:
            model.optimize_for_inference()
    except Exception as e:                    # plain inference still works, only slower
        print(f"[rfdetr] running without the inference optimisation ({e.__class__.__name__}: {e})", flush=True)
    resolution = meta.get("size")

    def predict(images, keep=0.05):
        if not images:
            return []
        res = model.predict(list(images), threshold=keep,
                            **({"shape": (resolution, resolution)} if resolution else {}))
        res = res if isinstance(res, list) else [res]
        out = []
        for img, d in zip(images, res):
            w, h = img.size
            boxes = [[round(float(x1) / w, 4), round(float(y1) / h, 4), round(float(x2) / w, 4),
                      round(float(y2) / h, 4), round(float(s), 4)]
                     for (x1, y1, x2, y2), s in zip(d.xyxy, d.confidence)]
            out.append(sorted(boxes, key=lambda b: -b[4]))
        return out

    dev = getattr(getattr(model, "model", None), "device", "cuda")
    return model, predict, dev


def init_checkpoint(args):
    """--init: a best/ folder of rfdetr_det.py (or a .pth) -> the checkpoint path; its size and input
    side are kept (a checkpoint of another size does not fit)."""
    if not args.init:
        return None
    path = args.init
    if os.path.isdir(path):
        with open(os.path.join(path, "det_train_meta.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("kind") != "rfdetr":
            sys.exit(f"--init {path}: not an RF-DETR detector (an RT-DETR cannot start an RF-DETR)")
        if m.get("base_model") != f"rfdetr-{args.size}":
            sys.exit(f"--init {path} is {m.get('base_model')}, not rfdetr-{args.size}: set --size to match")
        if not args.resolution and m.get("size"):
            args.resolution = m["size"]
        path = os.path.join(path, m["checkpoint"])
    if not os.path.exists(path):
        sys.exit(f"--init: no checkpoint at {path}")
    # rfdetr reads start weights with weights_only=True; a training checkpoint also pickles its
    # args, so only the weights (and the class names, as plain data) are kept - our own file, trusted
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model") or {k[len("model."):]: v for k, v in ck.get("state_dict", {}).items()
                                if k.startswith("model.")}
    if not state:
        sys.exit(f"--init {path}: no model weights in it")
    a = ck.get("args")
    names = a.get("class_names") if isinstance(a, dict) else getattr(a, "class_names", None)
    slim = os.path.join(args.out, "init_weights.pth")
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": state, **({"args": {"class_names": list(names)}} if names else {})}, slim)
    print(f"[rfdetr] starting from {path}", flush=True)
    return slim


def cmd_train(args):
    import rfdetr
    import detector as det_mod
    from PIL import Image
    best = os.path.join(args.out, "best")
    done = os.path.join(args.out, "train_done")
    if os.path.exists(done) and not args.fresh:
        print(f"[rfdetr] already trained: {best}")
        return
    init = init_checkpoint(args)
    ds, dev_frames = cmd_export(args)
    run_dir = os.path.join(args.out, "rf")
    # --init: start from a detector trained before (its weights, a fresh optimiser), not rfdetr's COCO weights
    model = getattr(rfdetr, SIZES[args.size])(**({"pretrain_weights": init} if init else {}))
    kw = dict(dataset_dir=ds, epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
              lr=args.lr, output_dir=run_dir)
    if args.resolution:
        kw["resolution"] = args.resolution
    # a run stopped half-way (Ctrl+C, a lost pod) goes on from its last finished epoch: rfdetr writes
    # rf/last.ckpt (weights, optimiser, schedule, epoch) after every epoch; --fresh starts over
    last = os.path.join(run_dir, "last.ckpt")
    if os.path.exists(last):
        if args.fresh:
            shutil.rmtree(run_dir)
        else:
            kw["resume"] = last
            print(f"[rfdetr] continuing a stopped run from {last}", flush=True)
    print(f"[rfdetr] training {SIZES[args.size]} for {args.epochs} epochs -> {run_dir}", flush=True)
    model.train(**kw)
    ckpt = next((os.path.join(run_dir, n) for n in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                                                     "checkpoint_best_regular.pth", "checkpoint.pth")
                 if os.path.exists(os.path.join(run_dir, n))), None)
    if ckpt is None:
        sys.exit(f"no checkpoint in {run_dir}")
    os.makedirs(best, exist_ok=True)
    shutil.copy(ckpt, os.path.join(best, "checkpoint.pth"))
    size = args.resolution or getattr(getattr(model, "model_config", None), "resolution", None)
    meta = {"kind": "rfdetr", "base_model": f"rfdetr-{args.size}", "checkpoint": "checkpoint.pth", "size": size,
            "tiles": bool(args.tiles), "select": args.select, "threshold": 0.5,
            "init": init,
            "extra": [{"name": x, "folder": os.path.abspath(d), "max_images": args.extra_max} for x, d in extra_sources(args)]}
    # the threshold, as detector.py picks it: the best F1 (or F2) on the held-out clips
    _, predict, _ = load(best, meta)
    dets = {}
    for i in range(0, len(dev_frames), 8):
        chunk = dev_frames[i:i + 8]
        imgs = [det_mod.load_image(args.root, f) for f in chunk]
        res = (det_mod.predict_tiled(None, None, imgs, keep=det_mod.KEEP_SCORE, predict_fn=predict)
               if args.tiles else predict(imgs, det_mod.KEEP_SCORE))
        for f, boxes in zip(chunk, res):
            dets[(f["video_id"], f["timestamp"])] = boxes
    thr, m = det_mod.best_threshold(dets, dev_frames, args.select)
    meta.update(threshold=thr, held_out=m, dev_clips=len({f["video_id"] for f in dev_frames}))
    with open(os.path.join(best, "det_train_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(done, "w") as fh:
        fh.write(f"{m[args.select]:.4f}\n")
    print(f"[rfdetr] held-out recall {m['recall']:.1%} precision {m['precision']:.1%} F1 {m['f1']:.3f} "
          f"F2 {m['f2']:.3f} at threshold {thr} -> {best}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("export", "train"))
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--out", default="/workspace/herd/detector_rfdetr_l")
    p.add_argument("--size", choices=tuple(SIZES), default="large")
    p.add_argument("--resolution", type=int, default=None, help="input side (default: the size's own)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tiles", type=int, default=0, help="1: threshold chosen with whole frame + tiles")
    p.add_argument("--select", choices=("f1", "f2"), default="f1")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None,
                   help="fine-tune a detector trained before: its best/ folder (or a .pth) instead of COCO weights")
    p.add_argument("--extra", nargs="*", default=[],
                   help="more boxes: extra_det.py's source folders (e.g. /workspace/extra_det/zenodo)")
    p.add_argument("--extra-max", type=int, default=0, help="at most this many train images per --extra source")
    p.add_argument("--repeat", nargs="*", default=[], help="an --extra set's train images k times: barn=3")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args(argv)
    {"export": cmd_export, "train": cmd_train}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
