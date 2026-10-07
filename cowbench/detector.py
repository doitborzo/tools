#!/usr/bin/env python3
"""The cow detector for the tests: RT-DETRv2 (Apache-2.0), fine-tuned on the
boxes of CBVD-5 train, run on val. The tests ask Muse about the boxes this
finds - never about the annotation's, which only scores.

    train   fine-tune PekingU/rtdetr_v2_r50vd on ava_train (val clips excluded,
            10% of train clips held out to pick the epoch and the threshold)
    detect  every val keyframe -> val_detections.jsonl: boxes with scores
    score   found vs annotated -> det_report.md, det_metrics.json

    python detector.py train  --root /workspace/cbvd5 --out /workspace/lora-runs/detector
    python detector.py detect --root /workspace/cbvd5 --out /workspace/lora-runs/detector
    python detector.py score  --out /workspace/lora-runs/detector

One class, "cow". Boxes are kept down to a low score in the detections file;
the threshold - the one with the best F1 on the held-out clips - is applied
by whoever reads it (frame.detected_cows), so it can change without a rerun.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Only if not there already: herd imports this with cowbench appended to its
# path, and a second copy in front would shadow herd's own report.py.
for _p in (HERE, os.path.join(HERE, "lora")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cbvd  # noqa: E402
import detect as detect_mod  # noqa: E402
import train_lora  # noqa: E402  (load_train, split_dev, read_frame)

MODEL = "PekingU/rtdetr_v2_r50vd"
VAL_MANIFEST = train_lora.VAL_MANIFEST
KEEP_SCORE = 0.05     # lowest score written to the detections file


def frames_of(rows):
    """Per-cow rows -> [{"video_id", "timestamp", "boxes": [[x1, y1, x2, y2], ...]}]."""
    frames = collections.OrderedDict()
    for r in rows:
        frames.setdefault((r["video_id"], r["timestamp"]), []).append(r["bbox"])
    return [{"video_id": v, "timestamp": t, "boxes": b} for (v, t), b in frames.items()]


def load_image(root, f):
    return train_lora.read_frame(root, cbvd.Box(f["video_id"], f["timestamp"], 0, 0, 0, 0, "1", ())
                                 ).convert("RGB")


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def processor_for(name, size):
    from transformers import RTDetrImageProcessor
    return RTDetrImageProcessor.from_pretrained(name, size={"height": size, "width": size})


def predict(model, processor, images, keep=KEEP_SCORE):
    """[[x1, y1, x2, y2, score], ...] per image, boxes normalised, best first.
    RT-DETR needs no NMS: each query is one object."""
    import torch
    pv = processor(images=images, return_tensors="pt")["pixel_values"].to(model.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=pv.is_cuda):
        out = model(pixel_values=pv)
    scores = out.logits.float().sigmoid()[..., 0]
    cx, cy, w, h = out.pred_boxes.float().unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)
    res = []
    for s, b in zip(scores.cpu(), boxes.cpu()):
        idx = (s >= keep).nonzero().flatten().tolist()
        res.append(sorted(([*map(lambda v: round(v, 4), b[i].tolist()), round(float(s[i]), 4)]
                           for i in idx), key=lambda x: -x[4]))
    return res


def tile_boxes(w, h, overlap=0.2):
    """Square tiles across a wide frame, overlapping by at least `overlap`:
    1920x1080 -> x 0-1080 and 840-1920. A square input squeezes a 16:9 frame
    to half its width; a tile keeps a far cow ~1.8x wider. [] for a frame
    that is not wide (nothing to gain)."""
    if w <= 1.2 * h:
        return []
    tw = h
    n = math.ceil((w - tw) / (tw * (1 - overlap))) + 1
    return [(round(x), 0, round(x) + tw, h) for x in np.linspace(0, w - tw, n)]


def nms(boxes, iou=0.5):
    """[[x1, y1, x2, y2, score], ...] -> the same, overlaps above iou removed, best first."""
    out = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        if all(detect_mod.iou(b[:4], o[:4]) < iou for o in out):
            out.append(b)
    return out


def predict_tiled(model, processor, images, keep=KEEP_SCORE, overlap=0.2, batch=12):
    """predict() on each whole frame plus its square tiles, boxes mapped back to
    the frame and merged by NMS. A tile box touching a cut inside the frame is
    dropped (half a cow; the neighbouring tile or the whole frame has her)."""
    jobs = []                                       # (image index, tile or None, PIL image)
    for k, img in enumerate(images):
        jobs.append((k, None, img))
        for t in tile_boxes(*img.size, overlap):
            jobs.append((k, t, img.crop(t)))
    found = [[] for _ in images]
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        for (k, t, _), boxes in zip(chunk, predict(model, processor, [j[2] for j in chunk], keep)):
            if t is None:
                found[k] += boxes
                continue
            w, h = images[k].size
            x0, y0, x1, y1 = t
            tw, th = x1 - x0, y1 - y0
            for b in boxes:
                if (x0 > 0 and b[0] < 0.01) or (x1 < w and b[2] > 0.99):
                    continue
                found[k].append([round((x0 + b[0] * tw) / w, 4), round((y0 + b[1] * th) / h, 4),
                                 round((x0 + b[2] * tw) / w, 4), round((y0 + b[3] * th) / h, 4), b[4]])
    return [nms(f) for f in found]


def prf(dets, frames, threshold, iou=0.5):
    """Recall, precision, F1 of detections (keyed by frame) at a threshold."""
    tp = fp = fn = 0
    for f in frames:
        pred = [b[:4] for b in dets[(f["video_id"], f["timestamp"])] if b[4] >= threshold]
        k = len(detect_mod.match(pred, f["boxes"], iou))
        tp, fp, fn = tp + k, fp + len(pred) - k, fn + len(f["boxes"]) - k
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            # recall weighed twice: in the barn a missed cow loses her time, an extra
            # box only makes a track no cow is given
            "f2": 5 * p * r / (4 * p + r) if p + r else 0.0}


def best_threshold(dets, frames, select="f1"):
    sweep = [round(0.1 + 0.05 * i, 2) for i in range(17)]
    return max(((t, prf(dets, frames, t)) for t in sweep), key=lambda x: (x[1][select], -x[0]))


def run_dets(model, processor, root, frames, batch=8, tiles=False):
    out = {}
    run = predict_tiled if tiles else predict
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        for f, boxes in zip(chunk, run(model, processor, [load_image(root, f) for f in chunk])):
            out[(f["video_id"], f["timestamp"])] = boxes
    return out


# -------------------------------------------------------------------- train

def zoom_crop(img, boxes, rng, min_w=0.45, min_h=0.6, min_visible=0.4):
    """A random part of the frame, so far cows are seen larger in training (and
    tiles - predict_tiled - look like what the model learnt). Boxes are clipped
    to it; a cow with less than min_visible of her box inside is dropped."""
    w, h = img.size
    cw, ch = rng.uniform(min_w, 1.0), rng.uniform(min_h, 1.0)
    x0, y0 = rng.uniform(0, 1 - cw), rng.uniform(0, 1 - ch)
    out = []
    for b in boxes:
        a = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
        c = [max(b[0], x0), max(b[1], y0), min(b[2], x0 + cw), min(b[3], y0 + ch)]
        if c[2] <= c[0] or c[3] <= c[1] or (c[2] - c[0]) * (c[3] - c[1]) / a < min_visible:
            continue
        out.append([(c[0] - x0) / cw, (c[1] - y0) / ch, (c[2] - x0) / cw, (c[3] - y0) / ch])
    img = img.crop((round(x0 * w), round(y0 * h), round((x0 + cw) * w), round((y0 + ch) * h)))
    return img, out


def cmd_train(args):
    import torch
    from transformers import RTDetrV2ForObjectDetection

    best_dir = os.path.join(args.out, "best")
    done = os.path.join(args.out, "train_done")
    # best/ appears after the first epoch, so it alone does not mean finished.
    if os.path.exists(done) and not args.fresh:
        print(f"[det] already trained: {best_dir}")
        return
    os.makedirs(args.out, exist_ok=True)
    rows, info = train_lora.load_train(args.root)
    rows, dev_rows = train_lora.split_dev(rows, args.holdout, args.seed)
    train_frames, dev_frames = frames_of(rows), frames_of(dev_rows)
    print(f"[det] {len(train_frames)} training keyframes ({len(rows)} cows), "
          f"{len(dev_frames)} held-out keyframes from {len({f['video_id'] for f in dev_frames})} clips")

    processor = processor_for(args.model, args.size)
    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.model, num_labels=1, id2label={0: "cow"}, label2id={"cow": 0},
        ignore_mismatched_sizes=True).to(device())
    cuda = device().type == "cuda"

    class Frames(torch.utils.data.Dataset):
        def __len__(self):
            return len(train_frames)

        def __getitem__(self, i):
            from PIL import ImageEnhance
            f = train_frames[i]
            img = load_image(args.root, f)
            boxes = [list(b) for b in f["boxes"]]
            rng = random.Random()
            if rng.random() < 0.5:   # the barns are not symmetric, the cows are
                img = img.transpose(0)   # FLIP_LEFT_RIGHT
                boxes = [[1 - b[2], b[1], 1 - b[0], b[3]] for b in boxes]
            if rng.random() < args.zoom:
                img, boxes = zoom_crop(img, boxes, rng)
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))
            return img, boxes

    def collate(batch):
        imgs, boxes = zip(*batch)
        pv = processor(images=list(imgs), return_tensors="pt")["pixel_values"]
        labels = []
        for bs in boxes:
            t = torch.tensor(bs, dtype=torch.float32).reshape(-1, 4)
            cxcywh = torch.stack([(t[:, 0] + t[:, 2]) / 2, (t[:, 1] + t[:, 3]) / 2,
                                  t[:, 2] - t[:, 0], t[:, 3] - t[:, 1]], -1)
            labels.append({"class_labels": torch.zeros(len(t), dtype=torch.long), "boxes": cxcywh})
        return pv, labels

    loader = torch.utils.data.DataLoader(Frames(), batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate,
                                         drop_last=True, persistent_workers=args.workers > 0)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr * 0.1},
                             {"params": rest, "lr": args.lr}], weight_decay=1e-4)
    total = args.epochs * len(loader)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    last = os.path.join(args.out, "last.pt")
    hist_path = os.path.join(args.out, "det_history.jsonl")
    start, best_f1 = 0, -1.0
    if os.path.exists(last):
        state = torch.load(last, map_location=device(), weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, best_f1 = state["epoch"], state["best_f1"]
        print(f"[det] resuming after epoch {start}")

    t0 = time.time()
    for epoch in range(start, args.epochs):
        model.train()
        losses = []
        for step, (pv, labels) in enumerate(loader):
            labels = [{k: v.to(model.device) for k, v in lab.items()} for lab in labels]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = model(pixel_values=pv.to(model.device), labels=labels).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            losses.append(float(loss))
            if step % 50 == 0:
                print(f"\r  epoch {epoch + 1}/{args.epochs}  step {step}/{len(loader)}  "
                      f"loss {sum(losses[-50:]) / len(losses[-50:]):.3f}", end="", flush=True)
        model.eval()
        dets = run_dets(model, processor, args.root, dev_frames, tiles=args.tiles)
        thr, m = best_threshold(dets, dev_frames, args.select)
        rec = {"epoch": epoch + 1, "loss": sum(losses) / len(losses), "threshold": thr, **m,
               "minutes": round((time.time() - t0) / 60, 1)}
        with open(hist_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"\n[det] epoch {epoch + 1}: held-out recall {m['recall']:.1%} precision "
              f"{m['precision']:.1%} F1 {m['f1']:.3f} F2 {m['f2']:.3f} at threshold {thr}", flush=True)
        if m[args.select] > best_f1:
            best_f1 = m[args.select]
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "det_train_meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"base_model": args.model, "size": args.size, "epoch": epoch + 1,
                           "threshold": thr, "tiles": bool(args.tiles), "zoom": args.zoom, "select": args.select,
                           "held_out": m, "dev_clips": len({f["video_id"] for f in dev_frames}),
                           "train_keyframes": len(train_frames), "train_cows": len(rows)}, fh, indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best_f1": best_f1}, last)
    with open(done, "w") as fh:
        fh.write(f"{best_f1:.4f}\n")
    if os.path.exists(last):
        os.remove(last)
    print(f"[det] done in {(time.time() - t0) / 60:.0f} min, best held-out F1 {best_f1:.3f} -> {best_dir}")


# --------------------------------------------------------------------- live

class Live:
    """The trained detector, in process, one image per call: what stress.py
    runs on every camera frame next to the vLLM server. Calls from many
    camera threads take turns on the one model; a frame is ~10-30 ms on a GPU."""

    def __init__(self, best_dir, threshold=None, tiles=None):
        import threading
        from transformers import RTDetrV2ForObjectDetection
        with open(os.path.join(best_dir, "det_train_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        import torch
        self.processor = processor_for(best_dir, meta["size"])
        model = RTDetrV2ForObjectDetection.from_pretrained(best_dir)
        dev = device()
        try:
            model = model.to(dev)
        except Exception as e:   # OutOfMemoryError, or AcceleratorError on newer torch
            # The GPU is shared with vLLM; with no room left, the camera test
            # still runs - on the CPU, which the report then says.
            if dev.type != "cuda" or "out of memory" not in str(e).lower():
                raise
            print(f"!! no GPU memory left for the detector ({e.__class__.__name__}) - running it on the CPU",
                  flush=True)
            torch.cuda.empty_cache()
            dev = torch.device("cpu")
            model = model.to(dev)
        self.model = model.eval()
        self.threshold = meta["threshold"] if threshold is None else threshold
        # tiles: the whole frame plus square tiles (predict_tiled) - ~3x the work,
        # far cows ~1.8x wider; by default as the detector was chosen with
        self.tiles = bool(meta.get("tiles", False)) if tiles is None else bool(tiles)
        self.name = meta["base_model"]
        self.device = str(dev)
        self._lock = threading.Lock()

    def __call__(self, img):
        """PIL image -> [{"bbox": [x1, y1, x2, y2] (0-1), "det_score"}], above the threshold."""
        return self.many([img])[0]

    def many(self, imgs):
        """Several PIL images in one pass (herd's bursts): a list per image."""
        with self._lock:
            # keep=threshold: nothing below it is used, and NMS over tiles stays small
            res = (predict_tiled if self.tiles else predict)(self.model, self.processor, list(imgs),
                                                              keep=self.threshold)
        return [[{"bbox": b[:4], "det_score": b[4]} for b in boxes if b[4] >= self.threshold] for boxes in res]


# ------------------------------------------------------------------- detect

def cmd_detect(args):
    import torch
    from transformers import RTDetrV2ForObjectDetection

    best_dir = os.path.join(args.out, "best")
    with open(os.path.join(best_dir, "det_train_meta.json"), encoding="utf-8") as fh:
        tmeta = json.load(fh)
    path = os.path.join(args.out, "val_detections.jsonl")
    if os.path.exists(path) and not args.fresh:
        print(f"[det] already detected: {path}")
        return
    processor = processor_for(best_dir, tmeta["size"])
    model = RTDetrV2ForObjectDetection.from_pretrained(best_dir).to(device()).eval()
    frames = frames_of(train_lora.read_jsonl(args.manifest))
    dets = run_dets(model, processor, args.root, frames, tiles=args.tiles)
    # Under a temporary name until det_meta.json (the threshold) is written
    # too: a crash in between must not leave a file that reads as finished.
    with open(path + ".part", "w", encoding="utf-8") as fh:
        for f in frames:
            fh.write(json.dumps({"video_id": f["video_id"], "timestamp": f["timestamp"],
                                 "boxes": dets[(f["video_id"], f["timestamp"])]}) + "\n")

    # Speed, one frame at a time as a camera delivers them; image decode excluded.
    imgs = [load_image(args.root, f) for f in frames[:40]]
    for img in imgs[:5]:
        predict(model, processor, [img])
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    t = time.perf_counter()
    for img in imgs[5:]:
        predict(model, processor, [img])
    sync()
    per_frame = (time.perf_counter() - t) / len(imgs[5:])
    meta = {"model": best_dir, "base_model": tmeta["base_model"], "size": tmeta["size"],
            "threshold": tmeta["threshold"], "keep_score": KEEP_SCORE,
            "manifest": args.manifest, "keyframes": len(frames),
            "ms_per_frame": round(per_frame * 1000, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    with open(os.path.join(args.out, "det_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(path + ".part", path)
    print(f"[det] {len(frames)} keyframes -> {path}  ({meta['ms_per_frame']} ms a frame, "
          f"threshold {meta['threshold']})")


# -------------------------------------------------------------------- score

def cmd_score(args):
    with open(os.path.join(args.out, "det_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    dets = {}
    with open(os.path.join(args.out, "val_detections.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            dets[(r["video_id"], r["timestamp"])] = r["boxes"]
    frames = frames_of(train_lora.read_jsonl(args.manifest))
    m = {"threshold": thr, "iou0.5": prf(dets, frames, thr, 0.5), "iou0.3": prf(dets, frames, thr, 0.3)}
    diffs = [sum(b[4] >= thr for b in dets[(f["video_id"], f["timestamp"])]) - len(f["boxes"]) for f in frames]
    m["count"] = {"annotated_mean": sum(len(f["boxes"]) for f in frames) / len(frames),
                  "found_mean": sum(len(f["boxes"]) + d for f, d in zip(frames, diffs)) / len(frames),
                  "mean_abs_error": sum(map(abs, diffs)) / len(diffs),
                  "exact_share": sum(d == 0 for d in diffs) / len(diffs)}
    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for f in frames for b in f["boxes"])
    cuts = [areas[len(areas) * q // 4] for q in (1, 2, 3)]
    by_size = collections.defaultdict(lambda: [0, 0])
    for f in frames:
        pred = [b[:4] for b in dets[(f["video_id"], f["timestamp"])] if b[4] >= thr]
        hit = {j for _, j, _ in detect_mod.match(pred, f["boxes"], 0.5)}
        for j, b in enumerate(f["boxes"]):
            q = sum((b[2] - b[0]) * (b[3] - b[1]) >= c for c in cuts)
            by_size[q][0] += 1
            by_size[q][1] += j in hit
    names = ["smallest", "smaller", "larger", "largest"]
    m["recall_by_size"] = {names[q]: n and k / n for q, (n, k) in sorted(by_size.items())}
    m["ms_per_frame"] = meta.get("ms_per_frame")
    with open(os.path.join(args.out, "det_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    pct = lambda x: f"{x:.1%}"
    lines = [f"# Cow detector: {meta.get('base_model')} fine-tuned on CBVD-5 train", "",
             f"{len(frames)} val keyframes, {sum(len(f['boxes']) for f in frames)} annotated cows. "
             f"Input {meta.get('size')} px, score threshold {thr} (best F1 on held-out train clips). "
             f"{meta.get('ms_per_frame')} ms a frame on {meta.get('gpu')}.", "",
             "| IoU needed | Found | Extra | Missed | Recall | Precision | F1 |", "|---|---|---|---|---|---|---|"]
    for k in ("iou0.5", "iou0.3"):
        d = m[k]
        lines.append(f"| {k[3:]} | {d['tp']} | {d['fp']} | {d['fn']} | {pct(d['recall'])} | "
                     f"{pct(d['precision'])} | {pct(d['f1'])} |")
    c = m["count"]
    lines += ["", f"Per keyframe {c['annotated_mean']:.1f} annotated, {c['found_mean']:.1f} found, "
                  f"off by {c['mean_abs_error']:.2f}; exact count on {pct(c['exact_share'])} of keyframes.", "",
              "Recall by cow size (IoU 0.5): " + ", ".join(f"{k} {pct(v)}" for k, v in m["recall_by_size"].items()),
              "", "Extras are a lower bound on false detections: a cow the annotators skipped counts as one."]
    with open(os.path.join(args.out, "det_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    d = m["iou0.5"]
    print(f"[det] recall {d['recall']:.1%} precision {d['precision']:.1%} at IoU 0.5, threshold {thr}; "
          f"{m['ms_per_frame']} ms a frame -> {os.path.join(args.out, 'det_report.md')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("train", "detect", "score"))
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--out", default="/workspace/lora-runs/detector")
    p.add_argument("--manifest", default=VAL_MANIFEST, help="keyframes to detect on / score against")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--size", type=int, default=960, help="square input side; 640 is RT-DETR's own, "
                                                         "960 keeps far cows a few more pixels")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threshold", type=float, default=None, help="score: override the chosen threshold")
    p.add_argument("--zoom", type=float, default=0.0,
                   help="train: share of images cut to a random part of the frame (far cows seen larger); 0.5 is a good start")
    p.add_argument("--tiles", type=int, default=0,
                   help="1: detect on the whole frame and its square tiles, merged (far cows ~1.8x wider, ~3x the work)")
    p.add_argument("--select", choices=("f1", "f2"), default="f1",
                   help="train: the epoch and threshold by F1, or by F2 (recall weighed twice: a missed cow costs more)")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args(argv)
    {"train": cmd_train, "detect": cmd_detect, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
