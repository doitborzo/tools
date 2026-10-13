#!/usr/bin/env python3
"""Rumination from video: DINOv2 embeddings of every frame -> a temporal transformer.

CBVD-5 is not used. The model learns on two public sets with rumination labelled per cow:

    kaggle  "Beef Cattle Behavior Data Set" (kaggle.com/datasets/lucyfirst/beef-cattle-behavior-data-set,
            licence not stated - research use, cite the authors): 6 beef cows in a barn, one surveillance
            camera, day and night. 4,337 clips of one cow each, 10 s, 25 fps, 224 x 224: ruminate 1,544,
            lie 1,362, stand 638, eat 546, drink 247 (one label per clip). Only these clips are read from
            the 48.5 GB zip (~6.4 GB).
    cvb     CSIRO "CVB: A Video Dataset of Cattle Visual Behaviors" (data.csiro.au/collection/csiro:58916,
            CC BY-NC-SA 4.0): beef cattle on pasture, 502 clips of 15 s at 30 fps, 1920 x 1080, every cow
            boxed and tracked in every frame with one of 11 behaviours (ruminating-standing,
            ruminating-lying, grazing, resting-..., drinking, walking, ...). 13.8 GB of frames.

One label space for both (CLASSES): rumination, feeding, drinking, resting, moving, other; and
posture (standing / lying) where the label says it.

    python herd/rumination.py fetch   --data /workspace/rum --sources kaggle,cvb
    python herd/rumination.py extract --data /workspace/rum --out /workspace/rum/feat     (GPU)
    python herd/rumination.py train   --feat /workspace/rum/feat --out /workspace/rum/model [--motion 1]

extract: every frame of a cow's crop (224, the barn's square crop with a 0.1 margin; the Kaggle
clips are such crops already) -> DINOv2-S (CLS + a 2 x 2 pool of the patch tokens, as herd's
frame encoder) -> one vector per frame, in bf16 on the GPU, kept in fp16. Each cow's sequence is
cut into 7 s windows (every 1.5 s); per window its label (the majority of its frames, which must
agree on >= 60%) and, from the crops, the motion rhythm (motion.py) - optional input (--motion 1).

train: the transformer (model.TemporalModel: frame vectors + real time) -> rumination (yes / no)
and the behaviour class, posture on the side. Validation never shares a recording with training:
kaggle holds out recording 00000000514000000 and every 5th short sequence; cvb keeps its authors'
train / val lists. Written: <out>/rumination.pt (the model, its threshold and the encoder it needs),
<out>/eval.json (per source: rumination precision / recall / F1 / AUC, behaviour accuracy).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import glob
import io
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cowbench"))

CLASSES = ["rumination", "feeding", "drinking", "resting", "moving", "other"]
STANDING, LYING = 0, 1
# label -> (class or None: the frame does not count, posture or -1)
KAGGLE_MAP = {"ruminate": ("rumination", -1), "eat": ("feeding", STANDING), "drink": ("drinking", STANDING),
              "lie": ("resting", LYING), "stand": ("resting", STANDING)}
CVB_MAP = {"none": ("other", -1), "grazing": ("feeding", STANDING), "walking": ("moving", STANDING),
           "running": ("moving", STANDING), "ruminating-standing": ("rumination", STANDING),
           "ruminating-lying": ("rumination", LYING), "resting-standing": ("resting", STANDING),
           "resting-lying": ("resting", LYING), "drinking": ("drinking", STANDING), "grooming": ("other", -1),
           "other": ("other", -1), "hidden": (None, -1)}
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/lucyfirst/beef-cattle-behavior-data-set"
CVB_LIST = "https://data.csiro.au/dap/ws/v2/collections/58916/data"
KAGGLE_VAL_RECORDING = "00000000514000000"
WINDOW_S, STRIDE_S, MIN_AGREE, MIN_SEEN = 7.0, 1.5, 0.6, 0.6
UA = {"User-Agent": "herd-rumination/1.0"}


def label_of(src, name):
    """A source's behaviour name -> (class index or -1, posture or -1)."""
    cls, post = (KAGGLE_MAP if src == "kaggle" else CVB_MAP).get(name, ("other", -1))
    return (CLASSES.index(cls) if cls else -1), post


# ------------------------------------------------------------------ fetch

def _get(url, path=None, tries=8, timeout=300, fallback=None):
    """GET with retries (CSIRO's store drops connections now and then); to a file if path.
    fallback: a second URL for the later tries (the DAP link that redirects to a fresh signed one)."""
    for k in range(tries):
        try:
            u = fallback if (fallback and k % 2 == 1) else url
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=timeout) as r:
                data = r.read()
            if path is None:
                return data
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)
            return path
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            if k == tries - 1:
                raise
            time.sleep(min(60, 2 ** k))


def fetch_kaggle(dest, workers=8, limit=0):
    """Only Category Videos/cows/<class>/*.mp4 from the 48.5 GB zip, read in place by HTTP Range."""
    from extra_det import RemoteZip
    out = os.path.join(dest, "kaggle")
    done = os.path.join(out, "done.json")
    if os.path.exists(done):
        print(f"[fetch] kaggle: done already ({out})")
        return
    print("[fetch] kaggle: reading the zip's index (1.15M entries, a few minutes)", flush=True)
    rz = RemoteZip(KAGGLE_URL)
    names = [n for n in rz.names() if n.startswith("Category Videos/cows/") and n.endswith(".mp4")]
    if limit:
        names = names[::max(1, len(names) // limit)][:limit]
    print(f"[fetch] kaggle: {len(names)} clips", flush=True)

    def one(n):
        cls = n.split("/")[2]
        path = os.path.join(out, cls, os.path.basename(n))
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as fh:
                fh.write(rz.read(n))
            os.replace(path + ".part", path)

    with cf.ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, names), 1):
            if k % 50 == 0 or k == len(names):
                print(f"\r  {k}/{len(names)} clips", end="", flush=True)
    print(flush=True)
    with open(done, "w") as fh:
        json.dump({"clips": len(names), "url": KAGGLE_URL}, fh)


def fetch_cvb(dest, workers=48, limit=0, step=1, split=None):
    """CVB's annotations, its train / val lists and the frames (every step-th) of every clip.
    225k single JPEGs: the store's signed links (valid 48 h) are used directly, many at a time."""
    out = os.path.join(dest, "cvb")
    os.makedirs(out, exist_ok=True)
    lst = os.path.join(out, "_files.json")
    if not os.path.exists(lst) or time.time() - os.path.getmtime(lst) > 40 * 3600:   # signed links age
        print("[fetch] cvb: the file list (~0.8 GB of JSON, ~1 min)", flush=True)
        d = json.loads(_get(CVB_LIST, timeout=900))
        files = [{"name": f["filename"], "url": f["link"]["href"],
                  "signed": (f.get("presignedLink") or {}).get("href"), "size": f.get("fileSize", 0)}
                 for f in d["file"]]
        with open(lst, "w") as fh:
            json.dump(files, fh)
    with open(lst) as fh:
        files = json.load(fh)
    clips = sorted({f["name"].split("/")[1] for f in files if f["name"].startswith("raw_frames/")})
    if split:                                        # e.g. val only, for the whole-system test
        lists = [f for f in files if f["name"].endswith(f"{split}_set_tasklist.csv")]
        if lists:
            p = os.path.join(out, lists[0]["name"])
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                _get(lists[0]["url"], p)
            want = {c.strip() for c in open(p).read().replace("\n", ",").split(",") if c.strip()}
            clips = [c for c in clips if c in want]
    if limit:
        clips = clips[::max(1, len(clips) // limit)][:limit]
    keep = set(clips)
    meta = [f for f in files if (f["name"].startswith("cvb_in_ava_format/") and f["name"].endswith((".csv", ".pbtx"))
                                 and "_set_tasklist" in f["name"] or f["name"].endswith(".pbtx"))
            or (f["name"].startswith("annotations/") and f["name"].endswith(".json") and f["name"].split("/")[1] in keep)]
    frames = []
    for f in files:
        m = re.match(r"raw_frames/([^/]+)/img_(\d+)\.jpg$", f["name"])
        if m and m.group(1) in clips and (int(m.group(2)) - 1) % step == 0:
            frames.append(f)
    print(f"[fetch] cvb: {len(clips)} clips, {len(meta)} annotation files, {len(frames)} frames "
          f"({sum(f['size'] for f in meta + frames) / 2**30:.1f} GB)", flush=True)
    frames = meta + frames                              # the annotations first, all in parallel

    def one(f):
        p = os.path.join(out, f["name"])
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            _get(f.get("signed") or f["url"], p, fallback=f["url"])

    with cf.ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, frames), 1):
            if k % 500 == 0 or k == len(frames):
                print(f"\r  {k}/{len(frames)} frames", end="", flush=True)
    print(flush=True)


def cmd_fetch(args):
    for s in args.sources.split(","):
        if s == "kaggle":
            fetch_kaggle(args.data, limit=args.limit)
        elif s == "cvb":
            fetch_cvb(args.data, limit=args.limit, step=args.cvb_step, split=args.cvb_split)
        else:
            sys.exit(f"unknown source {s}")


# ---------------------------------------------------------------- extract

def windows(times, labels, postures, seen=None, length=WINDOW_S, stride=STRIDE_S):
    """A cow's sequence -> 7 s windows [(t0, t1, class, posture, agreement)] whose frames agree.

    times (T,) s, labels (T,) class or -1 (not counted), postures (T,) or -1, seen (T,) bool."""
    times, labels, postures = np.asarray(times), np.asarray(labels), np.asarray(postures)
    seen = np.ones(len(times), bool) if seen is None else np.asarray(seen, bool)
    if len(times) == 0:
        return []
    out, t0, end = [], float(times[0]), float(times[-1])
    if end - t0 < length:                                # a short sequence: one window over all of it
        starts = [t0]
        length = end - t0 + 1e-6
    else:
        starts = list(np.arange(t0, end - length + 1e-6, stride))
        if end - length - starts[-1] > 0.3 * stride:    # the end of the sequence gets a window too
            starts.append(end - length)
    for s in starts:
        m = (times >= s) & (times < s + length)
        if m.sum() < 2:
            continue
        lab = labels[m & seen & (labels >= 0)]
        if len(lab) < MIN_SEEN * m.sum():
            continue
        c, n = collections.Counter(lab.tolist()).most_common(1)[0]
        if n < MIN_AGREE * len(lab):
            continue
        post = postures[m & seen & (labels == c) & (postures >= 0)]
        p = collections.Counter(post.tolist()).most_common(1)[0][0] if len(post) else -1
        out.append((float(s), float(s + length), int(c), int(p), round(n / len(lab), 3)))
    return out


def kaggle_split(name):
    """A clip's group (recording or short sequence) and whether it is held out."""
    rec = os.path.basename(name).split("_")[0]
    short = rec.isdigit() and len(rec) < 6             # the short drink / eat sequences: 1..200
    val = rec == KAGGLE_VAL_RECORDING or (short and int(rec) % 5 == 0)
    return rec, ("val" if val else "train")


def _save_seq(path, emb, times, labels, postures, seen, wins, motion, meta):
    np.savez(path + ".part.npz", emb=emb.astype(np.float16), t=np.asarray(times, np.float32),
             label=np.asarray(labels, np.int8), posture=np.asarray(postures, np.int8),
             seen=np.asarray(seen, bool), windows=np.asarray([w[:2] for w in wins], np.float32).reshape(-1, 2),
             win_label=np.asarray([w[2:4] for w in wins], np.int8).reshape(-1, 2),
             motion=np.asarray(motion, np.float32).reshape(len(wins), _motion_dim()), meta=json.dumps(meta))
    os.replace(path + ".part.npz", path)


def _motion_dim():
    import motion as motion_mod
    return motion_mod.DIM


def _motion(crops, times, seen, wins):
    import motion as motion_mod
    out = []
    for t0, t1, *_ in wins:
        m = (times >= t0) & (times < t1)
        f, ok = motion_mod.motion_features(crops[m], times[m], seen[m])
        out.append(f if ok else np.zeros(motion_mod.DIM, np.float32))
    return np.stack(out) if out else np.zeros((0, motion_mod.DIM), np.float32)


def extract_kaggle(data, out, enc, args):
    from cbvd_bursts import encode, read_video
    from PIL import Image
    clips = sorted(glob.glob(os.path.join(data, "kaggle", "*", "*.mp4")))
    os.makedirs(os.path.join(out, "kaggle"), exist_ok=True)
    print(f"[extract] kaggle: {len(clips)} clips", flush=True)
    for k, path in enumerate(clips, 1):
        cls = os.path.basename(os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(out, "kaggle", f"{cls}__{name}.npz")
        if os.path.exists(dst):
            continue
        frames, fps = read_video(path)
        if len(frames) < 2:
            continue
        crops = np.stack([f if f.shape[:2] == (args.crop, args.crop)
                          else np.asarray(Image.fromarray(f).resize((args.crop, args.crop), Image.BILINEAR))
                          for f in frames])
        times = np.arange(len(crops), dtype=np.float32) / fps
        c, p = label_of("kaggle", cls)
        labels, postures, seen = np.full(len(crops), c), np.full(len(crops), p), np.ones(len(crops), bool)
        wins = windows(times, labels, postures, seen)
        group, split = kaggle_split(name)
        _save_seq(dst, encode(enc, crops), times, labels, postures, seen, wins, _motion(crops, times, seen, wins),
                  {"source": "kaggle", "seq": name, "group": group, "split": split, "fps": fps, "label": cls})
        if k % 25 == 0 or k == len(clips):
            print(f"\r  {k}/{len(clips)} clips", end="", flush=True)
    print(flush=True)


def cvb_split(data):
    """CVB's own lists -> {clip: 'train' | 'val'}."""
    out = {}
    for split in ("train", "val"):
        p = os.path.join(data, "cvb", "cvb_in_ava_format", f"{split}_set_tasklist.csv")
        if os.path.exists(p):
            for c in open(p).read().replace("\n", ",").split(","):
                if c.strip():
                    out[c.strip()] = split
    return out


def cvb_tracks(ann):
    """A clip's COCO annotation -> {track: [(frame number, box 0-1, behaviour)]}."""
    ims = {im["id"]: im for im in ann["images"]}
    tracks = collections.defaultdict(list)
    for a in ann["annotations"]:
        im = ims[a["image_id"]]
        m = re.search(r"img_(\d+)\.jpg$", im["file_name"])
        if not m:
            continue
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        at = a.get("attributes", {})
        tracks[at.get("track_id", 0)].append((int(m.group(1)), [x / W, y / H, (x + w) / W, (y + h) / H],
                                              at.get("behavior", "other")))
    return {k: sorted(v) for k, v in tracks.items()}


def extract_cvb(data, out, enc, args):
    from cbvd_bursts import encode
    from common import crop
    from PIL import Image
    splits = cvb_split(data)
    anns = sorted(glob.glob(os.path.join(data, "cvb", "annotations", "*", "annotations", "instances_default.json")))
    os.makedirs(os.path.join(out, "cvb"), exist_ok=True)
    print(f"[extract] cvb: {len(anns)} clips", flush=True)
    pool = cf.ThreadPoolExecutor(8)
    for k, ap in enumerate(anns, 1):
        clip = ap.split(os.sep)[-3]
        fdir = os.path.join(data, "cvb", "raw_frames", clip)
        if not os.path.isdir(fdir):
            continue
        with open(ap) as fh:
            tracks = cvb_tracks(json.load(fh))
        todo = {tid: tr for tid, tr in tracks.items()
                if not os.path.exists(os.path.join(out, "cvb", f"{clip}__t{tid}.npz"))}
        if not todo:
            continue
        have = {int(re.search(r"img_(\d+)", f).group(1)): os.path.join(fdir, f) for f in os.listdir(fdir)
                if f.endswith(".jpg")}
        by_frame = collections.defaultdict(list)                   # frame -> [(track, box)]
        for tid, tr in todo.items():
            for n, b, _ in tr:
                if n in have:
                    by_frame[n].append((tid, b))

        def crops_of_frame(n):
            # a 1080p frame is read, its cows cut out and the frame let go: memory is the crops only
            img = np.asarray(Image.open(have[n]).convert("RGB"))
            return n, {tid: crop(img, b, args.crop, args.margin) for tid, b in by_frame[n]}

        cut = collections.defaultdict(dict)                        # track -> frame -> crop
        for n, got in pool.map(crops_of_frame, sorted(by_frame)):
            for tid, c in got.items():
                cut[tid][n] = c
        group = "_".join(clip.split("_")[1:5])                     # the camera and the recording
        for tid, tr in todo.items():
            tr = [x for x in tr if x[0] in cut[tid]]
            if len(tr) < 2:
                continue
            crops = np.stack([cut[tid][n] for n, _, _ in tr])
            times = np.array([(n - 1) / 30.0 for n, _, _ in tr], np.float32)
            lp = [label_of("cvb", b) for _, _, b in tr]
            labels = np.array([c for c, _ in lp])
            postures = np.array([p for _, p in lp])
            seen = labels >= 0
            wins = windows(times, labels, postures, seen)
            _save_seq(os.path.join(out, "cvb", f"{clip}__t{tid}.npz"), encode(enc, crops), times, labels, postures,
                      seen, wins, _motion(crops, times, seen, wins),
                      {"source": "cvb", "seq": f"{clip}__t{tid}", "group": group,
                       "split": splits.get(clip, "train"), "fps": 30.0})
        if args.cvb_clean:
            for f in os.listdir(fdir):
                os.remove(os.path.join(fdir, f))
        print(f"\r  {k}/{len(anns)} clips", end="", flush=True)
    print(flush=True)


def cmd_extract(args):
    import torch
    from model import FrameEncoder
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = FrameEncoder(args.encoder, args.grid).to(dev).eval()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "features.json"), "w") as fh:
        json.dump({"encoder": args.encoder, "grid": args.grid, "crop": args.crop, "margin": args.margin,
                   "dim": enc.out_dim, "classes": CLASSES, "window_s": WINDOW_S, "stride_s": STRIDE_S}, fh)
    for s in args.sources.split(","):
        (extract_kaggle if s == "kaggle" else extract_cvb)(args.data, args.out, enc, args)


# ------------------------------------------------------------------ train

class Windows:
    """Every window of every sequence of a split, the sequences in memory (fp16)."""

    def __init__(self, feat, split, sources):
        self.seqs, self.rows = [], []
        for s in sources:
            for p in sorted(glob.glob(os.path.join(feat, s, "*.npz"))):
                z = np.load(p)
                meta = json.loads(str(z["meta"]))
                if meta["split"] != split or not len(z["windows"]):
                    continue
                k = len(self.seqs)
                self.seqs.append({"emb": z["emb"], "t": z["t"], "seen": z["seen"], "meta": meta})
                for w, (t0, t1), (c, p_) in zip(range(len(z["windows"])), z["windows"], z["win_label"]):
                    self.rows.append({"seq": k, "t0": float(t0), "t1": float(t1), "label": int(c),
                                      "posture": int(p_), "motion": z["motion"][w], "source": s})
        self.dim = self.seqs[0]["emb"].shape[1] if self.seqs else 0

    def counts(self):
        return {s: dict(collections.Counter(CLASSES[r["label"]] for r in self.rows if r["source"] == s))
                for s in sorted({r["source"] for r in self.rows})}


def batch(ds, ids, rng, train, max_frames=175, use_motion=False):
    """Windows -> padded (x, t, valid[, motion]) tensors; in training frames drop and a stretch hides.
    Real times are kept (25 or 30 fps): the chewing rhythm must not be stretched."""
    import torch
    xs, ts, vs = [], [], []
    for i in ids:
        r = ds.rows[i]
        s = ds.seqs[r["seq"]]
        idx = np.flatnonzero((s["t"] >= r["t0"]) & (s["t"] < r["t1"]))
        if len(idx) > max_frames:
            idx = idx[np.round(np.linspace(0, len(idx) - 1, max_frames)).astype(int)]
        v = s["seen"][idx].copy()
        if train:
            v &= rng.random(len(idx)) > 0.15
            if rng.random() < 0.3 and len(idx) > 8:
                b = rng.integers(len(idx) // 8, len(idx) // 3)
                st = rng.integers(0, len(idx) - b)
                v[st:st + b] = False
            if not v.any():
                v[rng.integers(len(idx))] = True
        xs.append(s["emb"][idx].astype(np.float32))
        ts.append(s["t"][idx] - s["t"][idx[0]])
        vs.append(v)
    n = max(len(x) for x in xs)
    pad = lambda a, f: np.concatenate([a, np.full((n - len(a),) + a.shape[1:], f, a.dtype)])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = [torch.from_numpy(np.stack([pad(x, 0) for x in xs])).to(dev),
           torch.from_numpy(np.stack([pad(t, 0) for t in ts])).to(dev),
           torch.from_numpy(np.stack([pad(v, False) for v in vs])).to(dev)]
    if use_motion:
        m = np.stack([ds.rows[i]["motion"] for i in ids]).astype(np.float32)
        if train:
            m[rng.random(len(ids)) < 0.1] = 0
        out.append(torch.from_numpy(m).to(dev))
    else:
        out.append(None)
    return out


def balanced_sampler(ds, rng, n):
    """Windows drawn so that each source counts the same and rumination is half of each."""
    by = collections.defaultdict(list)
    for i, r in enumerate(ds.rows):
        by[(r["source"], r["label"] == 0)].append(i)
    keys = [k for k in by if by[k]]
    return [by[keys[j]][rng.integers(len(by[keys[j]]))] for j in rng.integers(len(keys), size=n)]


def predict(model, ds, use_motion, bs=64):
    import torch
    model.eval()
    p_rum, p_cls = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i in range(0, len(ds.rows), bs):
            ids = list(range(i, min(i + bs, len(ds.rows))))
            x, t, v, m = batch(ds, ids, rng, False, use_motion=use_motion)
            o = model(x, t, v, motion=m)
            p_rum.append(torch.sigmoid(o["rumination"]).float().cpu().numpy())
            p_cls.append(o["activity"].softmax(-1).float().cpu().numpy())
    return np.concatenate(p_rum), np.concatenate(p_cls)


def scores(p, y, thr):
    """Rumination precision / recall / F1 at thr, its AUC, and the F1-best threshold."""
    def at(t):
        pred = p >= t
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "tp": tp,
                "fp": fp, "fn": fn}
    npos, nneg = int(y.sum()), int((~y).sum())
    auc = None
    if npos and nneg:                                       # Mann-Whitney: P(a ruminating window scores higher)
        ranks = np.empty(len(p))
        ranks[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
        auc = float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    best = max((at(t) | {"threshold": round(float(t), 2)} for t in np.arange(0.05, 0.96, 0.05)),
               key=lambda d: d["f1"])
    return {"n": int(len(y)), "ruminating": int(y.sum()), "auc": auc, "at_threshold": at(thr) | {"threshold": thr},
            "best": best}


def evaluate(model, ds, use_motion, thr):
    p, pc = predict(model, ds, use_motion)
    y = np.array([r["label"] == 0 for r in ds.rows])
    lab = np.array([r["label"] for r in ds.rows])
    src = np.array([r["source"] for r in ds.rows])
    out = {}
    for s in sorted(set(src)) + ["all"]:
        m = np.ones(len(y), bool) if s == "all" else src == s
        out[s] = scores(p[m], y[m], thr)
        out[s]["behaviour_accuracy"] = float((pc[m].argmax(1) == lab[m]).mean())
    return out, p


def cmd_train(args):
    import torch
    import motion as motion_mod
    from model import TemporalModel, masked_ce
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sources = args.sources.split(",")
    tr, va = Windows(args.feat, "train", sources), Windows(args.feat, "val", sources)
    if not tr.rows or not va.rows:
        sys.exit(f"no windows to train / validate on under {args.feat} (extract first)")
    print(f"[train] train windows {len(tr.rows)} {tr.counts()}\n        val windows {len(va.rows)} {va.counts()}",
          flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = dict(d=args.d, layers=args.layers, heads=4, dropout=0.1, n_posture=2, n_activity=len(CLASSES),
              motion_dim=motion_mod.DIM if args.motion else 0)
    model = TemporalModel(tr.dim, **kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    os.makedirs(args.out, exist_ok=True)
    best, log = None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for _ in range(args.steps):
            ids = balanced_sampler(tr, rng, args.batch)
            x, t, v, m = batch(tr, ids, rng, True, use_motion=bool(args.motion))
            y = torch.tensor([tr.rows[i]["label"] for i in ids], device=dev)
            pst = torch.tensor([tr.rows[i]["posture"] for i in ids], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(x, t, v, motion=m)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(o["rumination"].float(), (y == 0).float())
                    + masked_ce(o["activity"].float(), y) + 0.3 * masked_ce(o["posture"].float(), pst))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        ev, _ = evaluate(model, va, bool(args.motion), 0.5)
        f1s = [ev[s]["best"]["f1"] for s in ev if s != "all"]
        score = float(np.mean(f1s))                       # each source counts the same
        log.append({"epoch": ep, "loss": tot / args.steps, "val_f1_mean": score,
                    **{s: {"f1": ev[s]["best"]["f1"], "auc": ev[s]["auc"]} for s in ev}})
        print(f"  epoch {ep:3d}  loss {tot / args.steps:.3f}  val rumination F1 (best thr) "
              + "  ".join(f"{s} {ev[s]['best']['f1']:.3f} auc {ev[s]['auc'] or 0:.3f}" for s in ev), flush=True)
        if best is None or score > best[0]:
            best = (score, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    ev, p = evaluate(model, va, bool(args.motion), 0.5)
    thr = ev["all"]["best"]["threshold"]                    # one threshold for all: the barn has no source
    ev, p = evaluate(model, va, bool(args.motion), thr)
    with open(os.path.join(args.feat, "features.json")) as fh:
        feats = json.load(fh)
    torch.save({"kind": "rumination", "kw": kw, "in_dim": tr.dim, "state": model.state_dict(), "classes": CLASSES,
                "threshold": thr, "motion": bool(args.motion), "features": feats, "epoch": best[1],
                "window_s": WINDOW_S}, os.path.join(args.out, "rumination.pt"))
    res = {"epoch": best[1], "threshold": thr, "motion": bool(args.motion), "sources": sources,
           "train_windows": tr.counts(), "val_windows": va.counts(), "val": ev, "log": log}
    with open(os.path.join(args.out, "eval.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[train] best epoch {best[1]}, threshold {thr} -> {args.out}/rumination.pt")
    print(f"{'source':8s} {'windows':>7s} {'rum':>5s}  {'precision':>9s} {'recall':>7s} {'F1':>6s} {'AUC':>6s}  behaviour acc")
    for s, e in ev.items():
        a = e["at_threshold"]
        print(f"{s:8s} {e['n']:7d} {e['ruminating']:5d}  {a['precision']:9.1%} {a['recall']:7.1%} {a['f1']:6.3f} "
              f"{(e['auc'] or 0):6.3f}  {e['behaviour_accuracy']:.1%}")
    return 0


# -------------------------------------------------------- the whole system

def load_rumination(path, dev):
    import torch
    from model import TemporalModel
    rk = torch.load(path, map_location=dev, weights_only=False)
    m = TemporalModel(rk["in_dim"], **rk["kw"]).to(dev).eval()
    m.load_state_dict(rk["state"])
    return m, rk


def _box_at(track, n):
    """A GT track [(frame, box, behaviour)] -> its box at frame n (the nearest annotated frame)."""
    return min(track, key=lambda x: abs(x[0] - n))[1]


def cmd_e2e(args):
    """CVB's val clips through the barn's own chain: raw frames -> detector (burst_detect_fps a second)
    -> boxes linked into chains (pipeline.link_chains) -> boxes for every frame -> crops -> DINOv2
    -> the rumination transformer on the 7 s burst. Each annotated cow of a burst (its label as in
    training) is matched to a chain (IoU >= 0.3 at mid-burst). Scored three ways:
      oracle   the same model on the annotation's boxes - the model alone
      found    the cows the detector found, on its boxes
      system   every annotated cow: one the detector missed counts as "not ruminating"; a chain on no
               annotated cow that says "ruminating" is a false alarm"""
    import torch
    import detect as detect_mod
    import detector as det_mod
    from cbvd_bursts import encode
    from common import DEFAULTS, crop, interpolate_box
    from model import FrameEncoder
    from motion import motion_features
    from PIL import Image
    from pipeline import link_chains
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, rk = load_rumination(args.model, dev)
    feats_meta, thr = rk["features"], (args.rum_threshold if args.rum_threshold is not None else rk["threshold"])
    enc = FrameEncoder(feats_meta["encoder"], feats_meta["grid"]).to(dev).eval()
    det = det_mod.Live(args.detector, args.threshold, tiles=args.tiles)
    det_fps, link_iou = DEFAULTS["sampling"]["burst_detect_fps"], DEFAULTS["tracker"]["iou"]
    size, margin = feats_meta["crop"], feats_meta["margin"]

    def rumination_p(crops, times, valid):
        x = torch.from_numpy(encode(enc, crops).astype(np.float32)).to(dev).unsqueeze(0)
        t = torch.from_numpy(np.asarray(times, np.float32) - times[0]).to(dev).unsqueeze(0)
        v = torch.from_numpy(np.asarray(valid, bool)).to(dev).unsqueeze(0)
        mot = None
        if model.motion_dim:
            m, ok = motion_features(np.stack(crops), np.asarray(times) - times[0], valid)
            mot = torch.from_numpy(m if ok else np.zeros_like(m)).to(dev).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(model(x, t, v, motion=mot)["rumination"])[0])

    splits = cvb_split(args.data)
    anns = sorted(glob.glob(os.path.join(args.data, "cvb", "annotations", "*", "annotations", "instances_default.json")))
    anns = [a for a in anns if splits.get(a.split(os.sep)[-3]) == "val"
            and os.path.isdir(os.path.join(args.data, "cvb", "raw_frames", a.split(os.sep)[-3]))]
    if args.limit:
        anns = anns[:args.limit]
    if not anns:
        sys.exit("no CVB val clip with frames: rumination.py fetch --sources cvb --cvb-split val")
    print(f"[e2e] {len(anns)} CVB val clips; detector {det.name} (threshold {det.threshold}, tiles {det.tiles}); "
          f"rumination threshold {thr}", flush=True)
    rows, extras, det_ms = [], [], []
    for k, ap in enumerate(anns, 1):
        clip = ap.split(os.sep)[-3]
        fdir = os.path.join(args.data, "cvb", "raw_frames", clip)
        have = sorted((int(re.search(r"img_(\d+)", f).group(1)), os.path.join(fdir, f))
                      for f in os.listdir(fdir) if f.endswith(".jpg"))
        with open(ap) as fh:
            tracks = cvb_tracks(json.load(fh))
        times_all = np.array([(n - 1) / 30.0 for n, _ in have])
        end = times_all[-1]
        starts = list(np.arange(0.0, max(0.0, end - args.burst_s) + 1e-6, args.stride)) or [0.0]
        cache = {}

        def frame(n):
            if n not in cache:
                cache[n] = np.asarray(Image.open(dict(have)[n]).convert("RGB"))
            return cache[n]

        for s0 in starts:
            fr = [(n, (n - 1) / 30.0) for n, _ in have if s0 <= (n - 1) / 30.0 < s0 + args.burst_s]
            if len(fr) < 10:
                continue
            step = max(1, int(round(len(fr) / (args.burst_s * det_fps))))
            sampled = fr[::step]
            t = time.perf_counter()
            found = det.many([Image.fromarray(frame(n)) for n, _ in sampled])
            det_ms.append((time.perf_counter() - t) * 1000 / len(sampled))
            chains = link_chains([(ts, [c["bbox"] for c in f]) for (_, ts), f in zip(sampled, found)], link_iou)
            chains = [ch for ch in chains if len(ch) >= max(2, 0.5 * len(sampled))]
            ts_all = [ts for _, ts in fr]
            mid_n, mid_t = fr[len(fr) // 2]
            # the annotated cows of this burst, labelled as in training
            gts = []
            for tid, tr in tracks.items():
                inb = [x for x in tr if s0 <= (x[0] - 1) / 30.0 < s0 + args.burst_s]
                if len(inb) < 2:
                    continue
                lp = [label_of("cvb", b) for _, _, b in inb]
                w = windows(np.array([(n - 1) / 30.0 for n, _, _ in inb]), np.array([c for c, _ in lp]),
                            np.array([p for _, p in lp]), length=args.burst_s + 1e-3)
                if not w:
                    continue
                gts.append({"track": tid, "label": w[0][2], "box": _box_at(inb, mid_n), "frames": inb})
            chain_mid = [interpolate_box(ch, mid_t) for ch in chains]
            pairs = detect_mod.match(chain_mid, [g["box"] for g in gts], 0.3)
            by_gt = {j: i for i, j, _ in pairs}
            for j, g in enumerate(gts):
                row = {"clip": clip, "start": round(float(s0), 2), "track": g["track"], "label": CLASSES[g["label"]],
                       "ruminating": g["label"] == 0}
                # oracle: the annotation's own boxes
                gb = {(n - 1) / 30.0: b for n, b, _ in g["frames"]}
                seen_n = {n for n, _, _ in g["frames"]}
                valid = [n in seen_n for n, _ in fr]          # the frames the annotation boxes her in
                crops = [crop(frame(n), interpolate_box(gb, ts), size, margin) for n, ts in fr]
                row["p_oracle"] = rumination_p(crops, ts_all, valid)
                if j in by_gt:
                    ch = chains[by_gt[j]]
                    lo, hi = min(ch), max(ch)
                    valid = [lo - 0.5 <= ts <= hi + 0.5 for ts in ts_all]
                    crops = [crop(frame(n), interpolate_box(ch, ts), size, margin) for n, ts in fr]
                    row["p"] = rumination_p(crops, ts_all, valid)
                else:
                    row["p"] = None                  # the detector missed this cow in this burst
                rows.append(row)
            for i, ch in enumerate(chains):
                if i in {a for a, _, _ in pairs}:
                    continue
                lo, hi = min(ch), max(ch)
                valid = [lo - 0.5 <= ts <= hi + 0.5 for ts in ts_all]
                crops = [crop(frame(n), interpolate_box(ch, ts), size, margin) for n, ts in fr]
                extras.append({"clip": clip, "start": round(float(s0), 2), "p": rumination_p(crops, ts_all, valid)})
        print(f"\r  {k}/{len(anns)} clips, {len(rows)} cow-bursts", end="", flush=True)
    print(flush=True)

    y = np.array([r["ruminating"] for r in rows])
    found = np.array([r["p"] is not None for r in rows])
    p_or = np.array([r["p_oracle"] for r in rows])
    p_det = np.array([r["p"] if r["p"] is not None else 0.0 for r in rows])
    res = {"clips": len(anns), "cow_bursts": len(rows), "ruminating": int(y.sum()), "threshold": thr,
           "detector": os.path.abspath(args.detector), "detector_threshold": det.threshold, "tiles": det.tiles,
           "detector_ms_per_frame": round(float(np.median(det_ms)), 1) if det_ms else None,
           "found_share": float(found.mean()), "found_share_ruminating": float(found[y].mean()) if y.any() else None,
           "oracle": scores(p_or, y, thr),
           "found": scores(p_det[found], y[found], thr),
           "system": scores(p_det, y, thr)}
    fa = sum(e["p"] >= thr for e in extras)
    s_ = res["system"]["at_threshold"]
    tp, fp = s_["tp"], s_["fp"] + fa
    pr = tp / (tp + fp) if tp + fp else 0.0
    res["system_with_extra_chains"] = {"extra_chains": len(extras), "false_alarms": int(fa), "precision": pr,
                                       "recall": s_["recall"],
                                       "f1": 2 * pr * s_["recall"] / (pr + s_["recall"]) if pr + s_["recall"] else 0.0}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.model)), "e2e_cvb.json")
    with open(out, "w") as fh:
        json.dump({**res, "rows": rows, "extras": extras}, fh, indent=1)
    print(f"\n[e2e] {res['cow_bursts']} annotated cow-bursts ({res['ruminating']} ruminating), detector found "
          f"{res['found_share']:.1%} of them ({(res['found_share_ruminating'] or 0):.1%} of the ruminating ones)")
    print(f"{'':28s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'AUC':>6s}")
    for name, key in (("model on annotated boxes", "oracle"), ("found cows, detector boxes", "found"),
                      ("system (misses = not rum.)", "system")):
        a = res[key]["at_threshold"]
        auc = res[key]["auc"]
        print(f"{name:28s} {a['precision']:9.1%} {a['recall']:7.1%} {a['f1']:6.3f} "
              f"{('%6.3f' % auc) if auc is not None else '     -'}")
    w = res["system_with_extra_chains"]
    print(f"{'system + extra chains':28s} {w['precision']:9.1%} {w['recall']:7.1%} {w['f1']:6.3f}"
          f"   ({w['false_alarms']} of {w['extra_chains']} chains on no annotated cow said ruminating)")
    print(f"-> {out}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("fetch", "extract", "train", "e2e"))
    p.add_argument("--data", default="/workspace/rum")
    p.add_argument("--feat", default=None, help="extract: where to write (default <data>/feat); train: where to read")
    p.add_argument("--out", default=None)
    p.add_argument("--sources", default="kaggle,cvb")
    p.add_argument("--limit", type=int, default=0, help="fetch: a trial with this many clips per source")
    p.add_argument("--cvb-step", type=int, default=2,
                   help="fetch: every N-th CVB frame (2: 15 fps, plenty for ~1 Hz chewing, half the 225k files)")
    p.add_argument("--cvb-clean", action="store_true", help="extract: delete a CVB clip's frames once done")
    p.add_argument("--cvb-split", default=None, help="fetch: only this CVB split's frames (val: for e2e)")
    p.add_argument("--model", default=None, help="e2e: rumination.pt")
    p.add_argument("--detector", default=None, help="e2e: the detector's best/")
    p.add_argument("--threshold", type=float, default=None, help="e2e: detector threshold (default its own)")
    p.add_argument("--tiles", type=int, default=None, help="e2e: 1 whole frame + tiles (small far cows)")
    p.add_argument("--rum-threshold", type=float, default=None, help="e2e: rumination threshold (default the model's)")
    p.add_argument("--burst-s", type=float, default=7.0)
    p.add_argument("--stride", type=float, default=3.5, help="e2e: a burst every N s of a clip")
    p.add_argument("--encoder", default="facebook/dinov2-small")
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--motion", type=int, default=0, help="train: 1 adds the motion rhythm (motion.py) to the frames")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=200, help="batches an epoch")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.feat = args.feat or os.path.join(args.data, "feat")
    if args.stage == "extract":
        args.out = args.out or args.feat
        return cmd_extract(args)
    if args.stage == "e2e":
        if not args.model or not args.detector:
            sys.exit("e2e needs --model <rumination.pt> --detector <best/>")
        return cmd_e2e(args)
    if args.stage == "train":
        args.out = args.out or os.path.join(args.data, "model" + ("_motion" if args.motion else ""))
        return cmd_train(args)
    return cmd_fetch(args)


if __name__ == "__main__":
    sys.exit(main())
