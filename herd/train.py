"""Stage A: train the temporal transformer and every head on saved frame vectors.

No images and no big encoder in the loop (cbvd_bursts.py ran DINOv2 once), so
a run takes minutes and settings are cheap to try. One run trains all heads;
each example trains only the heads it has labels for.

    python train.py --features /workspace/herd/features --out /workspace/herd/run1

Per step:
  bursts   P cows x 2 views. A view is a random time crop of the burst (4 s of
           7), played slightly faster or slower, with random frames dropped
           (a cow walks behind another). Two views of one burst are the same
           cow; every other burst in the batch is another cow - taken from the
           same clips where possible, so the scene cannot tell them apart.
           Losses: supervised contrastive on the fingerprints; posture,
           activity and rumination on the pooled output.
  frames   a batch of keyframe vectors with their exact boxes: posture and
           activity for the once-a-second path.

10% of the training clips are held out as dev: they pick the best epoch and
fit the reliability (NaN) model (abstain.py). val is scored at the end and
written as cowbench results.jsonl, comparable with the LoRA runs.
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

from common import ACTIVITIES, POSTURES, device, read_jsonl, write_json

FPS = 25.0


# ------------------------------------------------------------------ data

class Split:
    """All bursts and keyframes of one split, in memory (fp16)."""

    def __init__(self, folder, clips=None, extra_keys=False):
        rows = read_jsonl(os.path.join(folder, "index.jsonl"))
        self.meta = json.load(open(os.path.join(folder, "meta.json"), encoding="utf-8"))
        keep = (lambda c: True) if clips is None else (lambda c: c in clips)
        self.bursts, self.keys = [], []
        bf, bt, kf, bm, bok = [], [], [], [], []
        have_motion = True
        by_clip = collections.defaultdict(list)
        for r in rows:
            if r["kind"] in ("burst", "key") and keep(r["clip"]):
                by_clip[r["clip"]].append(r)
        for clip in sorted(by_clip, key=lambda c: int(c)):
            z = np.load(os.path.join(folder, f"{clip}.npz"))
            mpath = os.path.join(folder, f"{clip}.motion.npz")
            zm = np.load(mpath) if os.path.exists(mpath) else None
            for r in by_clip[clip]:
                if r["kind"] == "burst" and len(z["burst_feats"]):
                    self.bursts.append(r)
                    bf.append(z["burst_feats"][r["i"]])
                    bt.append(z["burst_times"][r["i"]])
                    if zm is None:
                        have_motion = False
                    elif have_motion:
                        bm.append(zm["motion"][r["i"]])
                        bok.append(bool(zm["ok"][r["i"]]))
                elif r["kind"] == "key":
                    self.keys.append(r)
                    kf.append(z["key_feats"][r["i"]])
        dim = self.meta["dim"]
        self.burst_feats = np.stack(bf) if bf else np.zeros((0, 1, dim), np.float16)
        self.burst_times = np.stack(bt) if bt else np.zeros((0, 1), np.float32)
        self.key_feats = np.stack(kf) if kf else np.zeros((0, dim), np.float16)
        self.dim = dim
        # the rhythm of each burst (cbvd_bursts.py motion); None until extracted for every clip
        self.burst_motion = (np.stack(bm) * np.array(bok, np.float32)[:, None]
                             if have_motion and bm and len(bm) == len(self.bursts) else None)
        self.motion_dim = 0 if self.burst_motion is None else self.burst_motion.shape[1]
        from model import box_pos
        box_of = {(k["clip"], k["uid"]): k.get("bbox", [0.4, 0.4, 0.6, 0.6]) for k in self.keys}
        self.key_pos = np.array([box_pos(k.get("bbox", [0.4, 0.4, 0.6, 0.6])) for k in self.keys], np.float32)
        self.burst_pos = np.array([np.mean([box_pos(box_of[(b["clip"], u)]) for u in b.get("uids", [])
                                            if (b["clip"], u) in box_of] or [[0.5, 0.5, 0.2, 0.2]], 0)
                                   for b in self.bursts], np.float32).reshape(-1, 4)
        self.clips = sorted({r["clip"] for r in self.bursts + self.keys}, key=int)
        self.n_extra_keys = 0
        if extra_keys:
            self._add_extra_keys(folder, keep)

    def _add_extra_keys(self, folder, keep):
        """Keyframe crops from the detector's boxes and jittered boxes (cbvd_bursts.py
        keys): the frame heads learn the boxes the barn will give them, not only
        the annotation's. Training only - dev and val stay on annotated boxes,
        and eval-det scores the detector's."""
        from model import box_pos
        rows = read_jsonl(os.path.join(folder, "keys_aug.jsonl"))
        path = os.path.join(folder, "keys_aug.npz")
        if not rows or not os.path.exists(path):
            sys.exit(f"--det-keys 1 needs {path}: herd.py keys --split train")
        feats = np.load(path)["feats"]
        idx = [i for i, r in enumerate(rows) if keep(r["clip"])]
        self.keys += [rows[i] for i in idx]
        self.key_feats = np.concatenate([self.key_feats, feats[idx]])
        self.key_pos = np.concatenate([self.key_pos, np.array([box_pos(rows[i]["bbox"]) for i in idx],
                                                              np.float32).reshape(-1, 4)])
        self.n_extra_keys = len(idx)


def clips_of(folder):
    return sorted({r["clip"] for r in read_jsonl(os.path.join(folder, "index.jsonl"))
                   if r["kind"] == "clip_done"}, key=int)


def view(feats, times, rng, crop_s, train=True, drop=0.15, block=0.3, speed=(0.8, 1.25), from_start=None):
    """One view of a burst -> (frames, times, valid) of a fixed length.

    from_start: take the crop at this offset (s) instead of a random one, no
    jitter and no dropping - the deterministic views used for scoring."""
    T = len(times)
    n = int(round(crop_s * FPS))
    f = rng.uniform(*speed) if train else 1.0
    span = int(math.ceil(n * f))
    start = (rng.randint(0, max(0, T - span)) if from_start is None
             else min(max(0, int(round(from_start * FPS))), max(0, T - span)))
    idx = np.clip(np.round(start + np.arange(n) * f).astype(int), 0, T - 1)
    t = np.arange(n, dtype=np.float32) / FPS           # nominal spacing: tempo changes, gait stays
    valid = np.ones(n, bool)
    if train:
        valid &= np.array([rng.random() > drop for _ in range(n)])
        if rng.random() < block:                         # a stretch where the cow is hidden
            b = rng.randint(n // 8, n // 3)
            s = rng.randint(0, n - b)
            valid[s:s + b] = False
        if not valid.any():
            valid[rng.randrange(n)] = True
    return feats[idx], t, valid


def burst_batch(split, ids, rng, crop_s, train=True, starts=None, max_crop_s=None):
    """Views of a batch of bursts, padded to one length. In training the view
    length is random in [crop_s, max_crop_s]: the barn sends whole 7 s bursts,
    scoring uses half bursts, and the model must take both."""
    import torch
    xs, ts, vs = [], [], []
    for k, i in enumerate(ids):
        T = len(split.burst_times[i]) / FPS
        cs = rng.uniform(crop_s, min(max_crop_s, T / 1.25)) if (train and max_crop_s) else crop_s
        x, t, v = view(split.burst_feats[i], split.burst_times[i], rng, cs, train,
                       from_start=None if starts is None else starts[k])
        xs.append(x), ts.append(t), vs.append(v)
    n = max(len(t) for t in ts)
    pad = lambda a, fill: np.concatenate([a, np.full((n - len(a),) + a.shape[1:], fill, a.dtype)])
    xs = [pad(x, 0) for x in xs]
    ts = [pad(t, 0) for t in ts]
    vs = [pad(v, False) for v in vs]
    dev = device()
    return (torch.from_numpy(np.stack(xs)).float().to(dev), torch.from_numpy(np.stack(ts)).to(dev),
            torch.from_numpy(np.stack(vs)).to(dev))


def positions(arr, ids, rng=None, jitter=0.0):
    """Box centre and size; a little jitter in training, so the model learns
    places in the barn, not the exact pixel of one annotated cow."""
    import torch
    p = arr[ids].copy()
    if rng is not None and jitter:
        p += np.array([[rng.uniform(-jitter, jitter) for _ in range(4)] for _ in ids], np.float32)
    return torch.from_numpy(p).to(device())


def motion_of(split, ids, rng=None, drop=0.1):
    """The bursts' motion features; in training a few are zeroed, as for a cow
    the barn saw too briefly to compute them."""
    import torch
    if split.burst_motion is None:
        sys.exit("the model takes motion features and this split has none yet: herd.py motion --split <split>")
    m = split.burst_motion[ids].copy()
    if rng is not None:
        m[[k for k in range(len(ids)) if rng.random() < drop]] = 0
    return torch.from_numpy(m).float().to(device())


def labels(split, ids, field):
    import torch
    return torch.tensor([split.bursts[i][field] for i in ids], device=device())


def pk_sampler(split, P, rng):
    """P bursts per batch, drawn clip by clip so negatives share a scene."""
    by_clip = collections.defaultdict(list)
    for i, r in enumerate(split.bursts):
        by_clip[r["clip"]].append(i)
    clips = [c for c in by_clip if by_clip[c]]
    P = min(P, len(split.bursts))
    while True:
        rng.shuffle(clips)
        batch = []
        for c in clips:
            pool = by_clip[c][:]
            rng.shuffle(pool)
            for i in pool[:max(2, P // 4)]:
                batch.append(i)
                if len(batch) == P:
                    yield batch
                    batch = []


# ----------------------------------------------------------------- scoring

def embed(model, split, ids, crop_s, starts, batch=64):
    """Deterministic views -> fingerprints, quality weights, burst head outputs."""
    import torch
    rng = random.Random(0)
    out = collections.defaultdict(list)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            x, t, v = burst_batch(split, chunk, rng, crop_s, train=False, starts=[starts[j] for j in chunk])
            o = model.temporal(x, t, v, positions(split.burst_pos, chunk),
                               motion_of(split, chunk) if model.motion_dim else None)
            out["fingerprint"].append(o["fingerprint"].cpu())
            out["quality_max"].append(o["weights"].max(1).values.cpu())
            out["quality_mean"].append(o["quality"].masked_fill(~v, 0).sum(1).div(v.sum(1)).cpu())
            out["posture"].append(o["posture"].softmax(-1).cpu())
            out["activity"].append(o["activity"].softmax(-1).cpu())
            out["rumination"].append(o["rumination"].sigmoid().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def reid(model, split, crop_s):
    """Re-identification within the split: the first crop_s seconds of every
    burst are the gallery, the last crop_s seconds the queries (disjoint when
    2*crop_s <= burst length). Per query: the best match, the margin to the
    best other cow, and whether it was right - the reliability model's data."""
    ids = list(range(len(split.bursts)))
    T = split.burst_times.shape[1] / FPS
    g = embed(model, split, ids, crop_s, {i: 0.0 for i in ids})
    q = embed(model, split, ids, crop_s, {i: T - crop_s for i in ids})
    sim = q["fingerprint"] @ g["fingerprint"].T
    clip = np.array([r["clip"] for r in split.bursts])
    rows = []
    for scope in ("clip", "all"):
        s = sim.copy()
        if scope == "clip":
            s[clip[:, None] != clip[None, :]] = -2
        order = np.argsort(-s, 1)
        for i in ids:
            cands = [j for j in order[i] if s[i, j] > -2]
            if len(cands) < 2:
                continue
            best, second = cands[0], cands[1]
            rows.append({"scope": scope, "query": i, "best": int(best), "correct": int(best == i),
                         "sim": float(s[i, best]), "margin": float(s[i, best] - s[i, second]),
                         "quality_max": float(q["quality_max"][i]), "quality_mean": float(q["quality_mean"][i]),
                         "area": float(split.bursts[i]["area"]), "n_candidates": len(cands)})
    return rows, q


def frame_predictions(model, split, batch=1024):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(split.keys), batch):
            x = torch.from_numpy(split.key_feats[i:i + batch]).float().to(device())
            o = model.frame(x, positions(split.key_pos, list(range(i, min(i + batch, len(split.keys))))))
            out.append(torch.stack([o["posture"].argmax(-1), o["activity"].argmax(-1)], 1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 2), int)


def best_threshold(p, y):
    """The rumination cut-off with the best F1 (rumination is rare: 0.5 is not it)."""
    ts = np.linspace(0.05, 0.95, 91)
    f1s = []
    for t in ts:
        pr = p >= t
        tp, fpos, fn = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
        f1s.append(2 * tp / max(1, 2 * tp + fpos + fn))
    f1s = np.array(f1s)
    # An over-confident model leaves F1 flat over a wide range; run2 then took
    # the plateau's lowest end (0.05) while val's best was 0.82. Take its middle.
    top = np.where(f1s >= f1s.max() - 0.005)[0]
    return float(f1s.max()), float(ts[top[len(top) // 2]])


def share_threshold(p, y):
    """The cut-off at which the share of bursts called ruminating equals the
    true share: per burst it is wrong more often than the F1 cut-off, but
    summed into minutes it neither inflates nor shrinks rumination time."""
    ts = np.linspace(0.05, 0.95, 91)
    target = float(np.mean(y)) if len(y) else 0.0
    return float(ts[int(np.argmin([abs(float(np.mean(p >= t)) - target) for t in ts]))])


def motion_baseline(train_dir, val_dir):
    """Rumination from the motion features alone (logistic regression): does
    the rhythm say anything by itself? F1 at the cut-off best on val - an upper
    bound, a diagnostic only. None until motion is extracted."""
    import torch

    def load(folder):
        rows = [r for r in read_jsonl(os.path.join(folder, "index.jsonl")) if r["kind"] == "burst"]
        xs, ys = [], []
        for clip in sorted({r["clip"] for r in rows}, key=int):
            path = os.path.join(folder, f"{clip}.motion.npz")
            if not os.path.exists(path):
                return None, None
            z = np.load(path)
            for r in rows:
                if r["clip"] == clip and r["i"] < len(z["motion"]) and z["ok"][r["i"]] and r["rumination"] >= 0:
                    xs.append(z["motion"][r["i"]])
                    ys.append(r["rumination"])
        return (np.stack(xs), np.array(ys)) if xs else (None, None)

    xt, yt = load(train_dir)
    xv, yv = load(val_dir)
    if xt is None or xv is None or yt.min() == yt.max():
        return None
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    X = torch.from_numpy((xt - mu) / sd).float()
    Y = torch.from_numpy(yt).float()
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-2)
    pw = torch.tensor(math.sqrt((1 - Y.mean()) / Y.mean().clamp(min=1e-3)))
    for _ in range(400):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(X).squeeze(-1), Y, pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = lin(torch.from_numpy((xv - mu) / sd).float()).squeeze(-1).sigmoid().numpy()
    return best_threshold(p, yv)[0]


def evaluate(model, split, crop_s, rum_thr=0.5):
    rows, q = reid(model, split, crop_s)
    m = {}
    for scope in ("clip", "all"):
        r = [x for x in rows if x["scope"] == scope]
        m[f"reid_top1_{scope}"] = float(np.mean([x["correct"] for x in r])) if r else None
    fp = frame_predictions(model, split)
    kp = np.array([k["posture"] for k in split.keys])
    ka = np.array([k["activity"] for k in split.keys])
    ok = kp >= 0
    m["frame_posture_error"] = float(np.mean(fp[ok, 0] != kp[ok])) if ok.any() else None
    m["frame_activity_error"] = float(np.mean(fp[:, 1] != ka)) if len(ka) else None
    br = np.array([b["rumination"] for b in split.bursts])
    pr = q["rumination"] >= rum_thr
    if len(br):
        tp, fn, fpos = int((pr & (br == 1)).sum()), int((~pr & (br == 1)).sum()), int((pr & (br == 0)).sum())
        m["rumination_threshold"] = float(rum_thr)
        m["rumination_recall"] = tp / max(1, tp + fn)
        m["rumination_precision"] = tp / max(1, tp + fpos)
        m["rumination_f1"] = 2 * tp / max(1, 2 * tp + fpos + fn)
        m["rumination_best_f1"], m["rumination_best_threshold"] = best_threshold(q["rumination"], br)
        bp = np.array([b["posture"] for b in split.bursts])
        okp = bp >= 0
        m["burst_posture_error"] = float(np.mean(q["posture"].argmax(1)[okp] != bp[okp])) if okp.any() else None
    return m, rows, q, fp


def write_cowbench(split, fp, q, out_dir, rum_thr=0.5):
    """val keyframes as cowbench results, the once-a-second path only: posture
    and activity (feeding / drinking / none) from the frame heads. Rumination
    stays with the bursts - it is scored there (eval_val.json) and reported
    as minutes from bursts - and is not mixed into these per-frame answers: a
    cow annotated ruminating is "none" here (neither feeding nor drinking),
    flagged gt_rumination, with her burst's rumination_p alongside. Then
    cowbench.py score / report / compare work on herd as on any run."""
    os.makedirs(out_dir, exist_ok=True)
    rum = {}
    for b, p in zip(split.bursts, q["rumination"]):
        for uid in b.get("uids", []):
            rum[uid] = float(p)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as fh:
        for k, (pp, pa) in zip(split.keys, fp):
            p = rum.get(k["uid"])
            fh.write(json.dumps({"id": k["uid"], "video_id": k["clip"], "timestamp": k["timestamp"],
                                 "bbox": k.get("bbox", [0, 0, 1, 1]),
                                 "gt_posture": POSTURES[k["posture"]] if k["posture"] >= 0 else None,
                                 "gt_activity": ACTIVITIES[k["activity"]], "posture": POSTURES[pp],
                                 "activity": ACTIVITIES[pa], "gt_rumination": bool(k["rumination"]),
                                 "rumination_p": p,
                                 "rumination": None if p is None else bool(p >= rum_thr)}) + "\n")
    write_json(os.path.join(out_dir, "run_meta.json"), {
        "model": "herd: DINOv2-S frame heads (1 fps); rumination scored on bursts, not here",
        "engine": "herd, PyTorch", "reasoning": "n/a", "temperature": 0.0, "seed": 0,
        "render_mode": "224 px crop of the cow", "unit": "cow", "frames": 1, "max_width": 224,
        "annotations": "annotations/ava_val_v2.1.csv", "boxes": {"source": "annotation"},
        "prompt_sha": "herd", "videos": "", "excluded": []})


# ------------------------------------------------------------------- train

def cmd_train(args):
    import torch
    from model import HerdModel, masked_bce, masked_ce, supcon

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tr_dir = os.path.join(args.features, "train")
    all_clips = clips_of(tr_dir)
    dev_clips = set(random.Random(args.seed).sample(all_clips, max(1, int(len(all_clips) * args.holdout))))
    train = Split(tr_dir, set(all_clips) - dev_clips, extra_keys=bool(args.det_keys))
    dev = Split(tr_dir, dev_clips)
    print(f"[train] {len(train.bursts)} bursts / {len(train.keys) - train.n_extra_keys} keyframe cows in "
          f"{len(train.clips)} clips; "
          f"dev {len(dev.bursts)} / {len(dev.keys)} in {len(dev.clips)} clips", flush=True)
    if train.n_extra_keys:
        print(f"[train] + {train.n_extra_keys} keyframe crops from detector and jittered boxes (frame heads)", flush=True)
    if args.motion and not train.motion_dim:
        sys.exit("--motion 1 needs the motion features: herd.py motion --split train (and val)")
    model = HerdModel(train.dim, use_pos=bool(args.pos), d=args.d, layers=args.layers, heads=args.heads,
                      motion_dim=train.motion_dim if args.motion else 0).to(device())
    if args.motion:
        print(f"[train] motion features: {train.motion_dim} per burst -> rumination, activity", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * max(1, len(train.bursts) // args.P)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    pos = np.mean([b["rumination"] for b in train.bursts]) if train.bursts else 0.5
    # sqrt of the class ratio: the full ratio made the head call everything rumination
    rum_w = torch.tensor(math.sqrt((1 - pos) / max(pos, 1e-3)), device=device())
    sampler = pk_sampler(train, args.P, rng)
    os.makedirs(args.out, exist_ok=True)
    hist = open(os.path.join(args.out, "history.jsonl"), "a", encoding="utf-8")
    best, t0, step = -1.0, time.time(), 0
    since_best = 0
    for epoch in range(args.epochs):
        if since_best >= args.patience:
            print(f"[train] no better dev score for {args.patience} epochs - stopping", flush=True)
            break
        model.train()
        losses = collections.defaultdict(list)
        for _ in range(max(1, len(train.bursts) // args.P)):
            ids = next(sampler)
            x1, t1, v1 = burst_batch(train, ids, rng, args.crop_s, max_crop_s=args.max_crop_s)
            x2, t2, v2 = burst_batch(train, ids, rng, args.crop_s, max_crop_s=args.max_crop_s)
            n = max(x1.shape[1], x2.shape[1])
            padt = lambda a, fill: torch.nn.functional.pad(a, (0, 0, 0, n - a.shape[1]) if a.dim() == 3
                                                           else (0, n - a.shape[1]), value=fill)
            bp = positions(train.burst_pos, ids, rng, 0.02)
            bm = torch.cat([motion_of(train, ids, rng), motion_of(train, ids, rng)]) if model.motion_dim else None
            o = model.temporal(torch.cat([padt(x1, 0), padt(x2, 0)]), torch.cat([padt(t1, 0), padt(t2, 0)]),
                               torch.cat([padt(v1, False), padt(v2, False)]), torch.cat([bp, bp]), bm)
            ident = torch.arange(len(ids), device=device()).repeat(2)
            post, act, rum = (labels(train, ids, f).repeat(2) for f in ("posture", "activity", "rumination"))
            l_id = supcon(o["fingerprint"], ident, args.temperature)
            l_post = masked_ce(o["posture"], post)
            l_act = masked_ce(o["activity"], act)
            keep = rum >= 0
            l_rum = torch.nn.functional.binary_cross_entropy_with_logits(
                o["rumination"][keep], rum[keep].float(), pos_weight=rum_w) if keep.any() else masked_bce(o["rumination"], rum)
            kidx = rng.sample(range(len(train.keys)), min(args.key_batch, len(train.keys)))
            kx = torch.from_numpy(train.key_feats[kidx]).float().to(device())
            fo = model.frame(kx, positions(train.key_pos, kidx, rng, 0.02))
            l_fp = masked_ce(fo["posture"], torch.tensor([train.keys[i]["posture"] for i in kidx], device=device()))
            l_fa = masked_ce(fo["activity"], torch.tensor([train.keys[i]["activity"] for i in kidx], device=device()))
            loss = args.w_id * l_id + l_post + l_act + l_rum + l_fp + l_fa
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in (("id", l_id), ("posture", l_post), ("activity", l_act), ("rumination", l_rum),
                         ("frame_posture", l_fp), ("frame_activity", l_fa), ("total", loss)):
                losses[k].append(float(v.detach()))
        m, _, _, _ = evaluate(model, dev, args.crop_s)
        # recall alone rewarded calling everything rumination (run1 kept epoch 3 of 40 for it)
        score = np.mean([v for v in (m.get("reid_top1_clip"), 1 - (m.get("frame_posture_error") or 0),
                                     1 - (m.get("frame_activity_error") or 0), m.get("rumination_best_f1"))
                         if v is not None])
        rec = {"epoch": epoch + 1, "step": step, "minutes": round((time.time() - t0) / 60, 1),
               **{f"loss_{k}": round(float(np.mean(v)), 4) for k, v in losses.items()},
               **{f"dev_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
               "dev_score": round(float(score), 4)}
        hist.write(json.dumps(rec) + "\n")
        hist.flush()
        print(f"[train] epoch {epoch + 1}/{args.epochs}  loss {rec['loss_total']:.3f}  dev re-ID in clip "
              f"{m.get('reid_top1_clip') or 0:.1%}, frame posture err {m.get('frame_posture_error') or 0:.1%}, "
              f"frame activity err {m.get('frame_activity_error') or 0:.1%}, rumination F1 "
              f"{m.get('rumination_best_f1') or 0:.1%} at {m.get('rumination_best_threshold') or 0:.2f}", flush=True)
        since_best += 1
        if score > best:
            best = score
            since_best = 0
            model.save(os.path.join(args.out, "model.pt"), {"crop_s": args.crop_s, "epoch": epoch + 1,
                                                             "features": train.meta, "dev": m,
                                                             "dev_clips": sorted(dev_clips, key=int),
                                                             "rum_threshold": m.get("rumination_best_threshold", 0.5)})
    hist.close()
    print(f"[train] best dev score {best:.3f} -> {os.path.join(args.out, 'model.pt')}", flush=True)
    cmd_eval(args)


def cmd_eval(args):
    from model import HerdModel
    model, ck = HerdModel.load(os.path.join(args.out, "model.pt"), map_location=device())
    model.to(device())
    dev = Split(os.path.join(args.features, "train"), set(ck["dev_clips"]))
    md, _, qd, _ = evaluate(model, dev, ck["crop_s"])
    thr = ck.get("rum_threshold")
    if thr is None:      # a checkpoint from before the cut-off was calibrated: do it now, on its dev clips
        thr = md.get("rumination_best_threshold", 0.5)
    yd = np.array([b["rumination"] for b in dev.bursts])
    thr_time = share_threshold(qd["rumination"], yd)
    # the rumination cut-offs travel with the model: pipeline and reports read them.
    # rumination_threshold: per burst (best F1); _time: for minutes in reports (true share).
    write_json(os.path.join(args.out, "heads.json"), {"rumination_threshold": thr,
                                                      "rumination_threshold_time": thr_time})
    val = Split(os.path.join(args.features, "val"))
    m, rows, q, fp = evaluate(model, val, ck["crop_s"], thr)
    yv = np.array([b["rumination"] for b in val.bursts])
    if len(yv):
        m["rumination_share_true"] = float(yv.mean())
        m["rumination_share_called_f1"] = float(np.mean(q["rumination"] >= thr))
        m["rumination_threshold_time"] = thr_time
        m["rumination_share_called_time"] = float(np.mean(q["rumination"] >= thr_time))
    base = motion_baseline(os.path.join(args.features, "train"), os.path.join(args.features, "val"))
    if base is not None:
        m["rumination_f1_motion_only"] = base
    m["motion"] = bool(model.motion_dim)
    write_json(os.path.join(args.out, "eval_val.json"), {"epoch": ck["epoch"], **m})
    with open(os.path.join(args.out, "reid_val.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    write_cowbench(val, fp, q, os.path.join(args.out, "eval-val"), thr)
    print("[eval] val: " + ", ".join(f"{k} {v:.2f}" if "threshold" in k else
                                     (f"{k} {v:.1%}" if isinstance(v, float) else f"{k} {v}") for k, v in m.items()))
    print(f"[eval] cowbench format: python cowbench/cowbench.py --out {os.path.join(args.out, 'eval-val')} score")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("train", "eval"))
    p.add_argument("--features", default="/workspace/herd/features")
    p.add_argument("--out", default="/workspace/herd/run1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--P", type=int, default=64, help="bursts per step (two views each)")
    p.add_argument("--key-batch", type=int, default=256)
    p.add_argument("--crop-s", type=float, default=3.5, help="seconds per view; 2 x 3.5 fits 7 s disjoint")
    p.add_argument("--max-crop-s", type=float, default=7.0, help="longest training view (the barn sends 7 s)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--w-id", type=float, default=2.0, help="ID loss weight (weighted highest, as planned)")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=6, help="stop after this many epochs without a better dev score")
    p.add_argument("--pos", type=int, default=0, help="1: give the heads where the cow is in the frame")
    p.add_argument("--det-keys", type=int, default=0,
                   help="1: the frame heads also learn on detector / jittered boxes; needs herd.py keys --split train")
    p.add_argument("--motion", type=int, default=0,
                   help="1: the burst's rhythm (motion.py) to rumination and activity; needs herd.py motion")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
