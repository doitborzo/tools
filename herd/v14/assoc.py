#!/usr/bin/env python3
"""The association transformer (v14): which detections, on which cameras and when, are one cow.

Runs once per step for the whole barn. Tokens (one transformer, ~3-5 M parameters):

    current   every detection now, all cameras: its DINOv2 vector (1920), its box in the
              picture (x1 y1 x2 y2 cx cy w h), detector confidence, an ANONYMOUS camera slot
              (shuffled in every training example, so no camera is ever memorised), time 0
    history   the detections of the last steps, each tagged with the track it belongs to
              (a "link" slot shared by the track's tokens) and its age in seconds
    memory    recent confident cross-camera matches as pairs (a link slot per pair): how
              this barn's cameras connect, learned in context - no training at onboarding
    scene     each camera's whole frame as 8 x 8 DINOv2 patch descriptions (scene.py),
              refreshed every minute

Outputs: per detection an association embedding (128, unit length: one cow -> nearly one
point, over cameras and time) and "another camera sees this cow now"; per scene cell, for
every other camera slot, "that camera sees this floor too" (the overlap map). Track
continuation is read from the embeddings (a current detection against each track's history
tokens), so no separate head.

Trained on MmCows' labelled frames outside the test window only: real 4-camera moments
(cow ids on every box) and virtual cameras - 2-3 overlapping windows cut out of one frame,
which gives as many camera layouts as wanted with the right answers known exactly.

    python herd/v14/assoc.py train --data /workspace/v14/mmcows --out /workspace/v14/assoc
    python herd/v14/assoc.py eval  --data /workspace/v14/mmcows --out /workspace/v14/assoc
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys
import time

import numpy as np

import v14common as C
from v14common import read_jsonl, write_json

SLOTS, LINKS, KINDS = 8, 128, 4          # camera slots, link slots, token kinds
CUR, HIST, MEM, SCENE = 0, 1, 2, 3
EMB = 128
STEP_S = 15.0                            # MmCows' labelled frames: every 15 s


# ------------------------------------------------------------------ model

def build(det_dim=1920, scene_dim=384, d=256, layers=4, heads=4, dropout=0.1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AssocNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.kw = dict(det_dim=det_dim, scene_dim=scene_dim, d=d, layers=layers, heads=heads, dropout=dropout)
            self.det = nn.Sequential(nn.LayerNorm(det_dim), nn.Dropout(dropout), nn.Linear(det_dim, d))
            self.scene = nn.Sequential(nn.LayerNorm(scene_dim), nn.Linear(scene_dim, d))
            self.geo = nn.Sequential(nn.Linear(8 + 2, d), nn.GELU(), nn.Linear(d, d))
            self.slot = nn.Embedding(SLOTS, d)
            self.kind = nn.Embedding(KINDS, d)
            self.link = nn.Embedding(LINKS, d)
            block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                               activation="gelu")
            self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.emb = nn.Linear(d, EMB)
            self.elsewhere = nn.Linear(d, 1)
            self.overlap = nn.Linear(d, SLOTS)

        def forward(self, b):
            is_scene = (b["kind"] == SCENE).unsqueeze(-1).float()
            x = self.det(b["feat"]) * (1 - is_scene) + self.scene(b["sfeat"]) * is_scene
            g = torch.cat([b["geo"], b["conf"].unsqueeze(-1), (b["dt"] / 60.0).unsqueeze(-1)], -1)
            h = x + self.geo(g) + self.slot(b["slot"]) + self.kind(b["kind"]) + self.link(b["link"])
            h = self.norm(self.encoder(h, src_key_padding_mask=~b["valid"]))
            return {"emb": F.normalize(self.emb(h), dim=-1), "elsewhere": self.elsewhere(h).squeeze(-1),
                    "overlap": self.overlap(h)}

    return AssocNet()


def load(path, device=None):
    import torch
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    m = build(**ck["kw"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck


# ------------------------------------------------------------------ tokens

def tokens(dets, scenes, det_dim=1920, scene_dim=384):
    """One example -> arrays. dets: [{feat, box, conf, dt, slot, kind, link, label}],
    scenes: [{sfeat (64, C), centres (64, 2), cell, slot, overlap (64, SLOTS) or None}]."""
    n = len(dets) + sum(len(s["sfeat"]) for s in scenes)
    A = {"feat": np.zeros((n, det_dim), np.float32), "sfeat": np.zeros((n, scene_dim), np.float32),
         "geo": np.zeros((n, 8), np.float32), "conf": np.zeros(n, np.float32), "dt": np.zeros(n, np.float32),
         "slot": np.zeros(n, np.int64), "kind": np.zeros(n, np.int64), "link": np.zeros(n, np.int64),
         "label": np.full(n, -1, np.int64), "elsewhere": np.full(n, -1.0, np.float32),
         "overlap": np.full((n, SLOTS), -1.0, np.float32)}
    for i, t in enumerate(dets):
        x1, y1, x2, y2 = t["box"]
        A["feat"][i] = t["feat"]
        A["geo"][i] = [x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        A["conf"][i], A["dt"][i], A["slot"][i] = t.get("conf", 1.0), t.get("dt", 0.0), t["slot"]
        A["kind"][i], A["link"][i], A["label"][i] = t.get("kind", CUR), t.get("link", 0), t.get("label", -1)
        A["elsewhere"][i] = t.get("elsewhere", -1.0)
    i = len(dets)
    for s in scenes:
        k = len(s["sfeat"])
        A["sfeat"][i:i + k] = s["sfeat"]
        A["geo"][i:i + k, 4:6] = s["centres"]
        A["geo"][i:i + k, 6:8] = s["cell"]
        A["slot"][i:i + k], A["kind"][i:i + k] = s["slot"], SCENE
        if s.get("overlap") is not None:
            A["overlap"][i:i + k] = s["overlap"]
        i += k
    return A


def batch(examples, device):
    import torch
    n = max(len(e["kind"]) for e in examples)
    out = {}
    for k in examples[0]:
        arrs = []
        for e in examples:
            a = e[k]
            pad = [(0, n - len(a))] + [(0, 0)] * (a.ndim - 1)
            fill = -1 if k in ("label", "elsewhere", "overlap") else 0
            arrs.append(np.pad(a, pad, constant_values=fill))
        out[k] = torch.from_numpy(np.stack(arrs)).to(device)
    out["valid"] = torch.from_numpy(np.stack([np.arange(n) < len(e["kind"]) for e in examples])).to(device)
    return out


# ------------------------------------------------------------------ MmCows as examples

class Data:
    def __init__(self, folder):
        import scene
        self.frames = read_jsonl(os.path.join(folder, "frames.jsonl"))
        z = np.load(os.path.join(folder, "assoc.npz"))
        self.box_feats, self.scene = z["box_feats"], z["scene"]
        rows = read_jsonl(os.path.join(folder, "assoc_rows.jsonl"))
        self.boxes = collections.defaultdict(list)          # frame -> [(cow, box, feat row)]
        for j, r in enumerate(rows):
            self.boxes[r["frame"]].append((r["cow"], r["box"], j))
        self.at = collections.defaultdict(dict)              # (split, key) -> {cam: frame}
        for fi, f in enumerate(self.frames):
            self.at[(f["split"], round(f["ts"] / STEP_S))][f["cam"]] = fi
        self.keys = {s: sorted(k for (sp, k) in self.at if sp == s) for s in ("train", "window")}
        # keys are 15-s steps; training frames may be sparser (one a minute = 4 keys): history and
        # memory look back by the split's own spacing, or they would find nothing
        self.stride = {s: max(1, int(np.median(np.diff(k)))) if len(k) > 1 else 1 for s, k in self.keys.items()}
        self.cams = sorted({f["cam"] for f in self.frames})
        self.full = {fi: scene.sub_grid(self.scene[fi]) for fi in range(len(self.frames))}
        self.covis = self._covisibility()

    def _covisibility(self, cells=8, min_n=3):
        """Static per camera pair: of the cows whose box centre is in cell c of camera a, the
        share also seen by camera b at that moment (training frames only) -> overlap targets."""
        cnt = collections.defaultdict(lambda: np.zeros((cells * cells,)))
        hit = collections.defaultdict(lambda: np.zeros((cells * cells,)))
        for (sp, k), cams in self.at.items():
            if sp != "train":
                continue
            present = {cam: {c for c, _, _ in self.boxes[fi]} for cam, fi in cams.items()}
            for a, fi in cams.items():
                for cow, b, _ in self.boxes[fi]:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    cell = min(cells - 1, int(cy * cells)) * cells + min(cells - 1, int(cx * cells))
                    for bcam in cams:
                        if bcam != a:
                            cnt[(a, bcam)][cell] += 1
                            hit[(a, bcam)][cell] += cow in present[bcam]
        out = {}
        for key, n in cnt.items():
            t = np.where(n >= min_n, (hit[key] / np.maximum(n, 1) >= 0.5).astype(np.float32), -1.0)
            out[key] = t
        return out

    # -------------------------------------------------------------- real cameras
    def real(self, split, key, rng, train=True, history=2, memory=16):
        cams = self.at.get((split, key), {})
        if len(cams) < 2:
            return None
        names = sorted(cams)
        slots = dict(zip(names, rng.sample(range(SLOTS), len(names))))
        dets, now = [], collections.defaultdict(set)
        for cam in names:
            for cow, _, _ in self.boxes[cams[cam]]:
                now[cow].add(cam)
        for cam in names:
            for cow, b, j in self.boxes[cams[cam]]:
                if train and rng.random() < 0.08:
                    continue                                  # a missed detection
                dets.append({"feat": self.box_feats[j], "box": jitter(b, rng) if train else b,
                             "conf": rng.uniform(0.4, 1.0) if train else 1.0, "slot": slots[cam], "kind": CUR,
                             "label": cow, "elsewhere": float(len(now[cow] - {cam}) > 0)})
        link = {}
        st = self.stride[split]
        for h in range(1, history + 1):
            past = self.at.get((split, key - h * st), {})
            for cam in names:
                if cam not in past:
                    continue
                for cow, b, j in self.boxes[past[cam]]:
                    if cow not in link:
                        link[cow] = 1 + len(link) % 60
                    lk = link[cow] if not (train and rng.random() < 0.05) else 1 + rng.randrange(60)
                    dets.append({"feat": self.box_feats[j], "box": jitter(b, rng) if train else b, "conf": 1.0,
                                 "dt": -h * st * STEP_S, "slot": slots[cam], "kind": HIST, "link": lk, "label": cow})
        pairs = 0
        for h in range(max(4, st), 21, max(2, st)):           # 1-5 minutes ago
            past = self.at.get((split, key - h), {})
            seen = collections.defaultdict(list)
            for cam in names:
                if cam in past:
                    for cow, b, j in self.boxes[past[cam]]:
                        seen[cow].append((cam, b, j))
            for cow, obs in seen.items():
                if len(obs) >= 2 and pairs < memory:
                    a, c = rng.sample(obs, 2) if train else obs[:2]
                    lk = 64 + pairs
                    for cam, b, j in (a, c):
                        dets.append({"feat": self.box_feats[j], "box": b, "conf": 1.0, "dt": -h * STEP_S,
                                     "slot": slots[cam], "kind": MEM, "link": lk, "label": cow})
                    pairs += 1
        import scene
        scenes = []
        cen = scene.cell_centres()
        for cam in names:
            ov = np.full((64, SLOTS), -1.0, np.float32)
            for other in names:
                if other != cam and (cam, other) in self.covis:
                    ov[:, slots[other]] = self.covis[(cam, other)]
            scenes.append({"sfeat": self.full[cams[cam]], "centres": cen, "cell": np.full((64, 2), 1 / 8, np.float32),
                           "slot": slots[cam], "overlap": ov})
        return tokens(dets, scenes, self.box_feats.shape[1], self.scene.shape[-1])

    # -------------------------------------------------------------- virtual cameras
    def virtual(self, split, key, rng, history=2, memory=16):
        cams = self.at.get((split, key), {})
        if not cams:
            return None
        cam = rng.choice(sorted(cams))
        n_v = rng.choice((2, 3))
        wins = []
        for _ in range(n_v):
            w, h = rng.uniform(0.45, 0.8), rng.uniform(0.45, 0.8)
            x0, y0 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            wins.append((x0, y0, x0 + w, y0 + h))
        slots = rng.sample(range(SLOTS), n_v)

        def inside(b, wn):
            ix = max(0.0, min(b[2], wn[2]) - max(b[0], wn[0])) * max(0.0, min(b[3], wn[3]) - max(b[1], wn[1]))
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area <= 0 or ix / area < 0.7:
                return None
            sx, sy = wn[2] - wn[0], wn[3] - wn[1]
            return [min(1, max(0, (b[0] - wn[0]) / sx)), min(1, max(0, (b[1] - wn[1]) / sy)),
                    min(1, max(0, (b[2] - wn[0]) / sx)), min(1, max(0, (b[3] - wn[1]) / sy))]

        def views(fi):
            out = []
            for v, wn in enumerate(wins):
                for cow, b, j in self.boxes[fi]:
                    nb = inside(b, wn)
                    if nb is not None:
                        out.append((v, cow, nb, j))
            return out

        dets = []
        now = views(cams[cam])
        where = collections.defaultdict(set)
        for v, cow, _, _ in now:
            where[cow].add(v)
        for v, cow, nb, j in now:
            if rng.random() < 0.08:
                continue
            dets.append({"feat": self.box_feats[j], "box": jitter(nb, rng), "conf": rng.uniform(0.4, 1.0),
                         "slot": slots[v], "kind": CUR, "label": cow, "elsewhere": float(len(where[cow] - {v}) > 0)})
        link = {}
        st = self.stride[split]
        for h in range(1, history + 1):
            past = self.at.get((split, key - h * st), {})
            if cam in past:
                for v, cow, nb, j in views(past[cam]):
                    link.setdefault(cow, 1 + len(link) % 60)
                    dets.append({"feat": self.box_feats[j], "box": jitter(nb, rng), "conf": 1.0,
                                 "dt": -h * st * STEP_S, "slot": slots[v], "kind": HIST, "link": link[cow], "label": cow})
        pairs = 0
        for h in range(max(4, st), 21, max(2, st)):
            past = self.at.get((split, key - h), {})
            if cam not in past:
                continue
            seen = collections.defaultdict(list)
            for v, cow, nb, j in views(past[cam]):
                seen[cow].append((v, nb, j))
            for cow, obs in seen.items():
                if len({o[0] for o in obs}) >= 2 and pairs < memory:
                    a = obs[0]
                    c = next(o for o in obs if o[0] != a[0])
                    for v, nb, j in (a, c):
                        dets.append({"feat": self.box_feats[j], "box": nb, "conf": 1.0, "dt": -h * STEP_S,
                                     "slot": slots[v], "kind": MEM, "link": 64 + pairs, "label": cow})
                    pairs += 1
        import scene
        scenes = []
        g = self.scene[cams[cam]]
        for v, wn in enumerate(wins):
            cen = scene.cell_centres(region=wn)
            ov = np.full((64, SLOTS), -1.0, np.float32)
            for u, wo in enumerate(wins):
                if u != v:
                    ov[:, slots[u]] = ((cen[:, 0] >= wo[0]) & (cen[:, 0] <= wo[2]) &
                                       (cen[:, 1] >= wo[1]) & (cen[:, 1] <= wo[3])).astype(np.float32)
            scenes.append({"sfeat": scene.sub_grid(g, wn), "centres": scene.cell_centres(),
                           "cell": np.full((64, 2), 1 / 8, np.float32), "slot": slots[v], "overlap": ov})
        return tokens(dets, scenes, self.box_feats.shape[1], self.scene.shape[-1])


def jitter(b, rng, s=0.06):
    w, h = b[2] - b[0], b[3] - b[1]
    d = [rng.gauss(0, s) * w, rng.gauss(0, s) * h, rng.gauss(0, s) * w, rng.gauss(0, s) * h]
    out = [b[0] + d[0], b[1] + d[1], b[2] + d[2], b[3] + d[3]]
    out = [min(1.0, max(0.0, v)) for v in out]
    if out[2] <= out[0] or out[3] <= out[1]:
        return list(b)
    return out


# ------------------------------------------------------------------ losses and scores

def losses(o, b, temperature=0.1):
    import torch
    import torch.nn.functional as F
    from model import supcon
    l_id = []
    for i in range(len(b["label"])):
        keep = (b["label"][i] >= 0) & b["valid"][i]
        if keep.sum() >= 2:
            l_id.append(supcon(o["emb"][i][keep], b["label"][i][keep], temperature))
    l_id = torch.stack(l_id).mean() if l_id else o["emb"].sum() * 0
    m = b["elsewhere"] >= 0
    l_else = F.binary_cross_entropy_with_logits(o["elsewhere"][m], b["elsewhere"][m]) if m.any() else o["emb"].sum() * 0
    mo = b["overlap"] >= 0
    l_ov = F.binary_cross_entropy_with_logits(o["overlap"][mo], b["overlap"][mo]) if mo.any() else o["emb"].sum() * 0
    return l_id, l_else, l_ov


def pair_scores(emb, label, kind, slot):
    """Current detections on different cameras: (similarity, same cow?) for every pair."""
    cur = np.where(kind == CUR)[0]
    s, y = [], []
    for a in range(len(cur)):
        for c in range(a + 1, len(cur)):
            i, j = cur[a], cur[c]
            if slot[i] != slot[j]:
                s.append(float(emb[i] @ emb[j]))
                y.append(int(label[i] == label[j]))
    return s, y


def track_hits(emb, label, kind, link):
    """Each current detection whose cow has a history track: is the closest track hers?"""
    hist = np.where(kind == HIST)[0]
    tracks = collections.defaultdict(list)
    for i in hist:
        tracks[link[i]].append(i)
    owner = {lk: collections.Counter(label[i] for i in idx).most_common(1)[0][0] for lk, idx in tracks.items()}
    hits = []
    for i in np.where(kind == CUR)[0]:
        if label[i] not in owner.values():
            continue
        best = max(tracks, key=lambda lk: max(float(emb[i] @ emb[j]) for j in tracks[lk]))
        hits.append(int(owner[best] == label[i]))
    return hits


def best_f1(s, y):
    s, y = np.asarray(s), np.asarray(y)
    if not len(s) or y.sum() == 0:
        return {"f1": None, "threshold": None, "precision": None, "recall": None, "pairs": int(len(s))}
    best = (0, 0.5, 0, 0)
    for t in np.unique(np.round(s, 3)):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        P = tp / max(1, pred.sum())
        R = tp / max(1, (y == 1).sum())
        f = 2 * P * R / (P + R) if P + R else 0
        if f > best[0]:
            best = (f, float(t), P, R)
    return {"f1": best[0], "threshold": best[1], "precision": best[2], "recall": best[3], "pairs": int(len(s)),
            "same_cow_pairs": int(y.sum())}


def at_threshold(s, y, t):
    s, y = np.asarray(s), np.asarray(y)
    if not len(s):
        return {"f1": None}
    pred = s >= t
    tp = int((pred & (y == 1)).sum())
    P = tp / max(1, pred.sum())
    R = tp / max(1, (y == 1).sum())
    return {"f1": 2 * P * R / (P + R) if P + R else 0.0, "precision": P, "recall": R, "threshold": t}


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ train / eval

def cmd_train(args):
    import torch
    from common import device as dev_of
    dev = dev_of()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    data = Data(args.data)
    keys = data.keys["train"]
    rng.shuffle(keys)
    n_val = max(1, len(keys) // 10)
    val_keys, tr_keys = sorted(keys[:n_val]), keys[n_val:]
    print(f"[assoc] {len(tr_keys)} training moments, {len(val_keys)} for picking the epoch (both outside the test "
          f"window); cameras {data.cams}", flush=True)
    model = build(data.box_feats.shape[1], data.scene.shape[-1], d=args.d, layers=args.layers, heads=args.heads,
                  dropout=args.dropout).to(dev)
    print(f"[assoc] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, from scratch", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tr_keys) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    os.makedirs(args.out, exist_ok=True)
    best, since, hist = -1.0, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ls = collections.defaultdict(list)
        for _ in range(steps):
            ex = []
            while len(ex) < args.batch:
                k = rng.choice(tr_keys)
                e = data.virtual("train", k, rng) if rng.random() < args.virtual else data.real("train", k, rng)
                if e is not None and (e["label"] >= 0).sum() >= 2:
                    ex.append(e)
            b = batch(ex, dev)
            o = model(b)
            l_id, l_else, l_ov = losses(o, b, args.temperature)
            loss = l_id + args.w_else * l_else + args.w_overlap * l_ov
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for n, v in (("id", l_id), ("elsewhere", l_else), ("overlap", l_ov)):
                ls[n].append(float(v.detach()) if torch.is_tensor(v) else float(v))
        model.eval()
        v = evaluate(model, data, "train", val_keys, dev)
        score = (v["cross_camera"]["f1"] or 0) + (v["track_top1"] or 0)
        rec = {"epoch": ep + 1, **{f"loss_{k}": round(float(np.mean(x)), 4) for k, x in ls.items()},
               "val_cross_camera_f1": v["cross_camera"]["f1"], "val_track_top1": v["track_top1"],
               "minutes": round((time.time() - t0) / 60, 1)}
        hist.append(rec)
        print(f"[assoc] epoch {ep + 1}: id {rec['loss_id']:.3f}  val cross-camera F1 "
              f"{v['cross_camera']['f1'] or 0:.3f}  track top-1 "
              f"{'-' if v['track_top1'] is None else format(v['track_top1'], '.1%')} ({v['track_cases']} cases)", flush=True)
        if score > best:
            best, since = score, 0
            torch.save({"kw": model.kw, "state": model.state_dict(), "epoch": ep + 1,
                        "pair_threshold": v["cross_camera"]["threshold"]}, os.path.join(args.out, "assoc.pt"))
        else:
            since += 1
            if since >= args.patience:
                print(f"[assoc] no better val score for {args.patience} epochs - stopping", flush=True)
                break
    write_json(os.path.join(args.out, "history.json"), hist)
    cmd_eval(args)


def evaluate(model, data, split, keys, dev, raw=False):
    """Cross-camera pairs (same cow?) and track continuation on the given moments."""
    import torch
    rng = random.Random(0)
    S, Y, hits, ov_s, ov_y = [], [], [], [], []
    for k0 in range(0, len(keys), 16):
        ex = [e for e in (data.real(split, k, rng, train=False) for k in keys[k0:k0 + 16]) if e is not None]
        if not ex:
            continue
        b = batch(ex, dev)
        with torch.no_grad():
            o = model(b) if not raw else None
        for i, e in enumerate(ex):
            n = len(e["kind"])
            if raw:                                           # plain DINOv2 similarity, no model
                f = e["feat"][:n]
                emb = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
            else:
                emb = o["emb"][i, :n].float().cpu().numpy()
                ov = torch.sigmoid(o["overlap"][i, :n]).float().cpu().numpy()
                m = e["overlap"] >= 0
                ov_s += ov[m].tolist()
                ov_y += e["overlap"][m].astype(int).tolist()
            s, y = pair_scores(emb, e["label"], e["kind"], e["slot"])
            S += s
            Y += y
            hits += track_hits(emb, e["label"], e["kind"], e["link"])
    return {"moments": len(keys), "cross_camera": best_f1(S, Y), "cross_camera_auc": auc(S, Y),
            "track_top1": float(np.mean(hits)) if hits else None, "track_cases": len(hits),
            "overlap_auc": auc(ov_s, ov_y) if ov_s else None, "_scores": (S, Y)}


def cmd_eval(args):
    from common import device as dev_of
    dev = dev_of()
    model, ck = load(os.path.join(args.out, "assoc.pt"), dev)
    model.to(dev)
    data = Data(args.data)
    keys = data.keys["window"]
    res = evaluate(model, data, "window", keys, dev)
    raw = evaluate(model, data, "window", keys, dev, raw=True)
    S, Y = res.pop("_scores")
    raw.pop("_scores")
    res["cross_camera_at_train_threshold"] = at_threshold(S, Y, ck.get("pair_threshold") or 0.5)
    out = {"model": {"epoch": ck["epoch"], "parameters": int(sum(v.numel() for v in ck["state"].values())), **ck["kw"]},
           "test_window": res, "baseline_plain_dinov2": raw,
           "what": "MmCows labelled moments inside the test window (never trained on): every pair of detections "
                   "on different cameras scored same cow / not (F1 at the best threshold and at the one picked on "
                   "the training side), and every detection's closest history track (top-1)"}
    write_json(os.path.join(args.out, "eval_assoc.json"), out)
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"[assoc] test window: cross-camera F1 {f(res['cross_camera']['f1'])} (plain DINOv2 "
          f"{f(raw['cross_camera']['f1'])}), track top-1 {f(res['track_top1'])} (plain {f(raw['track_top1'])}), "
          f"overlap AUC {f(res['overlap_auc'])}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True, help="data_mmcows.py's folder")
        s.add_argument("--out", required=True)
        if name == "train":
            s.add_argument("--epochs", type=int, default=60)
            s.add_argument("--patience", type=int, default=10)
            s.add_argument("--batch", type=int, default=16)
            s.add_argument("--virtual", type=float, default=0.5, help="share of virtual-camera examples")
            s.add_argument("--d", type=int, default=256)
            s.add_argument("--layers", type=int, default=4)
            s.add_argument("--heads", type=int, default=4)
            s.add_argument("--dropout", type=float, default=0.1)
            s.add_argument("--lr", type=float, default=3e-4)
            s.add_argument("--wd", type=float, default=0.05)
            s.add_argument("--temperature", type=float, default=0.1)
            s.add_argument("--w-else", type=float, default=0.5)
            s.add_argument("--w-overlap", type=float, default=0.5)
            s.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
