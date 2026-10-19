#!/usr/bin/env python3
"""The ID transformer (v14): a cow's sequence of crops -> one fingerprint, from scratch.

Input  : T frame vectors (DINOv2-S, 1920 numbers each: CLS + 2 x 2 pooled patches), their
         times in seconds and which are real (valid).
Output : fingerprint (512, unit length) and a quality score per frame (how much it counts).

Two ways to split (--split):
  days  MultiCamCows2024: days 1-5 (14-18 Aug) train, day 6 picks the epoch and the "not sure"
        model, day 7 is the test nobody looked at.
  cows  MmCows, one barn (data_mmcows.py idcrops): 8 of the 16 cows train, 3 pick the epoch,
        5 the model never saw are the test - album before 12:00, queries after 15:00. Its frames
        are a minute apart: --time-aug 0.02 also shows them squeezed to the live stream's pace.
A training batch is P cows x 2 sequences each, from different days (parts of the day) or
cameras where the cow has them. Loss: supervised contrastive (model.supcon). Small (~3 M
parameters), dropout, weight decay, early stopping.

    python herd/v14/idnet.py train --feats /workspace/v14/mcc --out /workspace/v14/idnet
    python herd/v14/idnet.py train --feats /workspace/v14/mmcows/idcrops --split cows --time-aug 0.02 --P 8 \
        --out /workspace/v14/idnet

Writes <out>/idnet.pt (model + its settings + the split + the enrolment threshold + the
"not sure" model) and <out>/eval_id.json: top-1 / top-5 / mAP and ID precision / recall / F1
(abstentions count as misses) on the test part.
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys
import time

import numpy as np

import v14common as C  # noqa: F401  (paths)
from v14common import read_jsonl, write_json

ID_DIM = 512
SEQ_FPS = 5.0                       # MultiCamCows tracklets are sampled at 5 Hz


# ------------------------------------------------------------------ model

def build(in_dim, d=256, layers=3, heads=4, dropout=0.1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from model import time_embedding

    class IDNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.kw = dict(in_dim=in_dim, d=d, layers=layers, heads=heads, dropout=dropout)
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, d))
            self.time = nn.Linear(d, d)
            self.d = d
            block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                               activation="gelu")
            self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.quality = nn.Linear(d, 1)
            self.id = nn.Linear(d, ID_DIM)

        def forward(self, x, t, valid):
            h = self.proj(x) + self.time(time_embedding(t, self.d))
            h = self.norm(self.encoder(h, src_key_padding_mask=~valid))
            q = self.quality(h).squeeze(-1).masked_fill(~valid, -1e4)
            w = q.softmax(-1)
            ids = F.normalize(self.id(h), dim=-1)
            fp = F.normalize((w.unsqueeze(-1) * ids).sum(1), dim=-1)
            return {"fingerprint": fp, "quality": q, "weights": w}

    return IDNet()


def load(path, device=None):
    import torch
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    m = build(**ck["kw"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def embed(model, seqs, device, max_len=32, batch=64):
    """[(feats (T, D) fp16, times (T,))] -> fingerprints (n, 512), quality max / mean (n,)."""
    import torch
    fps, qmax, qmean = [], [], []
    for s in range(0, len(seqs), batch):
        part = seqs[s:s + batch]
        L = min(max_len, max(len(f) for f, _ in part))
        X = np.zeros((len(part), L, part[0][0].shape[1]), np.float32)
        T = np.zeros((len(part), L), np.float32)
        V = np.zeros((len(part), L), bool)
        for i, (f, t) in enumerate(part):
            idx = np.linspace(0, len(f) - 1, min(L, len(f))).round().astype(int)
            X[i, :len(idx)], T[i, :len(idx)], V[i, :len(idx)] = f[idx], np.asarray(t)[idx] - t[idx[0]], True
        with torch.no_grad():
            o = model(torch.from_numpy(X).to(device), torch.from_numpy(T).to(device),
                      torch.from_numpy(V).to(device))
        fps.append(o["fingerprint"].float().cpu().numpy())
        q = o["quality"].float().cpu().numpy()
        for i in range(len(part)):
            qi = q[i][V[i]]
            qmax.append(float(qi.max())), qmean.append(float(qi.mean()))
    return (np.concatenate(fps) if fps else np.zeros((0, ID_DIM), np.float32)), np.array(qmax), np.array(qmean)


# ------------------------------------------------------------------ data

class Store:
    """The crop vectors of one feature folder (data_mcc.py): rows + one array, grouped into sequences."""

    def __init__(self, folder):
        self.rows = read_jsonl(os.path.join(folder, "crops.jsonl"))
        self.feats = np.load(os.path.join(folder, "feats.npy"), mmap_mode="r")
        seqs = collections.defaultdict(list)
        for i, r in enumerate(self.rows):
            seqs[(r["day"], r["cow"], r["seq"])].append(i)
        self.seqs = {}
        for k, idx in seqs.items():
            idx.sort(key=lambda i: (self.rows[i]["t"], self.rows[i]["name"]))
            self.seqs[k] = idx

    def days(self):
        return sorted({k[0] for k in self.seqs})

    def of_days(self, days):
        return {k: v for k, v in self.seqs.items() if k[0] in set(days)}

    def seq(self, idx, rng=None, length=None, step=1):
        """A window of a sequence -> (feats (T, D) float32, times (T,) seconds)."""
        if rng is not None and length:
            span = (length - 1) * step + 1
            if len(idx) > span:
                s = rng.randrange(0, len(idx) - span + 1)
                idx = idx[s:s + span:step]
            else:
                idx = idx[::max(1, min(step, len(idx) // max(1, length)))][:length]
        f = np.asarray(self.feats[idx], np.float32)
        t = np.array([self.rows[i]["t"] for i in idx], np.float32)
        return f, t - t[0]

    def whole(self, keys, max_len=32):
        out = []
        for k in keys:
            idx = self.seqs[k]
            if len(idx) > max_len:
                idx = [idx[i] for i in np.linspace(0, len(idx) - 1, max_len).round().astype(int)]
            out.append(self.seq(idx))
        return out


# ------------------------------------------------------------------ album and scores

def album(fps, cows, k_max=6):
    """Fingerprints of known sequences -> prototypes (k-means per cow) and their cow."""
    from gallery import spherical_kmeans
    by = collections.defaultdict(list)
    for f, c in zip(fps, cows):
        by[c].append(f)
    P, ids = [], []
    for c, fs in by.items():
        X = np.stack(fs)
        k = min(k_max, max(1, len(fs) // 5))
        for p in spherical_kmeans(X, k):
            P.append(p), ids.append(c)
    return np.stack(P).astype(np.float32), np.array(ids)


def match_rows(fps, protos, proto_ids, qmax, qmean, area=0.25):
    """Each query against the album -> best cow, similarity, margin to the next cow + NaN features."""
    sim = fps @ protos.T
    cows = sorted(set(proto_ids.tolist()))
    col = {c: np.where(proto_ids == c)[0] for c in cows}
    per_cow = np.stack([sim[:, col[c]].max(1) for c in cows], 1)
    order = np.argsort(-per_cow, 1)
    rows = []
    for i in range(len(fps)):
        b = order[i, 0]
        s2 = per_cow[i, order[i, 1]] if per_cow.shape[1] > 1 else -1.0
        rows.append({"cow": cows[b], "sim": float(per_cow[i, b]), "margin": float(per_cow[i, b] - s2),
                     "quality_max": float(qmax[i]), "quality_mean": float(qmean[i]), "area": float(area)})
    return rows


def fit_nan(rows, truth, max_error=0.01):
    import abstain
    y = np.array([r["cow"] == t for r, t in zip(rows, truth)], np.float64)
    X = abstain.feature_matrix(rows)
    m = abstain.fit_logistic(X, y)
    p = abstain.predict(m, X)
    return {**m, "features": list(abstain.FEATURES), "threshold": abstain.cutoff(p, y, max_error),
            "max_error": max_error, "fit_queries": int(len(y)), "fit_top1": float(y.mean()) if len(y) else None}


def nan_p(nan, rows):
    import abstain
    if not rows:
        return np.zeros(0)
    return abstain.predict(nan, abstain.feature_matrix(rows, tuple(nan["features"])))


def enrol_threshold(fps, cows):
    """The similarity at which two sequences are taken as one cow when the album enrols
    unknown cows: halfway between same-cow pairs (their 10th percentile) and different-cow
    pairs (their 99th percentile), on held-out data."""
    S = fps @ fps.T
    same = np.equal.outer(cows, cows)
    iu = np.triu_indices(len(cows), 1)
    s_same, s_diff = S[iu][same[iu]], S[iu][~same[iu]]
    if not len(s_same) or not len(s_diff):
        return 0.6
    lo, hi = float(np.percentile(s_diff, 99)), float(np.percentile(s_same, 10))
    return round((lo + hi) / 2, 4)


def score_days(model, store, device, album_days, query_days, nan=None):
    return score_keys(model, store, device, list(store.of_days(album_days)), list(store.of_days(query_days)))


def score_keys(model, store, device, a_keys, q_keys):
    a_fp, _, _ = embed(model, store.whole(a_keys), device)
    q_fp, qmax, qmean = embed(model, store.whole(q_keys), device)
    import metrics
    P, ids = album(a_fp, [k[1] for k in a_keys])
    q_ids = [k[1] for k in q_keys]
    known = set(ids.tolist())
    keep = [i for i, c in enumerate(q_ids) if c in known]             # a cow absent from the album cannot be found
    res = metrics.retrieval(q_fp[keep], [q_ids[i] for i in keep], P, ids)
    rows = match_rows(q_fp[keep], P, ids, qmax[keep], qmean[keep])
    res["queries_not_in_album"] = len(q_ids) - len(keep)
    return res, rows, [q_ids[i] for i in keep], q_fp, q_ids


# ------------------------------------------------------------------ splits

def split_plan(store, args):
    """Which sequences train, pick the epoch (val) and test - as album / query key lists.

    days: train = all days but the last two; val = album last train day, queries the day after;
          test = album that day, queries the last day (MultiCamCows2024).
    cows: the store has two parts of the day ("A" earlier, "B" later); cows are split at random
          (split_seed) into train / val / test; val and test cows: album from A, queries from B -
          cows the model never saw (MmCows, one barn)."""
    keys = sorted(store.seqs, key=lambda k: (k[0], str(k[1]), k[2]))
    days = store.days()
    if args.split == "groups":
        # v14 (D): the parts are named in the rows (data_wr10k.py combine): train, valA/valB (held-out
        # cows), testA/testB (MmCows' cows, never trained on)
        need = {"train", "valA", "valB", "testA", "testB"}
        if not need <= set(days):
            sys.exit(f"--split groups needs parts {sorted(need)}, found {days}")
        of = lambda d: [k for k in keys if k[0] == d]
        cows = lambda d: sorted({k[1] for k in of(d)}, key=str)
        print(f"[idnet] groups: {len(cows('train'))} training cows, {len(cows('valA'))} val cows, "
              f"{len(cows('testA'))} test cows (never trained on)", flush=True)
        return {"train": of("train"), "val_album": of("valA"), "val_query": of("valB"),
                "test_album": of("testA"), "test_query": of("testB"),
                "about": {"by": "groups", "train": cows("train"), "val": cows("valA"), "test": cows("testA"),
                          "args": {"split": "groups"}}}
    if args.split == "days":
        if len(days) < 3:
            sys.exit(f"need at least 3 days, found {days}")
        tr_d, val_d, test_d = days[:-2], days[-2], days[-1]
        pick = lambda ds: [k for k in keys if k[0] in set(ds)]
        print(f"[idnet] days: train {tr_d}, val {val_d}, test {test_d}", flush=True)
        return {"train": pick(tr_d), "val_album": pick(tr_d[-1:]), "val_query": pick([val_d]),
                "test_album": pick([val_d]), "test_query": pick([test_d]),
                "about": {"by": "days", "train": tr_d, "val": val_d, "test": test_d, "args": {"split": "days"}}}
    if len(days) != 2:
        sys.exit(f"--split cows needs two parts of the day (A, B), found {days}")
    cows = sorted({k[1] for k in keys})
    r = random.Random(args.split_seed)
    shuffled = cows[:]
    r.shuffle(shuffled)
    n_te, n_va = args.test_cows, args.val_cows
    if len(cows) < n_te + n_va + 2:
        sys.exit(f"--split cows: {len(cows)} cows, too few for {n_va} val + {n_te} test")
    test_c, val_c, train_c = sorted(shuffled[:n_te]), sorted(shuffled[n_te:n_te + n_va]), sorted(shuffled[n_te + n_va:])
    print(f"[idnet] cows: train {train_c}, val {val_c}, test {test_c} (never trained on)", flush=True)
    of = lambda cs, part=None: [k for k in keys if k[1] in set(cs) and (part is None or k[0] == part)]
    a, b = days
    return {"train": of(train_c), "val_album": of(val_c, a), "val_query": of(val_c, b),
            "test_album": of(test_c, a), "test_query": of(test_c, b),
            "about": {"by": "cows", "train": train_c, "val": val_c, "test": test_c, "album_part": a, "query_part": b,
                      "args": {"split": "cows", "val_cows": n_va, "test_cows": n_te, "split_seed": args.split_seed}}}


# ------------------------------------------------------------------ train

def cmd_train(args):
    import torch
    from model import supcon
    from common import device as dev_of
    dev = dev_of()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    store = Store(args.feats)
    plan = split_plan(store, args)
    tr = {k: store.seqs[k] for k in plan["train"]}
    by_cow = collections.defaultdict(list)
    for k in tr:
        if len(tr[k]) >= 2:
            by_cow[k[1]].append(k)
    cows = [c for c, ks in by_cow.items() if len(ks) >= 2]
    # batches from one source set at a time (--pure-batches): a cow in two sets under two names never
    # meets herself as a "different cow"
    ds_of = {c: store.rows[tr[by_cow[c][0]][0]].get("dataset") for c in cows}
    pools = collections.defaultdict(list)
    for c in cows:
        pools[ds_of[c]].append(c)
    pure = bool(getattr(args, "pure_batches", 1)) and len(pools) > 1
    if pure:
        print(f"[idnet] batches from one set at a time: {({k: len(v) for k, v in pools.items()})}", flush=True)
    print(f"[idnet] {len(tr)} train sequences, {len(cows)} cows with >= 2 sequences", flush=True)
    model = build(store.feats.shape[1], d=args.d, layers=args.layers, heads=args.heads, dropout=args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[idnet] {n_par / 1e6:.2f} M parameters, from scratch", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(tr) // (args.P * 2))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    os.makedirs(args.out, exist_ok=True)
    best, since, hist = -1.0, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            if pure:
                ds = rng.choices(list(pools), weights=[len(v) for v in pools.values()])[0]
                pick = rng.sample(pools[ds], min(args.P, len(pools[ds])))
            else:
                pick = rng.sample(cows, min(args.P, len(cows)))
            seqs, labels = [], []
            for ci, c in enumerate(pick):
                ks = by_cow[c]
                a = rng.choice(ks)
                other = [k for k in ks if k != a and (k[0] != a[0])] or [k for k in ks if k != a]
                b = rng.choice(other)
                for k in (a, b):
                    L = rng.randint(args.min_len, args.max_len)
                    f, t = store.seq(tr[k], rng, L, rng.choice((1, 1, 2, 3, 5)))   # 5, 2.5, 1.7, 1 Hz
                    if args.time_aug < 1 or getattr(args, "time_aug_max", 1.0) > 1:   # the gaps between frames, stretched or squeezed
                        t = t * float(np.exp(rng.uniform(np.log(args.time_aug), np.log(getattr(args, "time_aug_max", 1.0)))))
                    keep = np.array([rng.random() > args.drop for _ in range(len(f))]) if len(f) > 2 else np.ones(len(f), bool)
                    keep[rng.randrange(len(f))] = True
                    seqs.append((f[keep], t[keep] - t[keep][0])), labels.append(ci)
            L = max(len(f) for f, _ in seqs)
            X = np.zeros((len(seqs), L, seqs[0][0].shape[1]), np.float32)
            T = np.zeros((len(seqs), L), np.float32)
            V = np.zeros((len(seqs), L), bool)
            for i, (f, t) in enumerate(seqs):
                X[i, :len(f)], T[i, :len(f)], V[i, :len(f)] = f, t, True
            X += np.random.default_rng(rng.randrange(1 << 30)).normal(0, args.noise, X.shape).astype(np.float32) * V[..., None]
            o = model(torch.from_numpy(X).to(dev), torch.from_numpy(T).to(dev), torch.from_numpy(V).to(dev))
            loss = supcon(o["fingerprint"], torch.tensor(labels, device=dev), args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        res, _, _, _, _ = score_keys(model, store, dev, plan["val_album"], plan["val_query"])
        score = (res["top1"] or 0) + (res["mAP"] or 0)
        rec = {"epoch": ep + 1, "loss": round(float(np.mean(losses)), 4), "val_top1": res["top1"], "val_mAP": res["mAP"],
               "minutes": round((time.time() - t0) / 60, 1)}
        hist.append(rec)
        print(f"[idnet] epoch {ep + 1}: loss {rec['loss']:.3f}  val top-1 {res['top1']:.1%}  mAP {res['mAP']:.3f}", flush=True)
        if score > best:
            best, since = score, 0
            torch.save({"kw": model.kw, "state": model.state_dict(), "epoch": ep + 1, "split": plan["about"],
                        "feats": read_meta(args.feats)}, os.path.join(args.out, "idnet.pt"))
        else:
            since += 1
            if since >= args.patience:
                print(f"[idnet] no better val score for {args.patience} epochs - stopping", flush=True)
                break
    write_json(os.path.join(args.out, "history.json"), hist)
    cmd_eval(args)


def read_meta(folder):
    import json
    p = os.path.join(folder, "meta.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def cmd_eval(args):
    import torch
    from common import device as dev_of
    dev = dev_of()
    model, ck = load(os.path.join(args.out, "idnet.pt"), dev)
    model.to(dev)
    store = Store(args.feats)
    plan = split_plan(store, argparse.Namespace(**{**vars(args), **ck["split"]["args"]}))
    # the "not sure" model and the enrolment threshold, on the val part
    vres, vrows, vtruth, vfp, vids = score_keys(model, store, dev, plan["val_album"], plan["val_query"])
    nan = fit_nan(vrows, vtruth, args.max_error)
    thr = enrol_threshold(vfp, np.array(vids))
    # the test: a later day (days) or cows the model never saw (cows)
    tres, trows, ttruth, _, _ = score_keys(model, store, dev, plan["test_album"], plan["test_query"])
    p = nan_p(nan, trows)
    import metrics
    pred = [r["cow"] if pi >= nan["threshold"] else None for r, pi in zip(trows, p)]
    prf = metrics.id_prf(pred, ttruth)
    always = metrics.id_prf([r["cow"] for r in trows], ttruth)
    ck.update({"nan": nan, "enrol_threshold": thr})
    torch.save(ck, os.path.join(args.out, "idnet.pt"))
    out = {"model": {"epoch": ck["epoch"], "parameters": int(sum(v.numel() for v in ck["state"].values())), **ck["kw"]},
           "split": ck["split"], "val": vres, "test": tres,
           "test_id_with_not_sure": prf, "test_id_always_answer": always,
           "nan": {k: nan[k] for k in ("threshold", "max_error", "fit_queries", "fit_top1")},
           "enrol_threshold": thr,
           "what": "album = sequences of the earlier part (k-means looks per cow), queries = every sequence "
                   "of the later part; precision / recall / F1 count 'not sure' as a miss"}
    write_json(os.path.join(args.out, "eval_id.json"), out)
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    print(f"[idnet] test ({ck['split']['test']}): top-1 {pct(tres['top1'])}, top-5 {pct(tres['top5'])}, "
          f"mAP {tres['mAP'] if tres['mAP'] is None else round(tres['mAP'], 3)}", flush=True)
    print(f"[idnet] with 'not sure' (<= {args.max_error:.0%} wrong on val): precision {pct(prf['precision'])}, "
          f"recall {pct(prf['recall'])}, F1 {pct(prf['f1'])}, answered {pct(prf['answered_share'])}", flush=True)
    print(f"[idnet] enrolment threshold {thr}; -> {os.path.join(args.out, 'eval_id.json')}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--feats", required=True, help="data_mcc.py's or data_mmcows.py idcrops' feature folder")
        s.add_argument("--out", required=True)
        s.add_argument("--max-error", type=float, default=0.01)
        s.add_argument("--split", choices=("days", "cows", "groups"), default="days",
                       help="days: train on early days, test on the last (MultiCamCows); cows: one barn, "
                            "train on some cows, test on others never seen (MmCows)")
        s.add_argument("--val-cows", type=int, default=3)
        s.add_argument("--test-cows", type=int, default=5)
        s.add_argument("--split-seed", type=int, default=0)
        if name == "train":
            s.add_argument("--epochs", type=int, default=60)
            s.add_argument("--patience", type=int, default=10)
            s.add_argument("--P", type=int, default=32, help="cows per batch (2 sequences each)")
            s.add_argument("--min-len", type=int, default=4)
            s.add_argument("--max-len", type=int, default=24)
            s.add_argument("--drop", type=float, default=0.15, help="share of frames dropped at random")
            s.add_argument("--noise", type=float, default=0.02, help="noise on the frame vectors")
            s.add_argument("--d", type=int, default=256)
            s.add_argument("--layers", type=int, default=3)
            s.add_argument("--heads", type=int, default=4)
            s.add_argument("--dropout", type=float, default=0.1)
            s.add_argument("--lr", type=float, default=3e-4)
            s.add_argument("--wd", type=float, default=0.05)
            s.add_argument("--temperature", type=float, default=0.1)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--time-aug", type=float, default=1.0,
                           help="< 1: sequence times squeezed by up to this factor at random")
            s.add_argument("--time-aug-max", type=float, default=1.0, help="> 1: ... or stretched up to this factor")
            s.add_argument("--pure-batches", type=int, default=1, help="1: each batch from one source set")
    args = p.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
