#!/usr/bin/env python3
"""Identity on public sets with real cow ids: the fingerprint taught to know a cow across days and views.

The herd model's fingerprint was taught on CBVD-5, which has no cow ids: it learnt "the same cow within
7 s" and knows the view, not the cow (MmCows: a cow seen by two cameras got one track by look in
0.1-1.4% of cases). Three public sets name their cows:

    multicam   MultiCamCows2024 (Yu et al., University of Bristol, data.bris 2inu67jru7a6821kkgehxg3cv2,
               Non-Commercial Government Licence): 90 Holstein-Friesians, 3 ceiling cameras over a
               walkway, 7 days, 101,329 crops - one folder per cow per day, the same number every day,
               file <frame>_<camera>.jpg (tracklets sampled at 5 Hz). Crops only (3.6 GB of 36.5 GB).
    cows2021   Cows2021 (Gao et al., Bristol, data.bris 4vnrca7qw1642qlwxjadp87h7, NCGL): its
               identification test set - 181 cows, ~8.7k top-down crops over 4 weeks (the training part
               is tracklets without ids and is not used).
    opencows   OpenCows2020 (Andrew et al., Bristol, NCGL): its identification part, 46 cows, ~4.9k
               top-down crops.

    fetch    the crops, read in place from the zips (HTTP Range) -> DATA/raw/<source>/<cow>/...
    extract  (GPU) DINOv2 on every crop, as the barn does (square crop with the model's margin, its
             encoder and grid), then pseudo-bursts: a camera's 7 s of one cow (35 crops at 5 Hz) or a
             handful of one cow's pictures of one day, stretched to the barn's 175 frames, each with
             its cow's id -> DATA/feat/<source>/{train,val} in train.py's format. val: a sixth of each
             set's cows, never seen in training (the barn's cows are new cows too).
    train    a copy of the herd model's temporal transformer taught to put one cow's bursts together
             (supervised contrastive over cow ids, different days and cameras are positives), a
             quarter of the views cut to a single frame (mct.py tracks by one frame a second).
             Scored on the val cows: the same cow on another day, from another camera, and from one
             frame -> DATA/id_model/identity.pt, eval.json, and abstain.json (the NaN model refitted
             on these fingerprints). The herd model itself (posture, activity, rumination) is kept.

Then [model] identity = ".../identity.pt" in barn.toml: the gallery and mct.py use its fingerprint,
the gallery its abstain.json. Licences: all three are non-commercial.

    python herd/reid.py fetch   --data /workspace/reid
    python herd/reid.py extract --data /workspace/reid --model /workspace/herd/run9/model.pt
    python herd/reid.py train   --data /workspace/reid --model /workspace/herd/run9/model.pt
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import copy
import json
import math
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import read_jsonl, write_json  # noqa: E402

SOURCES = {
    "multicam": {"url": "https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip",
                 "licence": "Non-Commercial Government Licence", "cite": "Yu et al., MultiCamCows2024, arXiv 2410.12695"},
    "cows2021": {"url": "https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip",
                 "licence": "Non-Commercial Government Licence", "cite": "Gao et al., Cows2021, arXiv 2105.01938"},
    "opencows": {"url": None, "licence": "UK Non-Commercial Government Licence v2",
                 "cite": "Andrew et al., OpenCows2020"},          # the url is extra_det's (one zip for both)
}
SLOTS, FPS = 175, 25.0                   # the barn's burst: 7 s at 25 fps
WIN = 35                                 # MultiCamCows2024: 7 s of a tracklet at 5 Hz
PICS = 8                                 # stills of one cow on one day: a burst of up to this many


# ------------------------------------------------------------------ fetch

def plan(source, names):
    """Zip member names -> [(cow, day, cam, order, member)] of one source."""
    out = []
    for n in names:
        low = n.lower()
        if not low.endswith((".jpg", ".jpeg", ".png")):
            continue
        if source == "multicam":
            m = re.search(r"MultiCamCows2024Root/(2023\w{3}\d\d)/(\d{3})/(\d+)_(\d+)\.jpg$", n)
            if m:
                out.append((m.group(2), m.group(1), m.group(4), int(m.group(3)), n))
        elif source == "cows2021":
            m = re.search(r"/Identification/Test/(\d+)/image_(\d+)_(\d{4}-\d\d-\d\d)_", n)
            if m:
                out.append((m.group(1), m.group(3), "0", int(m.group(2)), n))
        elif source == "opencows":
            m = re.search(r"/identification/images/(?:train|test)/(\d+)/([^/]+)$", n)
            if m:
                out.append((m.group(1), "all", "0", len(out), n))
    return out


def cmd_fetch(args):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cowbench"))
    from extra_det import SOURCES as DET_SOURCES
    from extra_det import RemoteZip
    for src in args.sources:
        url = SOURCES[src]["url"] or DET_SOURCES["opencows"]["url"]
        folder = os.path.join(args.data, "raw", src)
        os.makedirs(folder, exist_ok=True)
        z = RemoteZip(url)
        items = plan(src, z.names())
        if args.max_per_source:
            items = items[:args.max_per_source]
        rows, todo = [], []
        for cow, day, cam, order, member in items:
            name = f"{cow}/{day}_{cam}_{order:07d}.jpg"
            rows.append({"cow": cow, "day": day, "cam": cam, "order": order, "file": name})
            if not os.path.exists(os.path.join(folder, name)):
                todo.append((member, name))
        print(f"[fetch] {src}: {len(rows)} crops of {len({r['cow'] for r in rows})} cows, {len(todo)} to fetch "
              f"({SOURCES[src]['licence']})", flush=True)

        def one(job):
            member, name = job
            path = os.path.join(folder, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path + ".part", "wb").write(z.read(member))
            os.replace(path + ".part", path)

        t0, done = time.time(), 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for _ in ex.map(one, todo):
                done += 1
                if done % 2000 == 0 or done == len(todo):
                    el = time.time() - t0
                    print(f"\r  {done}/{len(todo)}  {el / 60:.1f} min  ~{el / done * (len(todo) - done) / 60:.0f} min left",
                          end="", flush=True)
        print(flush=True)
        with open(os.path.join(folder, "index.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)


# ------------------------------------------------------------------ extract

def stretch(m, slots=SLOTS):
    """m frames over the burst's slots: slot j shows frame floor(j * m / slots)."""
    return np.minimum((np.arange(slots) * m) // slots, m - 1)


def make_bursts(rows, rng, val_cows, tracklets):
    """One source's crops (rows with "k", the crop's index into the features) -> pseudo-bursts
    [{"cow", "day", "cam", "frames": [k...], "split"}]: per cow, day and camera in time order, windows
    of WIN crops (7 s at 5 Hz) for tracklets; stills (no order: one picture per "order") in random
    groups of up to PICS of one day."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["cow"], r["day"], r["cam"])].append(r)
    out = []
    for (cow, day, cam), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["order"])
        size = WIN if tracklets else PICS
        if not tracklets:
            rng.shuffle(rs)
        chunks = [rs[i:i + size] for i in range(0, len(rs), size)]
        if len(chunks) > 1 and len(chunks[-1]) < max(2, size // 3):
            tail = chunks.pop()
            chunks[-1] += tail                               # a short tail joins the window before it
        for ch in chunks:
            if len(ch) >= 2:
                out.append({"cow": cow, "day": day, "cam": cam, "frames": [r["k"] for r in ch],
                            "split": "val" if cow in val_cows else "train"})
    return out


def write_split(folder, bursts, feats, src, meta):
    """Pseudo-bursts -> train.py's feature folder (index.jsonl, <clip>.npz, meta.json); a clip per cow."""
    os.makedirs(folder, exist_ok=True)
    by_cow = collections.defaultdict(list)
    for b in bursts:
        by_cow[b["cow"]].append(b)
    rows = []
    for clip, (cow, bs) in enumerate(sorted(by_cow.items())):
        bf = np.zeros((len(bs), SLOTS, feats.shape[1]), np.float16)
        bt = np.tile(np.arange(SLOTS, dtype=np.float32) / FPS, (len(bs), 1))
        for i, b in enumerate(bs):
            bf[i] = feats[np.asarray(b["frames"])[stretch(len(b["frames"]))]]
            rows.append({"kind": "burst", "clip": str(clip), "i": i, "identity": f"{src}:{cow}",
                         "track": f"{src}:{cow}:{b['day']}:{b['cam']}", "day": b["day"], "cam": b["cam"],
                         "posture": -1, "activity": -1, "rumination": -1, "area": 0.05, "uids": [],
                         "n_frames": len(b["frames"])})
        np.savez(os.path.join(folder, f"{clip}.npz"), burst_feats=bf, burst_times=bt,
                 key_feats=np.zeros((0, feats.shape[1]), np.float16))
        rows.append({"kind": "clip_done", "clip": str(clip)})
    with open(os.path.join(folder, "index.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    write_json(os.path.join(folder, "meta.json"), dict(meta, source=src))


def encode_all(paths, meta, batch=256):
    """Crops on disk -> the barn's frame vectors (fp16): each crop squared with the model's margin."""
    import torch
    from PIL import Image
    from common import crop, device
    from model import FrameEncoder
    dev = device()
    enc = FrameEncoder(meta["encoder"], meta["grid"]).to(dev).eval()
    load = lambda p: crop(np.asarray(Image.open(p).convert("RGB")), [0.0, 0.0, 1.0, 1.0], meta["crop"], meta["margin"])
    out, t0 = [], time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for i in range(0, len(paths), batch):
            x = torch.from_numpy(np.stack(list(ex.map(load, paths[i:i + batch])))).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(enc(x).float().cpu().numpy().astype(np.float16))
            done = min(i + batch, len(paths))
            if (i // batch) % 20 == 0 or done == len(paths):
                el = time.time() - t0
                print(f"\r  {done}/{len(paths)} crops  {el / 60:.1f} min  ~{el / done * (len(paths) - done) / 60:.0f} min left",
                      end="", flush=True)
    print(flush=True)
    return np.concatenate(out) if out else np.zeros((0, enc.out_dim), np.float16)


def cmd_extract(args):
    import torch
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    f = ck.get("features", {})
    meta = {"encoder": f.get("encoder", "facebook/dinov2-small"), "grid": f.get("grid", 2),
            "crop": f.get("crop", 224), "margin": f.get("margin", 0.1)}
    rng = random.Random(args.seed)
    for src in args.sources:
        raw = os.path.join(args.data, "raw", src)
        rows = read_jsonl(os.path.join(raw, "index.jsonl"))
        if not rows:
            print(f"[extract] {src}: nothing fetched", flush=True)
            continue
        cache = os.path.join(args.data, "feat", src, "crops.npz")
        if os.path.exists(cache) and json.load(open(cache + ".json")) == meta:
            feats = np.load(cache)["feats"]
        else:
            print(f"[extract] {src}: DINOv2 on {len(rows)} crops", flush=True)
            feats = encode_all([os.path.join(raw, r["file"]) for r in rows], meta)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, feats=feats)
            json.dump(meta, open(cache + ".json", "w"))
        for k, r in enumerate(rows):
            r["k"] = k
        cows = sorted({r["cow"] for r in rows})
        val = set(random.Random(args.seed).sample(cows, max(1, round(len(cows) * args.val_share))))
        bursts = make_bursts(rows, rng, val, tracklets=src == "multicam")
        for part in ("train", "val"):
            write_split(os.path.join(args.data, "feat", src, part), [b for b in bursts if b["split"] == part],
                        feats, src, dict(meta, dim=int(feats.shape[1])))
        n = collections.Counter(b["split"] for b in bursts)
        print(f"[extract] {src}: {len(cows)} cows ({len(val)} held out), {n['train']} train / {n['val']} val "
              f"bursts", flush=True)


# ------------------------------------------------------------------ train

def load_parts(data, sources, part):
    from train import Split
    out = None
    for src in sources:
        folder = os.path.join(data, "feat", src, part)
        if os.path.exists(os.path.join(folder, "index.jsonl")):
            s = Split(folder)
            out = s if out is None else out.extend(s)
    return out


def embed(model, split, single=False, batch=128):
    """Every burst whole (or its middle frame alone) -> fingerprints and the quality summary."""
    import torch
    from common import device
    dev = device()
    T = split.burst_feats.shape[1]
    fps, qmax, qmean = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(split.bursts), batch):
            x = torch.from_numpy(split.burst_feats[i:i + batch]).float().to(dev)
            if single:
                x = x[:, T // 2:T // 2 + 1]
            t = torch.arange(x.shape[1], device=dev).float().div(FPS).expand(len(x), -1)
            v = torch.ones(x.shape[:2], dtype=torch.bool, device=dev)
            o = model(x, t, v)
            fps.append(o["fingerprint"].float().cpu().numpy())
            qmax.append(o["weights"].max(1).values.float().cpu().numpy())
            qmean.append(o["quality"].float().mean(1).cpu().numpy())
    return np.concatenate(fps), np.concatenate(qmax), np.concatenate(qmean)


def reid_scores(split, fp, qfp=None, qmax=None, qmean=None):
    """Each burst (or its single frame: qfp) against the bursts of OTHER days (and, on its day, of other
    cameras): is the nearest one the same cow? -> metrics, and the rows the NaN model is fitted on."""
    qfp = fp if qfp is None else qfp
    ident = np.array([b["identity"] for b in split.bursts])
    day = np.array([b.get("day", "") for b in split.bursts])
    cam = np.array([b.get("cam", "") for b in split.bursts])
    src = np.array([i.split(":")[0] for i in ident])
    sim = qfp @ fp.T
    res, rows = {}, []
    for scope, allowed in (("other_day", day[None, :] != day[:, None]),
                           ("other_camera", (day[None, :] == day[:, None]) & (cam[None, :] != cam[:, None]))):
        allowed = allowed & (src[None, :] == src[:, None])       # within one set: its cows are its gallery
        s = np.where(allowed, sim, -2.0)
        askable = ((ident[None, :] == ident[:, None]) & allowed).any(1)
        if not askable.any():
            continue
        best = s.argmax(1)
        hit = ident[best] == ident
        res[f"{scope}_top1"] = float(hit[askable].mean())
        res[f"{scope}_queries"] = int(askable.sum())
        for s_ in sorted(set(src)):
            m = askable & (src == s_)
            if m.any():
                res[f"{scope}_top1_{s_}"] = float(hit[m].mean())
        if scope == "other_day":
            for q in np.where(askable)[0]:
                b = best[q]
                other = s[q][ident != ident[b]]
                second = float(other.max()) if other.size and other.max() > -2 else -1.0
                rows.append({"sim": float(s[q, b]), "margin": float(s[q, b]) - second, "correct": int(hit[q]),
                             "quality_max": float(qmax[q]) if qmax is not None else 0.0,
                             "quality_mean": float(qmean[q]) if qmean is not None else 0.0, "area": 0.05})
    return res, rows


def evaluate(model, split):
    fp, qmax, qmean = embed(model, split)
    m, rows = reid_scores(split, fp, qmax=qmax, qmean=qmean)
    one, _, _ = embed(model, split, single=True)
    m1, _ = reid_scores(split, fp, qfp=one)
    m.update({f"frame_{k}": v for k, v in m1.items()})
    m["score"] = float(np.mean([m.get("other_day_top1", 0.0), m.get("frame_other_day_top1", 0.0)]))
    return m, rows


def fit_abstain(rows, max_error, out_path):
    """The NaN model on these fingerprints: abstain.py's logistic over (sim, margin, quality, size)."""
    import abstain
    X = abstain.feature_matrix(rows)
    y = np.array([r["correct"] for r in rows], float)
    m = abstain.fit_logistic(X, y)
    p = abstain.predict(m, X)
    budgets = sorted({max_error, 0.01, 0.02, 0.05})
    th = {e: abstain.cutoff(p, y, e) for e in budgets}
    same = np.array([r["sim"] for r in rows if r["correct"]])
    out = {"features": list(abstain.FEATURES), **m, "max_error": max_error, "threshold": th[max_error],
           "scope": "all", "dev_queries": len(rows), "dev_top1": float(y.mean()),
           "same_cow_sim_p05": float(np.percentile(same, 5)) if len(same) else None,
           "fitted_on": "reid.py val cows, other days",
           "dev": abstain.coverage_table(p, y, [(f"{e:.0%}", th[e]) for e in budgets])}
    write_json(out_path, out)
    return out


def cmd_train(args):
    import torch
    import train as T
    from common import device
    from model import HerdModel, supcon
    base, ck = HerdModel.load(args.model, map_location=device())
    tr = load_parts(args.data, args.sources, "train")
    va = load_parts(args.data, args.sources, "val")
    if tr is None or va is None:
        sys.exit("no features: reid.py extract first")
    if tr.dim != base.in_dim:
        sys.exit(f"features of {tr.dim} dims, the model takes {base.in_dim}: extract with this model")
    print(f"[train] {len(tr.bursts)} bursts of {len(set(tr.identity.tolist()))} cows; val {len(va.bursts)} bursts "
          f"of {len({b['identity'] for b in va.bursts})} unseen cows", flush=True)
    dev = device()
    model = copy.deepcopy(base.temporal).to(dev)
    m0, _ = evaluate(model, va)
    print(f"[train] before: same cow another day {m0.get('other_day_top1', 0):.1%}, another camera "
          f"{m0.get('other_camera_top1', 0):.1%}, one frame vs another day {m0.get('frame_other_day_top1', 0):.1%}",
          flush=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, len(tr.bursts) // args.P)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.05)
    sampler = T.identity_sampler(tr, args.P, rng, K=args.K)
    os.makedirs(args.out, exist_ok=True)
    best, best_state, best_m, since = -1.0, None, None, 0
    hist = open(os.path.join(args.out, "history.jsonl"), "w", encoding="utf-8")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(steps):
            ids = next(sampler)
            x, t, v = T.burst_batch(tr, ids, rng, args.crop_s, max_crop_s=7.0)
            for r in range(len(ids)):              # some views one frame only: mct.py's once a second
                if rng.random() < args.single_p:
                    on = torch.nonzero(v[r]).flatten()
                    keep = on[rng.randrange(len(on))]
                    v[r] = False
                    v[r, keep] = True
            o = model(x, t, v)
            loss = supcon(o["fingerprint"], torch.as_tensor(tr.identity[ids], device=dev), args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        m, _ = evaluate(model, va)
        hist.write(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses)), **m}) + "\n")
        hist.flush()
        print(f"[train] epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.3f}  val: another day "
              f"{m.get('other_day_top1', 0):.1%}, another camera {m.get('other_camera_top1', 0):.1%}, one frame "
              f"{m.get('frame_other_day_top1', 0):.1%}", flush=True)
        since += 1
        if m["score"] > best:
            best, best_m, since = m["score"], dict(m, epoch=epoch + 1), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if since >= args.patience:
            break
    hist.close()
    model.load_state_dict(best_state)
    _, rows = evaluate(model, va)
    ab = fit_abstain(rows, args.max_error, os.path.join(args.out, "abstain.json"))
    torch.save({"in_dim": base.in_dim, "kw": {k: v for k, v in base.kw.items() if k != "frame_id"},
                "state": {k: v.cpu() for k, v in model.state_dict().items()}, "features": ck.get("features", {}),
                "base": os.path.abspath(args.model), "sources": args.sources, "eval": best_m, "before": m0},
               os.path.join(args.out, "identity.pt"))
    write_json(os.path.join(args.out, "eval.json"), {"before": m0, "after": best_m,
                                                     "abstain": {k: ab[k] for k in ("threshold", "dev_top1", "dev")}})
    print(f"[train] best epoch {best_m['epoch']}: another day {best_m.get('other_day_top1', 0):.1%} (was "
          f"{m0.get('other_day_top1', 0):.1%}), another camera {best_m.get('other_camera_top1', 0):.1%} (was "
          f"{m0.get('other_camera_top1', 0):.1%}), one frame {best_m.get('frame_other_day_top1', 0):.1%} (was "
          f"{m0.get('frame_other_day_top1', 0):.1%})", flush=True)
    for r in ab["dev"]:
        e = r["error_when_answered"]
        print(f"  NaN model: budget {r['budget']:>4}  answers {r['answered']:.1%}  wrong when answering "
              f"{'-' if e is None else f'{e:.1%}'}")
    print(f"-> {os.path.join(args.out, 'identity.pt')} (+ abstain.json): [model] identity in barn.toml")


def load_identity(path, features_meta, dev):
    """identity.pt -> (TemporalModel on dev, its abstain.json or None); checks it reads the same vectors."""
    import torch
    from model import TemporalModel
    ck = torch.load(path, map_location=dev, weights_only=False)
    want = {k: ck.get("features", {}).get(k) for k in ("encoder", "grid", "crop", "margin")}
    have = {k: features_meta.get(k) for k in want}
    if want != have:
        raise SystemExit(f"{path} reads frame vectors {want}, the herd model's are {have}")
    m = TemporalModel(ck["in_dim"], **ck["kw"]).to(dev).eval()
    m.load_state_dict(ck["state"])
    ab = os.path.join(os.path.dirname(os.path.abspath(path)), "abstain.json")
    return m, (json.load(open(ab, encoding="utf-8")) if os.path.exists(ab) else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    e = sub.add_parser("extract")
    t = sub.add_parser("train")
    for q in (f, e, t):
        q.add_argument("--data", required=True)
        q.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    f.add_argument("--workers", type=int, default=48)
    f.add_argument("--max-per-source", type=int, default=0, help="a quick look: only this many crops a set")
    for q in (e, t):
        q.add_argument("--model", required=True, help="the herd model (its encoder settings; train starts from it)")
        q.add_argument("--seed", type=int, default=0)
    e.add_argument("--val-share", type=float, default=1 / 6, help="cows held out of training, per set")
    t.add_argument("--out", default=None, help="default DATA/id_model")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--P", type=int, default=64, help="bursts a step")
    t.add_argument("--K", type=int, default=4, help="bursts of each cow in a step")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--temperature", type=float, default=0.1)
    t.add_argument("--crop-s", type=float, default=2.0, help="the shortest view (s); up to 5.6 s")
    t.add_argument("--single-p", type=float, default=0.25, help="share of views cut to one frame")
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--max-error", type=float, default=0.01, help="the NaN model's error budget")
    args = p.parse_args(argv)
    if args.cmd == "train" and args.out is None:
        args.out = os.path.join(args.data, "id_model")
    {"fetch": cmd_fetch, "extract": cmd_extract, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
