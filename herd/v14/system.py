#!/usr/bin/env python3
"""Onboarding a farm with inference only (v14): MmCows' 25 July as a farm the models never saw.

    setup   --start .. +--setup-min (default 12:00-13:00): the system watches all 4 cameras,
            follows every cow across them (association transformer), takes each track's
            fingerprint once a minute (ID transformer) and, at the end, enrols the album:
            tracks grouped by fingerprint, a group seen long enough = one cow (A01, A02 ..)
    test    the rest of the window (13:00-15:00): every minute each track's fingerprint is
            compared with the album, a scoreboard per track keeps a running score per cow,
            and the ID desk names the tracks - one cow, one track; "not sure" (the NaN
            model from MultiCamCows' val day) when the evidence is too weak

Nothing is trained here. Scored against MmCows' labelled frames (cow id on every box, every
15 s, 4 cameras): detector recall / precision, IDF1 and track continuity (within a camera
and across a camera change), the album (purity, coverage, duplicates), and identification
precision / recall / F1 with "not sure" counted as a miss. The album's cows are named after
the true cows by the setup hour alone (as a farmer would read the ear tags once).

--assoc none runs the same with each camera tracked on its own by box overlap, for comparison.

    python herd/v14/system.py --data /workspace/v14/mmcows --detector <best/> \\
        --idnet /workspace/v14/idnet/idnet.pt --assoc /workspace/v14/assoc/assoc.pt --out /workspace/v14/system
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

import v14common as C
from v14common import iou, local_hour, read_jsonl, write_json


# ------------------------------------------------------------------ tracks

class Tracks:
    def __init__(self, max_gap):
        self.max_gap = max_gap
        self.rows = {}                    # id -> {"sightings": [(t, cam, box, feat, area, conf)], "last": t}
        self.next = 1

    def new(self):
        tid = self.next
        self.next += 1
        self.rows[tid] = {"sightings": [], "last": None}
        return tid

    def add(self, tid, t, cam, box, feat, conf):
        area = (box[2] - box[0]) * (box[3] - box[1])
        self.rows[tid]["sightings"].append((t, cam, box, feat, area, conf))
        self.rows[tid]["last"] = t

    def active(self, t):
        return [k for k, r in self.rows.items() if r["last"] is not None and t - r["last"] <= self.max_gap]


def union_groups(n, pairs):
    parent = list(range(n))

    def f(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[f(a)] = f(b)
    g = collections.defaultdict(list)
    for i in range(n):
        g[f(i)].append(i)
    return list(g.values())


# ------------------------------------------------------------------ the run

def run(args):
    import torch
    from PIL import Image
    import assoc as A
    import idnet as I
    import metrics
    import scene
    from cbvd_bursts import encode
    from common import crop, device
    from detector import Live
    from model import FrameEncoder

    dev = device()
    live = read_jsonl(os.path.join(args.data, "live.jsonl"))
    if not live:
        sys.exit("no live frames: data_mmcows.py live first")
    masks = dict(np.load(os.path.join(args.data, "masks.npz"))) if os.path.exists(os.path.join(args.data, "masks.npz")) else {}
    t_start = min(r["ts"] for r in live if local_hour(r["ts"]) >= args.start)
    t_setup = t_start + args.setup_min * 60
    steps = collections.defaultdict(dict)
    for r in live:
        if args.start <= local_hour(r["ts"]) < args.end:
            steps[int(round((r["ts"] - t_start) / args.step))][r["cam"]] = r
    keys = sorted(steps)
    cams = sorted({c for s in steps.values() for c in s})
    slot_of = {c: i for i, c in enumerate(cams)}
    print(f"[system] {len(keys)} steps of {args.step} s, cameras {cams}, setup until "
          f"{local_hour(t_setup):.2f} h, assoc {'off' if args.assoc == 'none' else 'on'}", flush=True)

    det = Live(args.detector, threshold=args.det_threshold, calib=bool(args.calib))
    idm, idck = I.load(args.idnet, dev)
    idm.to(dev)
    nan, enrol_thr = idck.get("nan"), float(idck.get("enrol_threshold") or 0.6)
    meta = idck.get("feats") or {}
    enc = FrameEncoder(meta.get("encoder", "facebook/dinov2-small"), meta.get("grid", 2)).to(dev).eval()
    am = ack = None
    if args.assoc != "none":
        am, ack = A.load(args.assoc, dev)
        am.to(dev)
    pair_thr = float(args.pair_threshold or (ack or {}).get("pair_threshold") or 0.7)

    tracks = Tracks(args.max_gap)
    memory = collections.deque(maxlen=16)            # [(t, tok_a, tok_b)]
    scenes, scene_t = {}, -1e18
    det_log = []                                     # (t, cam, box, track, conf)
    minute_recs = []                                 # (minute, track, fp, qmax, qmean, area)
    last_minute = -1
    t0 = time.time()
    for n_step, k in enumerate(keys):
        t = t_start + k * args.step
        rows = steps[k]
        imgs = {c: Image.open(r["path"]).convert("RGB") for c, r in rows.items()}
        masked = {c: C.apply_mask(im, masks.get(c)) for c, im in imgs.items()}
        found = det.many([masked[c] for c in rows])
        cur = []                                     # {cam, box, conf, feat}
        crops = []
        for c, f in zip(rows, found):
            a = np.asarray(imgs[c])
            for d in f:
                cur.append({"cam": c, "box": d["bbox"], "conf": float(d["det_score"])})
                crops.append(crop(a, d["bbox"], meta.get("crop", 224), 0.1))
        feats = encode(enc, np.stack(crops)).astype(np.float32) if crops else np.zeros((0, enc.out_dim), np.float32)
        for d, f in zip(cur, feats):
            d["feat"] = f
        if am is not None and t - scene_t >= args.scene_every:
            g = scene.scene_grid(enc, [masked[c] for c in rows])
            scenes = {c: scene.sub_grid(gi) for c, gi in zip(rows, g)}
            scene_t = t
        # --- associate
        act = tracks.active(t)
        if am is None:
            assign = iou_assign(cur, act, tracks)
        else:
            assign = assoc_assign(am, cur, act, tracks, memory, scenes, slot_of, t, pair_thr, dev, args)
        for i, tid in enumerate(assign):
            if tid is None:
                tid = tracks.new()
                assign[i] = tid
            d = cur[i]
            tracks.add(tid, t, d["cam"], d["box"], d["feat"], d["conf"])
            det_log.append((t, d["cam"], d["box"], tid, d["conf"]))
        # --- once a minute: a fingerprint per track seen in the last minute
        minute = int((t - t_start) // 60)
        if minute != last_minute and last_minute >= 0:
            minute_recs += fingerprints(idm, tracks, t_start + last_minute * 60, t_start + minute * 60, last_minute,
                                        dev, args.min_frames)
        last_minute = minute
        if n_step % 50 == 0:
            print(f"\r  {n_step + 1}/{len(keys)} steps, {len(tracks.rows)} tracks, "
                  f"{(time.time() - t0) / 60:.1f} min", end="", flush=True)
    minute_recs += fingerprints(idm, tracks, t_start + last_minute * 60, t_start + (last_minute + 1) * 60,
                                last_minute, dev, args.min_frames)
    print(flush=True)

    # --- album from the setup minutes
    setup_m = args.setup_min
    alb_P, alb_ids, track_album = enrol([r for r in minute_recs if r[0] < setup_m], enrol_thr, args.min_minutes)
    print(f"[system] album: {len(set(alb_ids.tolist())) if len(alb_ids) else 0} cows from "
          f"{len({r[1] for r in minute_recs if r[0] < setup_m})} setup tracks (threshold {enrol_thr})", flush=True)
    # --- identification after setup: scoreboards + ID desk
    names = identify([r for r in minute_recs if r[0] >= setup_m], alb_P, alb_ids, nan, args.ema)

    # --- scoring against the labels
    sp = idck.get("split") or {}
    groups = ({"unseen_by_id_model (its test cows)": sp["test"], "id_model_val_cows": sp["val"],
               "id_model_train_cows": sp["train"]} if sp.get("by") == "cows" else None)
    res = score(args, det_log, track_album, names, t_start, t_setup, cams, metrics, groups)
    res["settings"] = {k: getattr(args, k) for k in ("step", "setup_min", "start", "end", "assoc", "det_threshold",
                                                     "calib", "max_gap", "min_frames", "min_minutes", "ema",
                                                     "scene_every")}
    res["settings"].update({"pair_threshold": pair_thr, "enrol_threshold": enrol_thr,
                            "nan_threshold": (nan or {}).get("threshold")})
    os.makedirs(args.out, exist_ok=True)
    tag = "assoc" if am is not None else "iou_per_camera"
    write_json(os.path.join(args.out, f"system_{tag}.json"), res)
    with open(os.path.join(args.out, f"tracks_{tag}.jsonl"), "w", encoding="utf-8") as fh:
        for tm, cam, box, tid, conf in det_log:
            fh.write(json.dumps({"t": tm, "cam": cam, "box": box, "track": tid, "conf": round(conf, 3)}) + "\n")
    report(res, tag)
    return res


def iou_assign(cur, act, tracks, thr=0.3):
    """Per camera, by box overlap with each track's last box on that camera (the baseline)."""
    import metrics
    out = [None] * len(cur)
    for cam in {d["cam"] for d in cur}:
        idx = [i for i, d in enumerate(cur) if d["cam"] == cam]
        tr = [(tid, tracks.rows[tid]["sightings"][-1][2]) for tid in act if tracks.rows[tid]["sightings"][-1][1] == cam]
        if not idx or not tr:
            continue
        M = np.array([[iou(cur[i]["box"], b) for _, b in tr] for i in idx])
        r, c = metrics.assignment(-M)
        for a, b in zip(r, c):
            if M[a, b] >= thr:
                out[idx[a]] = tr[b][0]
    return out


def assoc_assign(am, cur, act, tracks, memory, scenes, slot_of, t, pair_thr, dev, args):
    import torch
    import assoc as A
    import metrics
    import scene as S
    if not cur:
        return []
    dets = [{"feat": d["feat"], "box": d["box"], "conf": d["conf"], "slot": slot_of[d["cam"]], "kind": A.CUR}
            for d in cur]
    link_of = {}
    hist_idx = collections.defaultdict(list)
    for tid in sorted(act, key=lambda k: -tracks.rows[k]["last"])[:63]:
        link_of[tid] = 1 + len(link_of)
        for s in tracks.rows[tid]["sightings"][-args.history:]:
            hist_idx[tid].append(len(dets))
            dets.append({"feat": s[3], "box": s[2], "conf": s[5], "dt": s[0] - t, "slot": slot_of[s[1]],
                         "kind": A.HIST, "link": link_of[tid]})
    for j, (tm, a, b) in enumerate(list(memory)[-16:]):
        for s in (a, b):
            dets.append({"feat": s["feat"], "box": s["box"], "conf": 1.0, "dt": tm - t, "slot": slot_of[s["cam"]],
                         "kind": A.MEM, "link": 64 + j})
    cen = S.cell_centres()
    sc = [{"sfeat": g, "centres": cen, "cell": np.full((64, 2), 1 / 8, np.float32), "slot": slot_of[c]}
          for c, g in scenes.items() if c in slot_of]
    e = A.tokens(dets, sc, len(cur[0]["feat"]), next(iter(scenes.values())).shape[-1] if scenes else 384)
    with torch.no_grad():
        o = am(A.batch([e], dev))
    emb = o["emb"][0].float().cpu().numpy()
    n = len(cur)
    # groups: the same cow on several cameras now
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if cur[i]["cam"] != cur[j]["cam"] and float(emb[i] @ emb[j]) >= pair_thr]
    groups = union_groups(n, pairs)
    groups = [g for g in groups if len({cur[i]["cam"] for i in g}) == len(g)] + \
             [[i] for g in groups if len({cur[i]["cam"] for i in g}) != len(g) for i in g]
    # each group to a track: its best similarity to the track's history tokens
    tids = list(hist_idx)
    out = [None] * n
    if tids:
        M = np.array([[max(float(emb[i] @ emb[j]) for i in g for j in hist_idx[tid]) for tid in tids] for g in groups])
        r, c = metrics.assignment(-M)
        for a, b in zip(r, c):
            if M[a, b] >= pair_thr:
                for i in groups[a]:
                    out[i] = tids[b]
    for g in groups:                                   # a new cow seen by several cameras: one track
        if out[g[0]] is None and len(g) > 1:
            tid = tracks.new()
            for i in g:
                out[i] = tid
    for g in groups:                                   # confident cross-camera pairs -> the layout memory
        if len(g) >= 2:
            best = max(((i, j) for i in g for j in g if i < j), key=lambda p: float(emb[p[0]] @ emb[p[1]]))
            if float(emb[best[0]] @ emb[best[1]]) >= args.memory_threshold:
                memory.append((t, cur[best[0]], cur[best[1]]))
    return out


def fingerprints(idm, tracks, t0, t1, minute, dev, min_frames):
    """Each track seen in [t0, t1): its best view per moment (biggest box) -> one fingerprint."""
    import idnet as I
    seqs, meta = [], []
    for tid, r in tracks.rows.items():
        s = [x for x in r["sightings"] if t0 <= x[0] < t1]
        if len({x[0] for x in s}) < min_frames:
            continue
        best = {}
        for x in s:
            if x[0] not in best or x[4] > best[x[0]][4]:
                best[x[0]] = x
        xs = [best[k] for k in sorted(best)]
        seqs.append((np.stack([x[3] for x in xs]).astype(np.float32), np.array([x[0] - xs[0][0] for x in xs], np.float32)))
        meta.append((tid, float(np.mean([x[4] for x in xs]))))
    if not seqs:
        return []
    fp, qmax, qmean = I.embed(idm, seqs, dev, max_len=64)
    return [(minute, tid, fp[i], float(qmax[i]), float(qmean[i]), area) for i, (tid, area) in enumerate(meta)]


def enrol(recs, thr, min_minutes, k_max=6):
    """Setup fingerprints -> album: tracks grouped by their mean fingerprint (greedy, as gallery.enroll),
    a group with >= min_minutes fingerprints is one cow. -> prototypes, their cow, {track: album cow}."""
    from gallery import spherical_kmeans
    by = collections.defaultdict(list)
    for m, tid, fp, *_ in recs:
        by[tid].append(fp)
    units = sorted(((tid, fps) for tid, fps in by.items()), key=lambda u: -len(u[1]))
    groups = []                                        # [centre, [tids], [fps]]
    for tid, fps in units:
        c = np.mean(fps, 0)
        c /= np.linalg.norm(c) + 1e-9
        sims = [float(g[0] @ c) for g in groups]
        if sims and max(sims) >= thr:
            g = groups[int(np.argmax(sims))]
            g[1].append(tid), g[2].extend(fps)
            g[0] = np.mean(g[2], 0)
            g[0] /= np.linalg.norm(g[0]) + 1e-9
        else:
            groups.append([c, [tid], list(fps)])
    P, ids, track_album = [], [], {}
    n = 0
    for c, tids, fps in sorted(groups, key=lambda g: -len(g[2])):
        if len(fps) < min_minutes:
            continue
        n += 1
        name = f"A{n:02d}"
        X = np.stack(fps)
        for p in spherical_kmeans(X, min(k_max, max(1, len(fps) // 5))):
            P.append(p), ids.append(name)
        for tid in tids:
            track_album[tid] = name
    if not P:
        return np.zeros((0, 512), np.float32), np.array([]), {}
    return np.stack(P).astype(np.float32), np.array(ids), track_album


def identify(recs, P, ids, nan, ema):
    """Test minutes -> {(track, minute): album cow or None}: scoreboard per track (running
    score per cow), then each minute the ID desk names tracks strongest first, one cow once."""
    import idnet as I
    names = {}
    if not len(P):
        return names
    cows = sorted(set(ids.tolist()))
    col = {c: np.where(ids == c)[0] for c in cows}
    board = {}                                       # track -> running score per cow
    qual = collections.defaultdict(list)
    by_minute = collections.defaultdict(list)
    for r in recs:
        by_minute[r[0]].append(r)
    for m in sorted(by_minute):
        cand = []
        for _, tid, fp, qmax, qmean, area in by_minute[m]:
            sims = np.array([float((P[col[c]] @ fp).max()) for c in cows])
            board[tid] = sims if tid not in board else (1 - ema) * board[tid] + ema * sims
            qual[tid].append((qmax, qmean))
            s = board[tid]
            o = np.argsort(-s)
            q = np.mean(qual[tid], 0)
            row = {"cow": cows[o[0]], "sim": float(s[o[0]]), "margin": float(s[o[0]] - (s[o[1]] if len(o) > 1 else -1)),
                   "quality_max": float(q[0]), "quality_mean": float(q[1]), "area": float(area)}
            p = float(I.nan_p(nan, [row])[0]) if nan else 1.0
            cand.append((p, tid, row))
        taken = set()
        thr = (nan or {}).get("threshold", 0.0)
        for p, tid, row in sorted(cand, key=lambda x: -x[0]):
            if p >= thr and row["cow"] not in taken:
                names[(tid, m)] = row["cow"]
                taken.add(row["cow"])
            else:
                names[(tid, m)] = None
            names[("always", tid, m)] = row["cow"]
    return names


# ------------------------------------------------------------------ scores

def score(args, det_log, track_album, names, t_start, t_setup, cams, metrics, groups=None):
    labels = [r for r in read_jsonl(os.path.join(args.data, "labels.jsonl"))
              if args.start <= local_hour(r["ts"]) < args.end and r["cam"] in cams]
    by = collections.defaultdict(list)
    for tm, cam, box, tid, conf in det_log:
        by[(cam, int(round((tm - t_start) / args.step)))].append((box, tid, tm))
    gt_ids, tr_ids, n_pred, sight = [], [], 0, []
    setup_album, setup_gt = [], []
    id_pred, id_pred_always, id_truth = [], [], []
    det_tp = det_fn = det_fp = 0
    det_split = collections.Counter()
    true_cows = set()
    for r in labels:
        k = int(round((r["ts"] - t_start) / args.step))
        if abs(r["ts"] - (t_start + k * args.step)) > args.step / 2 + 0.6:
            continue
        dets = by.get((r["cam"], k), [])
        gts = r["boxes"]
        n_pred += len(dets)
        M = np.array([[iou(g[1], d[0]) for d in dets] for g in gts]) if dets and gts else np.zeros((len(gts), len(dets)))
        match = {}
        if M.size:
            a, b = metrics.assignment(-M)
            match = {i: j for i, j in zip(a, b) if M[i, j] >= args.match_iou}
        setup = r["ts"] < t_setup
        det_tp += len(match)
        det_fn += len(gts) - len(match)
        det_fp += len(dets) - len(match)
        det_split[("setup" if setup else "test", "tp")] += len(match)
        det_split[("setup" if setup else "test", "gt")] += len(gts)
        minute = int((r["ts"] - t_start) // 60)
        for i, (cow, _) in enumerate(gts):
            true_cows.add(cow)
            tid = dets[match[i]][1] if i in match else None
            gt_ids.append(cow), tr_ids.append(tid)
            sight.append((cow, r["ts"], r["cam"], tid))
            if setup:
                setup_album.append(track_album.get(tid) if tid is not None else None)
                setup_gt.append(cow)
            else:
                id_truth.append(cow)
                id_pred.append(names.get((tid, minute)) if tid is not None else None)
                id_pred_always.append(names.get(("always", tid, minute)) if tid is not None else None)
    alb = metrics.album_quality(setup_album, setup_gt, len(true_cows))
    to_cow = {a: g for a, g in alb["mapping"].items()}
    mapped = [to_cow.get(p) if p is not None else None for p in id_pred]
    mapped_always = [to_cow.get(p) if p is not None else None for p in id_pred_always]
    rate = lambda a, b: a / b if b else None
    by_group = {}
    for g, cows in (groups or {}).items():                     # e.g. cows the ID transformer never saw
        keep = [i for i, c in enumerate(id_truth) if c in set(cows)]
        by_group[g] = {"cows": sorted(cows),
                       "with_not_sure": metrics.id_prf([mapped[i] for i in keep], [id_truth[i] for i in keep]),
                       "always_answer": metrics.id_prf([mapped_always[i] for i in keep], [id_truth[i] for i in keep])}
    return {"labelled_sightings": len(gt_ids), "true_cows": len(true_cows),
            "detector": {"recall": rate(det_tp, det_tp + det_fn), "precision": rate(det_tp, det_tp + det_fp),
                         "recall_setup_hour": rate(det_split[("setup", "tp")], det_split[("setup", "gt")]),
                         "recall_test_hours": rate(det_split[("test", "tp")], det_split[("test", "gt")]),
                         "match_iou": args.match_iou},
            "tracking": {**metrics.idf1(gt_ids, tr_ids, n_pred), **metrics.continuity(sight),
                         "tracks": len({t for t in tr_ids if t is not None})},
            "album": alb,
            "identification": {"with_not_sure": metrics.id_prf(mapped, id_truth),
                               "always_answer": metrics.id_prf(mapped_always, id_truth),
                               "what": "after the setup hour, every labelled sighting: right if the ID desk named "
                                       "its track with the album cow that is this true cow; not found / not sure "
                                       "/ wrong all count against recall",
                               "by_cow_group": by_group}}


def report(res, tag):
    p = lambda v: "-" if v is None else f"{v:.1%}"
    d, tr, al, idn = res["detector"], res["tracking"], res["album"], res["identification"]["with_not_sure"]
    print(f"[system:{tag}] detector recall {p(d['recall'])} precision {p(d['precision'])}; IDF1 {p(tr['idf1'])}; "
          f"track kept: same camera {p(tr['kept_same_camera'])}, camera change {p(tr['kept_camera_change'])}", flush=True)
    print(f"[system:{tag}] album {al['album_cows']} cows for {al['true_cows']} true (purity {p(al['purity'])}, "
          f"coverage {p(al['coverage'])}, duplicates {al['duplicates']}); identification precision "
          f"{p(idn['precision'])} recall {p(idn['recall'])} F1 {p(idn['f1'])}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data_mmcows.py's folder")
    p.add_argument("--detector", required=True, help="version 10's detector best/")
    p.add_argument("--idnet", required=True)
    p.add_argument("--assoc", default="none", help="assoc.pt, or none: each camera on its own by box overlap")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=12.0)
    p.add_argument("--end", type=float, default=15.0)
    p.add_argument("--setup-min", type=int, default=60)
    p.add_argument("--step", type=float, default=2.0, help="s between processed moments (the live frames kept)")
    p.add_argument("--det-threshold", type=float, default=None, help="default: the detector's own")
    p.add_argument("--calib", type=int, default=0, help="1: CBVD-5's box calibration (made for CBVD-5's boxes)")
    p.add_argument("--max-gap", type=float, default=20.0, help="s a track waits for its cow before it ends")
    p.add_argument("--history", type=int, default=2, help="last sightings per track given to the transformer")
    p.add_argument("--pair-threshold", type=float, default=None, help="default: the one picked in training")
    p.add_argument("--memory-threshold", type=float, default=0.85)
    p.add_argument("--scene-every", type=float, default=60.0, help="s between whole-frame scene tokens")
    p.add_argument("--min-frames", type=int, default=5, help="moments in a minute for a fingerprint")
    p.add_argument("--min-minutes", type=int, default=5, help="setup fingerprints for an album cow")
    p.add_argument("--ema", type=float, default=0.5, help="weight of the newest minute on the scoreboard")
    p.add_argument("--match-iou", type=float, default=0.3, help="detection to labelled box, for scoring")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
