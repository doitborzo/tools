#!/usr/bin/env python3
"""Version 14 (E): naming without a tracking model - every box named on its own, then smoothed.

Same detector (version 10's) and ID transformer (version 14 D's) as before; nothing trained
here. What changes is the system around the ID transformer:

    chains   per camera, boxes are chained only by overlap with the last few seconds (what
             the simple tracker got right 95% of the time); no camera handover, no association
             transformer. A chain is a short memory, not a track that must survive.
    naming   every --step s, every box gets a fingerprint from its chain's last --window
             crops (ID transformer) and is compared with the album; the chain's scoreboard
             (running similarity per album cow, --ema) names it, one cow at most once per camera
             per moment; "not sure" from the ID transformer's own model (fitted on held-out
             cows it trained with) or always answering
    album    the setup hour's chain fingerprints (one per chain per minute), the barn's
             average cow taken out (v14 C), merged by similarity but never two chains seen on
             one camera at one moment (they are two cows); threshold calibrated on the setup
             hour without labels (v14 C)
             three ways, all scored:
               built      as the system made it (names A01.. mapped to true cows by the setup hour)
               confirmed  the farmer's 10-minute check: every album entry told its true cow (ear
                          tag) from the setup hour's labels; entries of one cow merged, entries
                          nobody could name dropped
               oracle     a perfect album: the setup hour's labelled boxes themselves (the ID
                          transformer's ceiling on this barn)

    python herd/v14/system_e.py --data /workspace/v14/mmcows --detector <best/> \\
        --idnet /workspace/v14/idnet_wr10k/idnet.pt --out /workspace/v14/system_V14E

Writes system_e.json (every score), RESULTS_E.md, tracks_e_<way>.jsonl + names_e_<way>.json
(per step, for video.py) for way in built / confirmed / oracle.
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


# ------------------------------------------------------------------ album

def constrained_merge(chains, fps, cannot, thr):
    """Average-linkage merging of chains by fingerprint similarity, never joining two chains that
    were seen on one camera at one moment. chains: ids; fps: {chain: mean fingerprint (unit)};
    cannot: {(a, b)} -> {chain: group index}."""
    groups = {i: {"members": [c], "sum": fps[c].copy(), "n": 1} for i, c in enumerate(chains)}
    of = {c: i for i, c in enumerate(chains)}
    banned = collections.defaultdict(set)
    for a, b in cannot:
        if a in of and b in of:
            banned[of[a]].add(of[b]), banned[of[b]].add(of[a])
    while True:
        ids = list(groups)
        if len(ids) < 2:
            break
        cents = np.stack([groups[i]["sum"] / groups[i]["n"] for i in ids])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
        S = cents @ cents.T
        np.fill_diagonal(S, -1)
        for k, i in enumerate(ids):
            for j in banned[i]:
                if j in groups:
                    S[k, ids.index(j)] = -1
        k = int(S.argmax())
        a, b = divmod(k, len(ids))
        if S[a, b] < thr:
            break
        ga, gb = ids[a], ids[b]
        groups[ga]["members"] += groups[gb]["members"]
        groups[ga]["sum"] += groups[gb]["sum"]
        groups[ga]["n"] += groups[gb]["n"]
        banned[ga] |= banned[gb]
        for j in banned[gb]:
            banned[j].discard(gb), banned[j].add(ga)
        del groups[gb]
    out = {}
    for n, (i, g) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1]["n"])):
        for c in g["members"]:
            out[c] = n
    return out


def seen_together(det_log):
    """{(chain a, chain b)}: two chains in one camera at one moment."""
    at = collections.defaultdict(set)
    for t, cam, _, cid, _ in det_log:
        at[(round(t, 1), cam)].add(cid)
    pairs = set()
    for s in at.values():
        s = sorted(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pairs.add((s[i], s[j]))
    return pairs


def build_album(recs, det_log, min_minutes, k_max=6):
    """Setup recs (minute, chain, fp, qmax, qmean, area) -> centring, prototypes, ids, {chain: album cow}."""
    import system as S
    from gallery import spherical_kmeans
    cen = S.centering(recs)
    recs_c = [(m, c, S.apply_centering(cen, fp), *rest) for m, c, fp, *rest in recs]
    thr, calib = S.self_threshold(recs_c, det_log, det_log[0][0] if det_log else 0.0)
    thr = thr if thr is not None else 0.5
    by = collections.defaultdict(list)
    for m, c, fp, *_ in recs_c:
        by[c].append(fp)
    mean = {c: (lambda v: v / (np.linalg.norm(v) + 1e-9))(np.mean(f, 0)) for c, f in by.items()}
    group_of = constrained_merge(sorted(by), mean, seen_together(det_log), thr)
    members = collections.defaultdict(list)
    for c, g in group_of.items():
        members[g] += by[c]
    P, ids, chain_album = [], [], {}
    n = 0
    for g, fps in sorted(members.items(), key=lambda kv: -len(kv[1])):
        if len(fps) < min_minutes:
            continue
        n += 1
        name = f"A{n:02d}"
        X = np.stack(fps)
        for p in spherical_kmeans(X, min(k_max, max(1, len(fps) // 5))):
            P.append(p), ids.append(name)
        for c, gg in group_of.items():
            if gg == g:
                chain_album[c] = name
    if not P:
        return cen, np.zeros((0, len(cen["mu"])), np.float32), np.array([]), {}, thr, calib
    return cen, np.stack(P).astype(np.float32), np.array(ids), chain_album, thr, calib


def labelled_chains(det_log, labels, t_setup, match_iou):
    """Setup hour: {(chain, minute): Counter(true cow)} from the labelled boxes a chain's box overlaps."""
    by = collections.defaultdict(list)
    for t, cam, box, cid, _ in det_log:
        by[(cam, round(t, 1))].append((box, cid))
    out = collections.defaultdict(collections.Counter)
    for r in labels:
        if r["ts"] >= t_setup:
            continue
        dets = by.get((r["cam"], round(r["ts"], 1))) or by.get((r["cam"], round(r["ts"] - 1, 1))) \
            or by.get((r["cam"], round(r["ts"] + 1, 1))) or []
        for cow, gb in r["boxes"]:
            best = max(dets, key=lambda d: iou(d[0], gb), default=None)
            if best is not None and iou(best[0], gb) >= match_iou:
                out[(best[1], int(r["ts"] // 60))][cow] += 1
    return out


# ------------------------------------------------------------------ the run

def run(args):
    from PIL import Image
    import idnet as I
    import metrics
    import system as S
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
    print(f"[system-e] {len(keys)} steps of {args.step} s, cameras {cams}, setup until {local_hour(t_setup):.2f} h", flush=True)

    det = Live(args.detector, threshold=args.det_threshold, calib=False)
    idm, idck = I.load(args.idnet, dev)
    idm.to(dev)
    nan = idck.get("nan")
    meta = idck.get("feats") or {}
    enc = FrameEncoder(meta.get("encoder", "facebook/dinov2-small"), meta.get("grid", 2)).to(dev).eval()

    chains = S.Tracks(args.max_gap)
    det_log = []                                     # (t, cam, box, chain, conf)
    minute_recs = []                                 # setup: (minute, chain, fp, qmax, qmean, area)
    per_step = []                                    # (k, chain, fp, qmax, qmean, area)  after setup
    last_minute = -1
    t0 = time.time()
    for n_step, k in enumerate(keys):
        t = t_start + k * args.step
        rows = steps[k]
        imgs = {c: Image.open(r["path"]).convert("RGB") for c, r in rows.items()}
        masked = {c: C.apply_mask(im, masks.get(c)) for c, im in imgs.items()}
        found = det.many([masked[c] for c in rows])
        cur, crops = [], []
        for c, f in zip(rows, found):
            a = np.asarray(imgs[c])
            for d in f:
                cur.append({"cam": c, "box": d["bbox"], "conf": float(d["det_score"])})
                crops.append(crop(a, d["bbox"], meta.get("crop", 224), 0.1))
        feats = encode(enc, np.stack(crops)).astype(np.float32) if crops else np.zeros((0, enc.out_dim), np.float32)
        assign = S.iou_assign(cur, chains.active(t), chains, thr=args.chain_iou)
        for i, cid in enumerate(assign):
            if cid is None:
                cid = chains.new()
                assign[i] = cid
            d = cur[i]
            chains.add(cid, t, d["cam"], d["box"], feats[i], d["conf"])
            det_log.append((t, d["cam"], d["box"], cid, d["conf"]))
        minute = int((t - t_start) // 60)
        if t < t_setup:
            if minute != last_minute and last_minute >= 0:
                minute_recs += S.fingerprints(idm, chains, t_start + last_minute * 60, t_start + minute * 60,
                                              last_minute, dev, args.min_frames)
        else:
            # after setup: a fingerprint per box from its chain's last --window sightings
            seqs, meta_s = [], []
            for i, cid in enumerate(assign):
                s = chains.rows[cid]["sightings"][-args.window:]
                seqs.append((np.stack([x[3] for x in s]).astype(np.float32), np.array([x[0] - s[0][0] for x in s], np.float32)))
                meta_s.append((cid, float(np.mean([x[4] for x in s]))))
            if seqs:
                fp, qmax, qmean = I.embed(idm, seqs, dev, max_len=args.window)
                per_step += [(k, cid, fp[i], float(qmax[i]), float(qmean[i]), area) for i, (cid, area) in enumerate(meta_s)]
        last_minute = minute
        if n_step % 50 == 0:
            print(f"\r  {n_step + 1}/{len(keys)} steps, {len(chains.rows)} chains, {(time.time() - t0) / 60:.1f} min", end="", flush=True)
    print(flush=True)
    if t_setup > t_start + last_minute * 60:
        minute_recs += S.fingerprints(idm, chains, t_start + last_minute * 60, t_setup, last_minute, dev, args.min_frames)
    minute_recs = [r for r in minute_recs if r[0] < args.setup_min]

    labels = [r for r in read_jsonl(os.path.join(args.data, "labels.jsonl"))
              if args.start <= local_hour(r["ts"]) < args.end and r["cam"] in set(cams)]
    truth_setup = labelled_chains(det_log, labels, t_setup, args.match_iou)

    # --- the album, three ways
    cen, P, ids, chain_album, thr, calib = build_album(minute_recs, [d for d in det_log if d[0] < t_setup], args.min_minutes)
    setup_pairs = [(chain_album[c], cow) for (c, m), cnt in truth_setup.items() if c in chain_album for cow, n in cnt.items() for _ in range(n)]
    mapping, _ = metrics.best_mapping(setup_pairs)
    print(f"[system-e] album: {len(set(ids.tolist()))} cows from {len({r[1] for r in minute_recs})} setup chains "
          f"(threshold {thr}, {calib})", flush=True)
    albums = {"built": (P, ids, {str(k): v for k, v in mapping.items()})}
    # confirmed: every entry named by its majority true cow; entries of one cow merged, unnamed dropped
    votes = collections.defaultdict(collections.Counter)
    for (c, m), cnt in truth_setup.items():
        if c in chain_album:
            votes[chain_album[c]].update(cnt)
    named = {a: cnt.most_common(1)[0][0] for a, cnt in votes.items()}
    keep = [i for i, a in enumerate(ids) if a in named]
    albums["confirmed"] = (P[keep], np.array([f"cow{named[ids[i]]}" for i in keep]),
                           {f"cow{v}": v for v in set(named.values())})
    print(f"[system-e] confirmed album: {len(set(named.values()))} cows named of {len(set(ids.tolist()))} entries", flush=True)
    # oracle: the setup hour's labelled chain-minutes themselves
    o_fps, o_ids = [], []
    for m, c, fp, *_ in minute_recs:
        cnt = truth_setup.get((c, int((t_start + m * 60) // 60)))
        if cnt and cnt.most_common(1)[0][1] >= 2:
            o_fps.append(S.apply_centering(cen, fp)), o_ids.append(f"cow{cnt.most_common(1)[0][0]}")
    if o_fps:
        oP, oids = I.album(np.stack(o_fps), o_ids)
        albums["oracle"] = (oP, oids, {f"cow{c}": c for c in {int(x[3:]) for x in oids}})
        print(f"[system-e] oracle album: {len(set(oids.tolist()))} cows from {len(o_fps)} labelled chain-minutes", flush=True)

    # --- naming after setup: scoreboard per chain, one cow once per camera per moment
    by_step = collections.defaultdict(list)
    for r in per_step:
        by_step[r[0]].append(r)
    cam_of = {}
    for t, cam, box, cid, _ in det_log:
        cam_of[(cid, int(round((t - t_start) / args.step)))] = cam
    results, named_out = {}, {}
    for way, (AP, Aids, to_cow) in albums.items():
        if not len(AP):
            continue
        cows = sorted(set(Aids.tolist()))
        col = {c: np.where(Aids == c)[0] for c in cows}
        board, said = {}, {}
        for k in sorted(by_step):
            cand = []
            for _, cid, fp, qmax, qmean, area in by_step[k]:
                fpc = S.apply_centering(cen, fp)
                sims = np.array([float((AP[col[c]] @ fpc).max()) for c in cows])
                board[cid] = sims if cid not in board else (1 - args.ema) * board[cid] + args.ema * sims
                s = board[cid]
                o = np.argsort(-s)
                row = {"cow": cows[o[0]], "sim": float(s[o[0]]), "margin": float(s[o[0]] - (s[o[1]] if len(o) > 1 else -1)),
                       "quality_max": qmax, "quality_mean": qmean, "area": area}
                p = float(I.nan_p(nan, [row])[0]) if nan else 1.0
                cand.append((p, cid, row))
            taken = set()
            for p, cid, row in sorted(cand, key=lambda x: -x[0]):
                key = (cam_of.get((cid, k)), row["cow"])
                if p >= (nan or {}).get("threshold", 0.0) and key not in taken:
                    said[(cid, k)] = (row["cow"], row["cow"])
                    taken.add(key)
                else:
                    said[(cid, k)] = (None, row["cow"])
        results[way] = score_e(args, det_log, labels, said, to_cow, t_start, t_setup, metrics)
        named_out[way] = said
        r = results[way]
        print(f"[system-e:{way}] ID F1 {r['with_not_sure']['f1'] or 0:.1%} (answered {r['with_not_sure']['answered_share']:.0%}), "
              f"always answering {r['always_answer']['f1'] or 0:.1%}", flush=True)

    # --- detector, chains, album quality
    setup_album = [(chain_album.get(c), cow) for (c, m), cnt in truth_setup.items() for cow, n in cnt.items() for _ in range(n)]
    alb_q = metrics.album_quality([a for a, _ in setup_album], [g for _, g in setup_album], len({r["boxes"][i][0] for r in labels for i in range(len(r["boxes"]))}))
    det_q = detector_quality(args, det_log, labels, t_start)
    out = {"settings": {k: getattr(args, k) for k in ("step", "setup_min", "start", "end", "max_gap", "chain_iou", "window",
                                                     "min_frames", "min_minutes", "ema", "match_iou")},
           "detector": det_q, "chains": len(chains.rows), "album": {"built": {k: v for k, v in alb_q.items() if k != "mapping"},
                                                                  "threshold": thr, "self_calibration": calib,
                                                                  "confirmed_cows": len(set(named.values())),
                                                                  "oracle_cows": len(set(o_ids))},
           "identification": results,
           "what": "every labelled sighting after the setup hour: right if the box's name (its chain's scoreboard) is "
                   "that true cow; 'not sure' counts against recall"}
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "system_e.json"), out)
    for way, said in named_out.items():
        tag = f"e_{way}"
        with open(os.path.join(args.out, f"tracks_{tag}.jsonl"), "w", encoding="utf-8") as fh:
            for tm, cam, box, cid, conf in det_log:
                fh.write(json.dumps({"t": tm, "cam": cam, "box": box, "track": cid, "conf": round(conf, 3)}) + "\n")
        AP, Aids, to_cow = albums[way]
        write_json(os.path.join(args.out, f"names_{tag}.json"), {
            "t_start": t_start, "t_setup": t_setup, "step": args.step, "setup_min": args.setup_min, "per_step": True,
            "track_album": {str(c): a for c, a in chain_album.items()} if way == "built" else
                           {str(c): f"cow{named[a]}" for c, a in chain_album.items() if a in named},
            "names": [[cid, k, n, always] for (cid, k), (n, always) in said.items()],
            "album_to_cow": to_cow})
    report_e(out, os.path.join(args.out, "RESULTS_E.md"), args.version)
    return out


def detector_quality(args, det_log, labels, t_start):
    by = collections.defaultdict(list)
    for t, cam, box, cid, _ in det_log:
        by[(cam, round(t, 1))].append(box)
    tp = fn = fp = 0
    for r in labels:
        dets = by.get((r["cam"], round(r["ts"], 1))) or by.get((r["cam"], round(r["ts"] - 1, 1))) or by.get((r["cam"], round(r["ts"] + 1, 1)))
        if dets is None:
            continue
        used = set()
        for cow, gb in r["boxes"]:
            best = max(((i, iou(d, gb)) for i, d in enumerate(dets) if i not in used), key=lambda x: x[1], default=None)
            if best and best[1] >= args.match_iou:
                tp += 1
                used.add(best[0])
            else:
                fn += 1
        fp += len(dets) - len(used)
    return {"recall": tp / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None}


def score_e(args, det_log, labels, said, to_cow, t_start, t_setup, metrics):
    """Every labelled sighting after setup -> the name of the overlapping box at that moment."""
    by = collections.defaultdict(list)
    for t, cam, box, cid, _ in det_log:
        by[(cam, round(t, 1))].append((box, cid, int(round((t - t_start) / args.step))))
    pred, pred_always, truth = [], [], []
    for r in labels:
        if r["ts"] < t_setup:
            continue
        dets = by.get((r["cam"], round(r["ts"], 1))) or by.get((r["cam"], round(r["ts"] - 1, 1))) or by.get((r["cam"], round(r["ts"] + 1, 1))) or []
        for cow, gb in r["boxes"]:
            best = max(dets, key=lambda d: iou(d[0], gb), default=None)
            truth.append(cow)
            if best is None or iou(best[0], gb) < args.match_iou:
                pred.append(None), pred_always.append(None)
                continue
            n, always = said.get((best[1], best[2]), (None, None))
            pred.append(to_cow.get(str(n)) if n is not None else None)
            pred_always.append(to_cow.get(str(always)) if always is not None else None)
    return {"with_not_sure": metrics.id_prf(pred, truth), "always_answer": metrics.id_prf(pred_always, truth)}


def hm(h):
    """Local hour (float) -> HH:MM."""
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


def window_text(st):
    return (f"onboarding {hm(st['start'])}-{hm(st['start'] + st['setup_min'] / 60)} ({st['setup_min']} min), "
            f"scored {hm(st['start'] + st['setup_min'] / 60)}-{hm(st['end'])}")


def report_e(out, path, version="E"):
    p = lambda v: "-" if v is None else f"{v:.1%}"
    idn, al = out["identification"], out["album"]
    L = [f"# Version 14 ({version}): naming every box on its own, no tracking model", "",
         f"MmCows, 25 July: {window_text(out['settings'])}.", "",
         "Version 14 (D)'s ID transformer and version 10's detector, nothing trained. Boxes are chained per camera "
         "by overlap over a few seconds only; every box is named from its chain's last crops against the album, "
         "smoothed by the chain's scoreboard; one cow at most once per camera per moment. The album from the setup "
         "hour: the barn's average cow taken out, chains merged by similarity but never two seen together on one "
         "camera. Three albums scored on the same run:", "",
         "| album | how | ID F1 with 'not sure' | answered | ID F1 always answering |", "|---|---|---|---|---|"]
    how = {"built": f"as the system made it ({al['built']['album_cows']} entries for {al['built']['true_cows']} cows, purity "
                    f"{p(al['built']['purity'])}, coverage {p(al['built']['coverage'])})",
           "confirmed": f"the farmer named every entry (ear tag): {al['confirmed_cows']} cows",
           "oracle": f"perfect: the setup hour's labelled boxes themselves ({al['oracle_cows']} cows) - the ID transformer's ceiling here"}
    for way in ("built", "confirmed", "oracle"):
        if way in idn:
            r = idn[way]
            L.append(f"| **{way}** | {how[way]} | **{p(r['with_not_sure']['f1'])}** | {p(r['with_not_sure']['answered_share'])} | "
                     f"**{p(r['always_answer']['f1'])}** |")
    d = out["detector"]
    L += ["", f"Detector recall / precision {p(d['recall'])} / {p(d['precision'])}; {out['chains']} chains; album threshold "
              f"{al['threshold']} ({al['self_calibration']}).", "",
          "Reading it: *built* is the fully automatic system; *confirmed* is what a 10-minute check at onboarding buys; "
          "*oracle* is the most this ID transformer can give on this barn - what is missing from 100% there is the ID "
          "transformer (the view it never trained on), what is missing between oracle and confirmed is the album, "
          "between confirmed and built is the farmer's check.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--idnet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=12.0)
    p.add_argument("--end", type=float, default=15.0)
    p.add_argument("--setup-min", type=int, default=60)
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--det-threshold", type=float, default=None)
    p.add_argument("--max-gap", type=float, default=6.0, help="s a chain waits for its box before it ends")
    p.add_argument("--chain-iou", type=float, default=0.3)
    p.add_argument("--window", type=int, default=8, help="a box's fingerprint: its chain's last N crops")
    p.add_argument("--min-frames", type=int, default=5)
    p.add_argument("--min-minutes", type=int, default=3)
    p.add_argument("--ema", type=float, default=0.3, help="weight of the newest step on a chain's scoreboard")
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--version", default="E", help="the name on the results page (G: v14 G's longer onboarding)")
    args = p.parse_args(argv)
    run(args)


def compare_onboarding(main, control):
    """Two system_e.json scored on the same hour after onboardings of different length -> markdown lines."""
    p = lambda v: "-" if v is None else f"{v:.1%}"
    sm, sc = main["settings"], control["settings"]
    L = ["## Longer onboarding against the control, the same scored hour", "",
         f"- this run: {window_text(sm)}", f"- control: {window_text(sc)}", "",
         "| album | ID F1 always answering: this run | control | ID F1 with 'not sure': this run | control |",
         "|---|---|---|---|---|"]
    for way in ("built", "confirmed", "oracle"):
        a, b = main["identification"].get(way), control["identification"].get(way)
        if a and b:
            L.append(f"| {way} | **{p(a['always_answer']['f1'])}** | {p(b['always_answer']['f1'])} | "
                     f"{p(a['with_not_sure']['f1'])} | {p(b['with_not_sure']['f1'])} |")
    am, ac = main["album"]["built"], control["album"]["built"]
    L += [f"| built album: entries / purity / coverage | {am['album_cows']} / {p(am['purity'])} / {p(am['coverage'])} | "
          f"{ac['album_cows']} / {p(ac['purity'])} / {p(ac['coverage'])} | | |", "",
          "Reading it: the oracle row is the album's pictures alone - a gap there is what the extra onboarding hours "
          "showed the ID transformer (more poses and places per cow); the confirmed row adds the farmer's naming; the "
          "built row is fully automatic. Close rows mean the length of the onboarding is not what limits it.", ""]
    return L


def compare_main(argv):
    p = argparse.ArgumentParser(description="compare two system_e.json scored on the same hour")
    p.add_argument("--main", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--append", required=True, help="the results page to add the comparison to")
    a = p.parse_args(argv)
    L = compare_onboarding(json.load(open(a.main, encoding="utf-8")), json.load(open(a.control, encoding="utf-8")))
    with open(a.append, "a", encoding="utf-8") as fh:
        fh.write("\n" + "\n".join(L))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    sys.exit(compare_main(sys.argv[2:]) if sys.argv[1:2] == ["compare"] else main())
