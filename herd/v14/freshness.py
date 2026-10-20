#!/usr/bin/env python3
"""Is the album going stale? A test of version 14 (E)'s good first minutes, on MmCows' labelled boxes alone.

The claim: E names cows well right after the setup hour because the album's pictures are minutes old (the
same stall, pose and light), and worse later because the cows have changed since. Three predictions,
each tested here with no detector, no tracking and no scoreboard - single labelled boxes, so tracking
cannot be the cause:

  1 gap      album = every cow's boxes in a short window (--album-min minutes), queries = boxes before and
             after it; top-1 by the signed gap (query time - album time). Windows across the whole day: if
             accuracy falls with the gap from EVERY window, in both directions, it is freshness; if one hour
             is good whatever the album, it is an easy hour.
  2 place    at the same gap, queries split by how far the cow is from where the album saw her on that
             camera: same spot / near / moved / not on this camera in the album. Freshness through place
             and pose shows as same spot >> moved at equal gaps.
  3 model    the same with plain DINOv2 vectors (--model none) next to the ID model: if both fall alike, the
             cows' look drifts and no metric trained elsewhere will hide it.

One unit = one cow on one camera in one minute (one labelled box, the first of the minute: frames are
15 s apart inside 12:00-15:00 and 60 s outside, so units are alike everywhere). Chance = 1 / cows in the album.

    python herd/v14/freshness.py --data /workspace/v14/mmcows --idnet /workspace/v14/idnet_wr10k/idnet.pt \\
        --out /workspace/v14/freshness
    python herd/v14/freshness.py --data /workspace/v14/mmcows --model none --out /workspace/v14/freshness_dinov2

Writes <out>/freshness.json and FRESHNESS.md.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

import v14common as C  # noqa: F401  (paths)
from v14common import local_hour, read_jsonl, write_json

GAP_BINS = (-180, -120, -60, -30, -15, -5, 0, 5, 15, 30, 60, 120, 180, 1e9)
PLACE = (("same spot", 0.03), ("near", 0.10), ("moved", 9.0))


def units(frames, rows, feats, per_minute=1):
    """Labelled boxes -> one per (cow, camera, minute): {t, minute, cow, cam, centre, feat}."""
    seen = collections.Counter()
    out = []
    order = sorted(range(len(rows)), key=lambda i: frames[rows[i]["frame"]]["ts"])
    for i in order:
        r, f = rows[i], frames[rows[i]["frame"]]
        m = int(f["ts"] // 60)
        k = (r["cow"], f["cam"], m)
        if seen[k] >= per_minute:
            continue
        seen[k] += 1
        b = r["box"]
        out.append({"t": f["ts"], "minute": m, "cow": r["cow"], "cam": f["cam"], "centre": ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
                    "feat": np.asarray(feats[i], np.float32)})
    return out


def fingerprints(us, model, dev):
    """Each unit -> unit-length vector: the ID model on a one-frame sequence, or the plain vector (model None)."""
    X = np.stack([u["feat"] for u in us]).astype(np.float32)
    if model is None:
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    import idnet as I
    fp, _, _ = I.embed(model, [(x[None], np.zeros(1, np.float32)) for x in X], dev)
    return fp


def place_of(u, album_units):
    mine = [a["centre"] for a in album_units if a["cow"] == u["cow"] and a["cam"] == u["cam"]]
    if not mine:
        return "not on this camera in the album"
    d = min(float(np.hypot(u["centre"][0] - c[0], u["centre"][1] - c[1])) for c in mine)
    return next(name for name, lim in PLACE if d < lim)


def gap_bin(g):
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        if lo <= g < hi:
            return f"{lo:+.0f}..{hi:+.0f}" if hi < 1e8 else f">= {lo:+.0f}"
    return f"< {GAP_BINS[0]:+.0f}"


def run(us, fps, windows, album_min, min_cows):
    """Every album window -> every query outside it: (window, signed gap in minutes, place, right)."""
    import idnet as I
    t = np.array([u["t"] for u in us])
    cows = np.array([u["cow"] for u in us])
    recs, used = [], []
    for w0 in windows:
        a = np.where((t >= w0) & (t < w0 + album_min * 60))[0]
        if len({cows[i] for i in a}) < min_cows:
            continue
        P, ids = I.album(fps[a], cows[a].tolist())
        known = set(ids.tolist())
        q = np.where(((t < w0) | (t >= w0 + album_min * 60)) & np.isin(cows, list(known)))[0]
        if not len(q):
            continue
        sims = fps[q] @ P.T
        pick = ids[np.argmax(sims, 1)]
        centre = w0 + album_min * 30
        a_units = [us[i] for i in a]
        for qi, p in zip(q, pick):
            recs.append({"window": float(w0), "gap_min": (t[qi] - centre) / 60.0, "place": place_of(us[qi], a_units),
                         "right": bool(p == cows[qi]), "album_cows": len(known)})
        used.append({"start_hour": round(local_hour(w0), 3), "album_cows": len(known), "album_units": int(len(a)),
                     "queries": int(len(q))})
    return recs, used


def table(recs, key):
    acc = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        k = key(r)
        acc[k][0] += r["right"]
        acc[k][1] += 1
    return {k: {"top1": v[0] / v[1], "queries": v[1]} for k, v in acc.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data_mmcows.py's folder (frames.jsonl, assoc_rows.jsonl, assoc.npz)")
    p.add_argument("--idnet", default=None, help="the ID model (idnet.pt of v14 A-E); omit with --model none")
    p.add_argument("--model", choices=("idnet", "none"), default="idnet", help="none: plain DINOv2 vectors")
    p.add_argument("--out", required=True)
    p.add_argument("--album-min", type=float, default=10.0, help="minutes of boxes in one album")
    p.add_argument("--every-min", type=float, default=30.0, help="minutes between album windows")
    p.add_argument("--from-hour", type=float, default=0.0)
    p.add_argument("--to-hour", type=float, default=24.0)
    p.add_argument("--min-cows", type=int, default=12, help="an album window with fewer cows is skipped")
    p.add_argument("--per-minute", type=int, default=1)
    a = p.parse_args(argv)

    frames = read_jsonl(os.path.join(a.data, "frames.jsonl"))
    rows = read_jsonl(os.path.join(a.data, "assoc_rows.jsonl"))
    if not rows:
        sys.exit("run data_mmcows.py frames + feats first")
    feats = np.load(os.path.join(a.data, "assoc.npz"))["box_feats"]
    us = [u for u in units(frames, rows, feats, a.per_minute) if a.from_hour <= local_hour(u["t"]) < a.to_hour]
    model = dev = None
    if a.model == "idnet":
        if not a.idnet:
            sys.exit("--idnet <idnet.pt> (or --model none)")
        import idnet as I
        from common import device
        dev = device()
        model, ck = I.load(a.idnet, dev)
        if ck.get("arch") == "opencowid":
            sys.exit("an OpenCowID model reads pictures, not the DINOv2 vectors here: use v14 D's idnet.pt")
        model.to(dev)
    fps = fingerprints(us, model, dev)
    t0, t1 = min(u["t"] for u in us), max(u["t"] for u in us)
    step = a.every_min * 60
    windows = list(np.arange(np.floor(t0 / step) * step, t1, step))
    recs, used = run(us, fps, windows, a.album_min, a.min_cows)
    if not recs:
        sys.exit("no album window had enough cows")

    order = lambda k: GAP_BINS.index(float(k.split("..")[0])) if ".." in k else (99 if k.startswith(">=") else -1)
    by_gap = table(recs, lambda r: gap_bin(r["gap_min"]))
    by_place = table([r for r in recs if 0 < abs(r["gap_min"]) <= 60], lambda r: (gap_bin(r["gap_min"]), r["place"]))
    by_window = table([r for r in recs if 0 < abs(r["gap_min"]) <= 15], lambda r: round(local_hour(r["window"]), 2))
    by_window_far = table([r for r in recs if abs(r["gap_min"]) >= 60], lambda r: round(local_hour(r["window"]), 2))
    chance = float(np.mean([1 / r["album_cows"] for r in recs]))
    out = {"model": a.model if a.model == "none" else a.idnet, "units": len(us), "windows": used, "chance": chance,
           "by_gap": by_gap, "by_gap_and_place": {f"{g} | {pl}": v for (g, pl), v in by_place.items()},
           "by_window_near": by_window, "by_window_far": by_window_far,
           "settings": {k: getattr(a, k) for k in ("album_min", "every_min", "from_hour", "to_hour", "min_cows", "per_minute")}}
    os.makedirs(a.out, exist_ok=True)
    write_json(os.path.join(a.out, "freshness.json"), out)

    pc = lambda v: f"{100 * v:.1f}%"
    L = [f"# Is the album going stale? ({'plain DINOv2' if a.model == 'none' else 'ID model ' + os.path.basename(os.path.dirname(a.idnet))})", "",
         f"{len(us)} labelled cow-camera-minutes; {len(used)} albums of {a.album_min:g} min, one every {a.every_min:g} min; "
         f"single boxes, no tracking. Chance {pc(chance)}.", "",
         "## 1. Top-1 by the time from the album (minutes; minus = before it)", "", "| gap | top-1 | queries |", "|---|---|---|"]
    L += [f"| {k} | {pc(v['top1'])} | {v['queries']} |" for k, v in sorted(by_gap.items(), key=lambda kv: order(kv[0]))]
    L += ["", "Freshness: highest at small gaps on both sides, falling with the gap either way.", "",
          "## 2. Same gap, same place or moved (gaps within an hour)", "", "| gap | " + " | ".join(n for n, _ in PLACE) +
          " | not on this camera in the album |", "|---" * (len(PLACE) + 2) + "|"]
    places = [n for n, _ in PLACE] + ["not on this camera in the album"]
    for g in sorted({g for g, _ in by_place}, key=order):
        cells = [f"{pc(by_place[(g, pl)]['top1'])} ({by_place[(g, pl)]['queries']})" if (g, pl) in by_place else "-" for pl in places]
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L += ["", "Same spot well above moved at equal gaps: what stays fresh is the place and pose, not the clock.", "",
          "## 3. Every album window: near queries (within 15 min) against far ones (an hour or more)", "",
          "| album from (local hour) | top-1 near | top-1 far |", "|---|---|---|"]
    for h in sorted(set(by_window) | set(by_window_far)):
        n, f = by_window.get(h), by_window_far.get(h)
        L.append(f"| {h:.2f} | {pc(n['top1']) + ' (' + str(n['queries']) + ')' if n else '-'} | "
                 f"{pc(f['top1']) + ' (' + str(f['queries']) + ')' if f else '-'} |")
    L += ["", "Near >> far for every window: it is freshness. One hour good for near and far alike: an easy hour.", ""]
    with open(os.path.join(a.out, "FRESHNESS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
