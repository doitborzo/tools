#!/usr/bin/env python3
"""Cameras that share floor, calibrated by the cows themselves - nothing measured in the barn.

[mct] link joins one cow's tracks on two cameras by where she stands on the floor, which needs every
camera's picture mapped onto one floor plan (floor_points). Marks measured by hand are not needed:
a cow that gets up, walks or turns does it at the same moment on every camera that sees her. The
detector looks at every second of the recordings, on each camera's whole picture (a run's store will
not do when the masks cut the shared floor away from one camera) - the boxes of every second:

  1. each track's motion (how far her box moved each second, in her own heights) on each camera; the
     tracks of two cameras that move together (correlated over 20 s and more) are the same cow - and
     the lag where they move together best is the two cameras' clock difference
  2. her floor point (the box's bottom centre) on both cameras at the same seconds: point pairs. RANSAC
     finds the floor mapping (a homography) most pairs agree with, and with it each camera's lens
     distortion (wide lenses bend the floor: the one-parameter division model, tried over a grid)
  3. checked on every cow both cameras see at once, still ones too: the share whose mapped point lands
     on a cow of the other camera; a pair of cameras that shares no floor is reported as such
  4. the floor plan in about centimetres: the reference camera's floor straightened so that cows are the
     same size near and far (a lying cow's box ~150 cm wide); the link distance from the pairs' spread

  -> OUT/calib.json, OUT/<a>_<b>.jpg (the other camera's cows drawn onto this one's picture: they should
     sit on its cows), and with --apply the cameras' floor_points / distortion and [mct] link in barn.toml.
     OUT/<a>_<b>_moving.jpg, found or not: both pictures, the paths of the cows that walked, the pairs.

    python herd/herd.py autocalib --config barn.toml --videos rec --out calib
    python herd/herd.py autocalib ... --pairs cam1:cam2 cam2:cam3 --apply
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import blank_excluded, homography, in_zone, load_config, recordings, undistort  # noqa: E402

K_GRID = (0.0, -0.1, -0.2, -0.3)
MIN_COWS = 4                 # moving cows that agree on the floor mapping, at least


# ------------------------------------------------------------------ data

def load_tracks(store, cam):
    """{track: {second: (u, v, box height, box width)}} - u v the box's bottom centre (0-1)."""
    return _tracks(store.query("SELECT track, ts, x1, y1, x2, y2 FROM seconds WHERE cam = ?", (cam,)))


def detect_tracks(cfg, cam, recs, out, live=None, every_s=1.0):
    """The camera's cows over its whole picture, every second of its recordings, tracked -> as load_tracks.

    A run's store has only the cows in each camera's own mask - and overlapping cameras' masks are cut
    so that the shared floor belongs to one of them: the other never stores the cows there. So the
    detector looks again here, on the whole picture (only the exclude zone grey, as in the barn); the
    boxes are kept in OUT/boxes_<cam>.json and a second run on the same recordings reads them back."""
    import cv2
    from PIL import Image
    from pipeline import Tracker
    cache = os.path.join(out, f"boxes_{cam}.json")
    names = [os.path.basename(p_) for p_, _ in recs]
    if os.path.exists(cache):
        old = json.load(open(cache, encoding="utf-8"))
        if old.get("files") == names:
            print(f"[calib] {cam}: boxes of {len(names)} recordings from {cache}", flush=True)
            return _tracks(old["rows"])
    if live is None:
        import detector as det_mod
        weights = cfg["detector"]["weights"]
        if not weights or not os.path.isdir(weights):
            sys.exit(f"[calib] [detector] weights = {weights!r}: no detector there (herd.py autocalib needs it to "
                     "see the cows on the whole picture; or give --store of a run)")
        live = det_mod.Live(weights, threshold=cfg["detector"]["threshold"] or None,
                            tiles=cfg["detector"].get("tiles"))
    c = next(x for x in cfg["cameras"] if x["id"] == cam)
    whole = {**c, "mask": None, "tracking_mask": None}            # the whole picture, minus exclude
    width = cfg["sampling"]["frame_width"]
    tracker, rows = Tracker(cam), []

    def flush(batch):
        for (ts, w, h), found in zip([b[0] for b in batch], live.many([b[1] for b in batch])):
            boxes = [d["bbox"] for d in found if in_zone(d["bbox"], whole, w, h)]
            for tid, b in tracker.update(ts, boxes):
                rows.append([tid, ts, *[round(float(v), 4) for v in b]])
        batch.clear()

    for path, start in recs:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[calib] {cam}: cannot open {path}", flush=True)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = fps if 1.0 <= fps <= 120.0 else 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        t, bad, last, said, batch = -1.0, 0, None, 0.0, []
        while True:
            if not cap.grab():                                    # a damaged stretch: skipped, as frames_from
                bad += 1
                if bad > 100 or (n and cap.get(cv2.CAP_PROP_POS_FRAMES) >= n - 1):
                    break
                continue
            bad = 0
            ms = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t = ms if ms > t else t + 1.0 / fps
            sec = int(t // every_s)
            if sec == last:
                continue
            last = sec
            ok, f = cap.retrieve()
            if not ok:
                continue
            img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            if w > width:
                img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            if c.get("blank_excluded", True):
                img = blank_excluded(img, c.get("exclude"))
            batch.append(((float(start + sec * every_s), w, h), Image.fromarray(img)))
            if len(batch) == 8:
                flush(batch)
            if t - said >= 60:
                said = t
                print(f"\r[calib] {cam}: {os.path.basename(path)} {int(t // 60)} min, {len(rows)} cow-seconds",
                      end="", flush=True)
        flush(batch)
        cap.release()
    print(f"\r[calib] {cam}: {len(rows)} cow-seconds in {len(recs)} recordings" + " " * 20, flush=True)
    os.makedirs(out, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"files": names, "rows": rows}, fh)
    return _tracks(rows)


def _tracks(rows):
    out = collections.defaultdict(dict)
    for track, ts, x1, y1, x2, y2 in rows:
        out[track][int(round(ts))] = ((x1 + x2) / 2, y2, max(y2 - y1, 0.02), max(x2 - x1, 0.02))
    return out


def motion(tr, t0, n):
    """A track's motion a second, in its own box heights, smoothed over 3 s, on seconds t0..t0+n (nan off)."""
    s = np.full(n, np.nan)
    secs = sorted(tr)
    for a, b in zip(secs, secs[1:]):
        if b == a + 1 and t0 <= b < t0 + n:
            (u0, v0, h0, _), (u1, v1, h1, _) = tr[a], tr[b]
            s[b - t0] = np.hypot(u1 - u0, v1 - v0) / ((h0 + h1) / 2)
    k = np.ones(3) / 3
    ok = ~np.isnan(s)
    num = np.convolve(np.where(ok, s, 0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)


def corr_at(a, b, lag, min_overlap):
    """Pearson correlation of a[t] and b[t + lag] where both are seen (>= min_overlap s), else None."""
    if lag > 0:
        a, b = a[:-lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[:lag]
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < min_overlap:
        return None
    x, y = a[ok], b[ok]
    if x.std() < 0.02 or y.std() < 0.02:                       # one of them never moved: no evidence
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _motions(T, t0, n, min_overlap):
    """Each track's motion over t0..t0+n, less what all the camera's cows 'did' together that second (the
    picture's own jitter - exposure, compression - moves every box at once, cow or not)."""
    M = {k: motion(v, t0, n) for k, v in T.items() if len(v) >= min_overlap}
    if len(M) >= 4:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                common = np.nanmedian(np.array(list(M.values())), axis=0)
        common = np.where(np.isnan(common), 0.0, common)
        M = {k: m - common for k, m in M.items()}
    return {k: m for k, m in M.items() if np.nanstd(m) > 0.02}


OFF_LAGS = tuple(d for d in range(-60, 61, 4) if abs(d) >= 12)


def standout(a, b, lag, min_overlap):
    """How far a[t], b[t + lag] move together above what they do at other lags (12-60 s off): a cow that
    gets up on both cameras at once stands out; two boxes that only jitter do not. -> (corr, margin)."""
    c = max((x for d in (-1, 0, 1) if (x := corr_at(a, b, lag + d, min_overlap)) is not None), default=None)
    if c is None:
        return None, 0.0
    off = [x for d in OFF_LAGS if (x := corr_at(a, b, lag + d, min_overlap)) is not None]
    return c, c - (max(off) if len(off) >= 5 else 1.0)


def moving_pairs(TA, TB, max_lag, min_overlap=20, min_corr=0.5, min_margin=0.3):
    """Tracks of camera A and B that move together -> (lag of B behind A in s, [(track A, track B, corr)]).

    A pair counts when the two move together clearly better at that lag than at other lags, and each
    track has one partner at most (the one it moves with best)."""
    secs = [s for T in (TA, TB) for tr in T.values() for s in tr]
    if not secs:
        return 0, []
    t0, n = min(secs), max(secs) - min(secs) + 1
    MA, MB = _motions(TA, t0, n, min_overlap), _motions(TB, t0, n, min_overlap)
    # the clocks' lag: voted by the tracks that move most (every lag tried on them only)
    top = lambda M: dict(sorted(M.items(), key=lambda kv: -np.nanstd(kv[1]) * np.sum(~np.isnan(kv[1])))[:15])
    votes = collections.Counter()
    for (ka, a), (kb, b) in itertools.product(top(MA).items(), top(MB).items()):
        cs = [(c, lag) for lag in range(-max_lag, max_lag + 1) if (c := corr_at(a, b, lag, min_overlap)) is not None]
        if cs and max(cs)[0] >= min_corr:
            c, lag = max(cs)
            if standout(a, b, lag, min_overlap)[1] >= min_margin:
                votes[lag] += c
    if not votes:
        return 0, []
    lag = max(votes, key=lambda x: (votes[x], -abs(x)))
    cand = []                                        # every pair at that lag (a second either way)
    for (ka, a), (kb, b) in itertools.product(MA.items(), MB.items()):
        c, margin = standout(a, b, lag, min_overlap)
        if c is not None and c >= min_corr and margin >= min_margin:
            cand.append((ka, kb, c))
    pairs, used = [], set()
    for ka, kb, c in sorted(cand, key=lambda p: -p[2]):
        if ("a", ka) not in used and ("b", kb) not in used:
            used |= {("a", ka), ("b", kb)}
            pairs.append((ka, kb, c))
    return lag, pairs


# ------------------------------------------------------------------ the floor mapping

def apply_h(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def pair_points(TA, TB, pairs, lag, kA, kB, arA, arB):
    """The moving pairs' floor points at their common seconds -> [(pts B, pts A, A's box heights, pair)]
    (undistorted coordinates)."""
    corr = []
    for ka, kb, _ in pairs:
        common = [s for s in TA[ka] if s + lag in TB[kb]]
        if len(common) < 8:
            continue
        a = np.array([TA[ka][s] for s in common])
        b = np.array([TB[kb][s + lag] for s in common])
        pa = np.c_[undistort(a[:, 0], a[:, 1], kA, arA)]
        pb = np.c_[undistort(b[:, 0], b[:, 1], kB, arB)]
        corr.append((pb, pa, a[:, 2], (ka, kb)))
    return corr


def pair_errors(H, corr, cat=None):
    """Each pair's median error (in A's box heights) under H - all points mapped at once."""
    if cat is None:
        cat = (np.concatenate([c[0] for c in corr]), np.concatenate([c[1] for c in corr]),
               np.concatenate([c[2] for c in corr]), np.cumsum([len(c[0]) for c in corr])[:-1])
    pb, pa, ha, cuts = cat
    err = np.hypot(*(apply_h(H, pb) - pa).T) / ha
    err = np.where(np.isfinite(err), err, 1e9)
    return np.array([np.median(e) for e in np.split(err, cuts)])


def fit_pair(TA, TB, pairs, lag, kA, kB, arA, arB, rng, iters=2000, rel_thr=0.6):
    """RANSAC homography B -> A (undistorted coordinates) from the moving pairs -> (H, inlier pairs, corrs)."""
    corr = pair_points(TA, TB, pairs, lag, kA, kB, arA, arB)
    if len(corr) < 4:
        return None, [], corr
    cat = (np.concatenate([c[0] for c in corr]), np.concatenate([c[1] for c in corr]),
           np.concatenate([c[2] for c in corr]), np.cumsum([len(c[0]) for c in corr])[:-1])
    best, best_in = None, []
    for _ in range(iters):
        idx = rng.choice(len(corr), 4, replace=False)
        src, dst = [], []
        for i in idx:
            j = rng.integers(len(corr[i][0]))
            src.append(corr[i][0][j])
            dst.append(corr[i][1][j])
        H = homography(src, dst)
        if not np.all(np.isfinite(H)):
            continue
        inl = list(np.flatnonzero(pair_errors(H, corr, cat) < rel_thr))
        if len(inl) > len(best_in):
            best, best_in = H, inl
    if best is None or len(best_in) < 3:
        return None, best_in, corr
    return refit(corr, best, best_in, rel_thr)


def refit(corr, H, inl, rel_thr=0.6):
    """H fitted again on every point of the agreeing pairs (twice: the pairs that agree may grow)."""
    for _ in range(2):
        H2 = homography(np.concatenate([corr[i][0] for i in inl]), np.concatenate([corr[i][1] for i in inl]))
        inl2 = list(np.flatnonzero(pair_errors(H2, corr) < rel_thr))
        if len(inl2) < 3:
            break
        H, inl = H2, inl2
    return H, inl, corr


def agreement(TA, TB, H, lag, kA, kB, arA, arB, rel_thr=0.6):
    """Every second both cameras saw cows: B's cows mapped into A's picture that land inside it - the share
    that lands on one of A's cows (within rel_thr of her height), each of A's cows taken once.

    In a full pen almost any point is near some cow, so the same is counted with the mapped points
    moved 3 cow heights aside (4 ways): what a mapping that knows nothing scores. -> (share, how many,
    that chance share)."""
    byA, byB = collections.defaultdict(list), collections.defaultdict(list)
    for tr in TA.values():
        for s, p in tr.items():
            byA[s].append(p)
    for tr in TB.values():
        for s, p in tr.items():
            byB[s - lag].append(p)
    hit = n = 0
    chance = np.zeros(4)
    for s, pb in byB.items():
        pa = byA.get(s)
        if not pa:
            continue
        a, b = np.array(pa), np.array(pb)
        qa = np.c_[undistort(a[:, 0], a[:, 1], kA, arA)]
        qb = apply_h(H, np.c_[undistort(b[:, 0], b[:, 1], kB, arB)])
        inside = (qb[:, 0] > 0.02) & (qb[:, 0] < 0.98) & (qb[:, 1] > 0.02) & (qb[:, 1] < 0.98)
        qb = qb[inside]
        if not len(qb):
            continue
        step = 3 * float(np.median(a[:, 2]))
        for k, off in enumerate([(0, 0), (step, 0), (-step, 0), (0, step), (0, -step)]):
            d = np.hypot(*(qa[None, :, :] - (qb + off)[:, None, :]).transpose(2, 0, 1)) / a[None, :, 2]
            got = 0
            for _ in range(min(d.shape)):                             # one of A's cows for one of B's
                i, j = np.unravel_index(np.argmin(d), d.shape)
                if d[i, j] >= rel_thr:
                    break
                got += 1
                d[i, :], d[:, j] = np.inf, np.inf
            if k == 0:
                hit += got
            else:
                chance[k - 1] += got
        n += len(qb)
    return (hit / n if n else 0.0), n, (float(chance.mean()) / n if n else 0.0)


def straighten(TA, kA, arA, width_cm=150.0):
    """The reference camera's floor in about cm: a projective straightening (a, b) that makes the cows'
    box widths the same near and far, scaled so a box is width_cm wide -> 3x3 R (undistorted -> cm)."""
    from scipy.optimize import minimize
    boxes = [p for tr in TA.values() for p in tr.values()]
    if not boxes:
        return np.diag([1000.0, 1000.0, 1.0])
    b = np.array(boxes[:: max(1, len(boxes) // 4000)])
    half = b[:, 3] / 2                                     # the box's bottom edge
    lu, lv = undistort(b[:, 0] - half, b[:, 1], kA, arA)
    ru, rv = undistort(b[:, 0] + half, b[:, 1], kA, arA)
    L, R_ = np.c_[lu, lv], np.c_[ru, rv]

    def widths(ab):
        P = np.array([[1, 0, 0], [0, 1, 0], [ab[0], ab[1], 1.0]])
        return np.hypot(*(apply_h(P, R_) - apply_h(P, L)).T), P

    def loss(ab):
        den = ab[0] * np.r_[L[:, 0], R_[:, 0]] + ab[1] * np.r_[L[:, 1], R_[:, 1]] + 1
        if den.min() < 0.05:
            return 1e9
        w, _ = widths(ab)
        return float(np.var(np.log(np.maximum(w, 1e-6))))
    ab = minimize(loss, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6}).x
    w, P = widths(ab)
    return np.diag([width_cm / np.median(w), width_cm / np.median(w), 1.0]) @ P


# ------------------------------------------------------------------ main

def grid_points(region, n=3):
    (u0, v0), (u1, v1) = region
    return [(u, v) for u in np.linspace(u0, u1, n) for v in np.linspace(v0, v1, n)]


def frame_at(videos, cam, s):
    """The camera's picture (BGR) at second s, from the recording that holds it."""
    import cv2
    if not videos.get(cam):
        return None
    vpath, start = max((v for v in videos[cam] if v[1] <= s), key=lambda v: v[1], default=videos[cam][0])
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, s - start) * (fps if 1.0 <= fps <= 120.0 else 25.0)))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def walked(tr):
    """How far a track got from where it mostly was, in its own box heights (a lying cow: ~0)."""
    p = np.array(list(tr.values()))
    mid = np.median(p[:, :2], 0)
    return float(np.max(np.hypot(*(p[:, :2] - mid).T)) / np.median(p[:, 2]))


def draw_moving(path, videos, cam_a, cam_b, TA, TB, mv):
    """What calibration has to go on, drawn whatever came of it: both cameras side by side, the path of
    every cow that walked (orange; the rest a grey dot where she lay or stood), the pairs found moving
    together numbered alike on both."""
    import cv2
    pics = []
    for cam, T in ((cam_a, TA), (cam_b, TB)):
        secs = sorted(s for tr in T.values() for s in tr)
        img = frame_at(videos, cam, secs[len(secs) // 2]) if secs else None
        if img is None:
            return None
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 720 / h), 720))
        h, w = img.shape[:2]
        n = 0
        for tr in T.values():
            p = np.array(list(tr.values()))
            if walked(tr) > 1.0 and len(p) >= 10:
                n += 1
                cv2.polylines(img, [np.int32(p[:, :2] * [w, h])], False, (0, 140, 255), 2)
            else:
                x, y = np.median(p[:, :2], 0)
                cv2.circle(img, (int(x * w), int(y * h)), 4, (170, 170, 170), -1)
        side = 0 if cam == cam_a else 1
        for i, pr in enumerate(mv):
            tr = T[pr[side]]
            x, y = np.median(np.array(list(tr.values()))[:, :2], 0)
            cv2.circle(img, (int(x * w), int(y * h)), 16, (40, 40, 230), 3)
            cv2.putText(img, str(i + 1), (int(x * w) + 18, int(y * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (40, 40, 230), 3)
        cv2.rectangle(img, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(img, f"{cam}: {n} of {len(T)} tracks walked (orange)", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        pics.append(img)
    img = np.hstack(pics)
    cv2.rectangle(img, (0, img.shape[0] - 50), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, f"moving together on both (red, numbered alike): {len(mv)} - {MIN_COWS} or more needed",
                (12, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(path, img)
    return path


def draw(path, videos, cam_a, cam_b, TA, TB, H, lag, kA, kB, arA, arB):
    """B's cows of one second drawn onto A's picture of that second (red), A's own (green)."""
    import cv2
    counts = collections.Counter(s for tr in TA.values() for s in tr) + collections.Counter(
        s - lag for tr in TB.values() for s in tr)
    if not counts:
        return None
    if not videos.get(cam_a):
        return None
    s = max(counts, key=counts.get)
    img = frame_at(videos, cam_a, s)
    if img is None:
        return None
    h, w = img.shape[:2]

    def back(q):                                    # undistorted A coordinates -> A's picture (invert numerically)
        u, v = q
        for _ in range(20):
            uu, vv = undistort(u, v, kA, arA)
            u, v = u + (q[0] - uu), v + (q[1] - vv)
        return int(u * w), int(v * h)
    for tr in TA.values():
        if s in tr:
            cv2.circle(img, (int(tr[s][0] * w), int(tr[s][1] * h)), 9, (60, 220, 60), 3)
    for tr in TB.values():
        if s + lag in tr:
            u, v = tr[s + lag][:2]
            q = apply_h(H, np.array([undistort(u, v, kB, arB)]))[0]
            if -0.1 < q[0] < 1.1 and -0.1 < q[1] < 1.1:
                x, y = back(q)
                cv2.drawMarker(img, (x, y), (40, 40, 230), cv2.MARKER_CROSS, 22, 3)
    cv2.putText(img, f"{cam_a}: its cows (green) and {cam_b}'s mapped onto it (red crosses)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(path, img)
    return path


def toml_value(v):
    return json.dumps(v)


def apply_config(path, calib, link_dist):
    """barn.toml: the calibrated cameras' floor_points / distortion / aspect, and [mct] link on."""
    text = open(path, encoding="utf-8").read()
    shutil.copy(path, path + ".bak")
    blocks = re.split(r"(?m)^(?=\[\[cameras\]\])", text)
    out = [blocks[0]]
    for b in blocks[1:]:
        m = re.search(r'(?m)^id\s*=\s*"([^"]+)"', b)
        cam = m.group(1) if m else None
        if cam in calib:
            b = re.sub(r"(?m)^(floor_points|distortion|aspect)\s*=.*\n", "", b)
            add = "".join(f"{k} = {toml_value(calib[cam][k])}\n" for k in ("floor_points", "distortion", "aspect"))
            b = b.rstrip("\n") + "\n" + add + "\n"
        out.append(b)
    text = "".join(out)
    # [mct]: enabled, mode, link_dist (set or added)
    m = re.search(r"(?m)^\[mct\][^\n]*\n", text)
    if m:
        end = re.search(r"(?m)^\[", text[m.end():])
        sec = text[m.end(): m.end() + end.start()] if end else text[m.end():]
        for k, v in (("enabled", "true"), ("mode", '"link"'), ("link_dist", str(int(link_dist)))):
            if re.search(rf"(?m)^{k}\s*=", sec):
                sec = re.sub(rf"(?m)^{k}\s*=[^#\n]*", f"{k} = {v}    ", sec, count=1)
            else:
                sec = f"{k} = {v}\n" + sec
        text = text[:m.end()] + sec + (text[m.end() + end.start():] if end else "")
    else:
        text += f'\n[mct]\nenabled = true\nmode = "link"\nlink_dist = {int(link_dist)}\n'
    open(path, "w", encoding="utf-8").write(text)
    load_config(path)                                             # still a config


def main(argv=None, live=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--videos", default=None, help="the recordings: the detector looks at every second of them, on "
                   "the whole picture (boxes kept in OUT, a second run reads them back)")
    p.add_argument("--store", default=None, help="instead: a run's boxes (its store) - only when no camera's mask "
                   "cuts the shared floor away (a run stores the cows in each camera's mask only)")
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", nargs="*", default=None, help="camera pairs to calibrate, cam1:cam2 (default: every "
                   "pair not in [mct] disjoint)")
    p.add_argument("--max-lag", type=int, default=90, help="the cameras' clocks may differ by this many s")
    p.add_argument("--apply", action="store_true", help="write the result into the config (barn.toml.bak kept)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    from store import Store
    import cv2
    cfg = load_config(args.config)
    ids = [c["id"] for c in cfg["cameras"]]
    if not args.videos and not args.store:
        p.error("--videos (the recordings) or --store (a run's boxes) is needed")
    disjoint = {frozenset(x) for x in cfg["mct"].get("disjoint", [])}
    pairs = [tuple(x.split(":")) for x in args.pairs] if args.pairs else \
        [(a, b) for a, b in itertools.combinations(ids, 2) if frozenset((a, b)) not in disjoint]
    for a, b in pairs:
        for x in (a, b):
            if x not in ids:
                sys.exit(f"[calib] {x}: no such camera in {args.config} ({', '.join(ids)})")
    os.makedirs(args.out, exist_ok=True)
    videos, aspect = collections.defaultdict(list), {c: 16 / 9 for c in ids}
    if args.videos:
        recs, _ = recordings(args.videos, ids, cfg["farm"]["timezone_offset_hours"])
        for cam, path, ts in sorted(recs, key=lambda r: r[2]):
            videos[cam].append((path, ts))
        for cam, vs in videos.items():
            cap = cv2.VideoCapture(vs[0][0])
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if w and h:
                aspect[cam] = w / h
    used = sorted({x for ab in pairs for x in ab})
    if args.store:
        store = Store(args.store)
        T = {c: load_tracks(store, c) for c in used}
        cut = [c["id"] for c in cfg["cameras"] if c["id"] in used and c.get("mask")]
        if cut:
            print(f"[calib] {', '.join(cut)}: a mask - the run stored only the cows inside it; if it cuts the shared "
                  "floor away, give --videos instead of --store", flush=True)
    else:
        T = {}
        for c in used:
            if not videos.get(c):
                sys.exit(f"[calib] {c}: no recordings in {args.videos} (named {c}_YYYY-MM-DD_HH-MM-SS)")
            T[c] = detect_tracks(cfg, c, videos[c], args.out, live=live)
    rng = np.random.default_rng(args.seed)
    found = {}
    for a, b in pairs:
        if not T.get(a) or not T.get(b):
            print(f"[calib] {a}-{b}: no cows seen on {a if not T.get(a) else b}")
            continue
        lag, mv = moving_pairs(T[a], T[b], args.max_lag)
        print(f"[calib] {a}-{b}: {len(mv)} tracks moving together (clock lag {lag:+d} s)", flush=True)
        seen = draw_moving(os.path.join(args.out, f"{a}_{b}_moving.jpg"), videos, a, b, T[a], T[b], mv)
        if seen:
            print(f"[calib] {a}-{b}: what moved on both - {seen}", flush=True)
        best = None
        # the pairs that agree, found once with straight lenses; each lens pair of the grid then fitted on them
        H0, inl0, _ = fit_pair(T[a], T[b], mv, lag, 0.0, 0.0, aspect[a], aspect[b], rng, rel_thr=0.9)
        agree = [mv[i] for i in inl0] if H0 is not None else []
        for kA, kB in itertools.product(K_GRID, K_GRID) if len(agree) >= MIN_COWS else []:
            corr = pair_points(T[a], T[b], agree, lag, kA, kB, aspect[a], aspect[b])
            H, inl, _ = refit(corr, homography(np.concatenate([c[0] for c in corr]),
                                               np.concatenate([c[1] for c in corr])), list(range(len(corr))))
            share, n, chance = agreement(T[a], T[b], H, lag, kA, kB, aspect[a], aspect[b])
            if best is None or (share - chance, len(inl)) > (best["share"] - best["chance"], best["inliers"]):
                best = {"H": H, "inliers": len(inl), "share": share, "chance": chance, "checked": n, "kA": kA,
                        "kB": kB, "lag": lag, "moving": agree}
        if best is None or best["inliers"] < MIN_COWS or best["share"] < 0.35 or best["share"] - best["chance"] < 0.2:
            why = f"{len(mv)} cows moving in both, {MIN_COWS} needed" if best is None else \
                f"best: {best['inliers']} cows, {best['share']:.0%} of the other's cows land on one - " \
                f"{best['chance']:.0%} would by chance alone"
            print(f"[calib] {a}-{b}: no shared floor found ({why})", flush=True)
            continue
        print(f"[calib] {a}-{b}: shared floor - {best['inliers']} moving cows agree; {best['share']:.0%} of {b}'s "
              f"{best['checked']} cow-seconds over {a}'s picture land on one of {a}'s cows ({best['chance']:.0%} "
              f"would by chance); lens {a} k={best['kA']}, {b} k={best['kB']}", flush=True)
        found[(a, b)] = best
    if not found:
        sys.exit("[calib] no camera pair with shared floor - nothing written. A longer recording (an hour or more, "
                 "with cows getting up and walking through the overlap: feeding time) helps")
    # one floor plan: from a reference camera along the pairs found (B -> A homographies chained)
    graph = collections.defaultdict(list)
    for (a, b), r in found.items():
        graph[a].append((b, r["H"]))                              # maps b -> a
        graph[b].append((a, np.linalg.inv(r["H"])))
    ref = next(iter(found))[0]
    to_ref = {ref: np.eye(3)}
    lens = {}
    for (a, b), r in found.items():
        lens.setdefault(a, r["kA"]), lens.setdefault(b, r["kB"])
    queue = [ref]
    while queue:
        x = queue.pop(0)
        for y, Hyx in graph[x]:
            if y not in to_ref:
                to_ref[y] = to_ref[x] @ Hyx
                queue.append(y)
    R = straighten(T[ref], lens[ref], aspect[ref])
    # the link distance: how far one cow's two points lie apart on the plan (the moving pairs that agree)
    spread = []
    for (a, b), r in found.items():
        corr = pair_points(T[a], T[b], r["moving"], r["lag"], r["kA"], r["kB"], aspect[a], aspect[b])
        for i in np.flatnonzero(pair_errors(r["H"], corr) < 0.6):
            pb, pa, _, _ = corr[i]
            qa = apply_h(R @ to_ref[a], pa)
            qb = apply_h(R @ to_ref[a] @ r["H"], pb)
            spread += list(np.hypot(*(qa - qb).T))
    link = float(np.clip(1.5 * np.percentile(spread, 90), 80, 300)) if spread else 160.0
    calib = {}
    for cam, Hc in to_ref.items():
        k, ar = lens[cam], aspect[cam]
        pts = [p for tr in T[cam].values() for p in tr.values()]
        uv = np.array([p_[:2] for p_ in pts])
        lo, hi = np.percentile(uv, 5, 0), np.percentile(uv, 95, 0)
        fp = []
        for u, v in grid_points((lo, hi)):
            q = apply_h(R @ Hc, np.array([undistort(u, v, k, ar)]))[0]
            fp.append([round(float(u), 4), round(float(v), 4), round(float(q[0]), 1), round(float(q[1]), 1)])
        calib[cam] = {"floor_points": fp, "distortion": k, "aspect": round(ar, 4)}
    pics = []
    for (a, b), r in found.items():
        pic = draw(os.path.join(args.out, f"{a}_{b}.jpg"), videos, a, b, T[a], T[b], r["H"], r["lag"],
                   r["kA"], r["kB"], aspect[a], aspect[b])
        if pic:
            pics.append(pic)
    report = {"pairs": {f"{a}:{b}": {k: v for k, v in r.items() if k not in ("H", "moving")} | {"moving": len(r["moving"])}
                        for (a, b), r in found.items()},
              "reference": ref, "link_dist": round(link), "cameras": calib}
    with open(os.path.join(args.out, "calib.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[calib] one floor plan for {', '.join(sorted(calib))} (about cm, from {ref}); link_dist {link:.0f}")
    if pics:
        print("[calib] look: " + ", ".join(pics) + " - the red crosses should sit on the green circles")
    if args.apply:
        apply_config(args.config, calib, link)
        print(f"[calib] {args.config}: floor_points, distortion for {', '.join(sorted(calib))}; [mct] enabled, "
              f"mode link, link_dist {link:.0f} (the old file: {args.config}.bak)")
    else:
        print(f"[calib] -> {os.path.join(args.out, 'calib.json')} (--apply writes it into {args.config})")


if __name__ == "__main__":
    sys.exit(main())
