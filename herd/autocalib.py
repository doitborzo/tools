#!/usr/bin/env python3
"""Cameras that share floor, calibrated by the cows themselves - nothing measured in the barn.

[mct] link joins one cow's tracks on two cameras by where she stands on the floor, which needs every
camera's picture mapped onto one floor plan (floor_points). Marks measured by hand are not needed:
a cow that gets up, walks or turns does it at the same moment on every camera that sees her. From a
run on recordings (herd.py trial, or herd.py run --start) - the boxes of every second of every camera:

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

    python herd/herd.py autocalib --config barn.toml --store trial/pass2/herd.sqlite --videos rec --out calib
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

from common import homography, load_config, recordings, undistort  # noqa: E402

K_GRID = (0.0, -0.1, -0.2, -0.3)


# ------------------------------------------------------------------ data

def load_tracks(store, cam):
    """{track: {second: (u, v, box height, box width)}} - u v the box's bottom centre (0-1)."""
    out = collections.defaultdict(dict)
    for track, ts, x1, y1, x2, y2 in store.query(
            "SELECT track, ts, x1, y1, x2, y2 FROM seconds WHERE cam = ?", (cam,)):
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


def moving_pairs(TA, TB, max_lag, min_overlap=20, min_corr=0.5):
    """Tracks of camera A and B that move together -> (lag of B behind A in s, [(track A, track B, corr)])."""
    secs = [s for T in (TA, TB) for tr in T.values() for s in tr]
    if not secs:
        return 0, []
    t0, n = min(secs), max(secs) - min(secs) + 1
    MA = {k: motion(v, t0, n) for k, v in TA.items() if len(v) >= min_overlap}
    MB = {k: motion(v, t0, n) for k, v in TB.items() if len(v) >= min_overlap}
    MA = {k: m for k, m in MA.items() if np.nanstd(m) > 0.02}
    MB = {k: m for k, m in MB.items() if np.nanstd(m) > 0.02}
    # the clocks' lag: voted by the tracks that move most (every lag tried on them only)
    top = lambda M: dict(sorted(M.items(), key=lambda kv: -np.nanstd(kv[1]) * np.sum(~np.isnan(kv[1])))[:15])
    votes = collections.Counter()
    for (ka, a), (kb, b) in itertools.product(top(MA).items(), top(MB).items()):
        cs = [(c, lag) for lag in range(-max_lag, max_lag + 1) if (c := corr_at(a, b, lag, min_overlap)) is not None]
        if cs and max(cs)[0] >= min_corr:
            c, lag = max(cs)
            votes[lag] += c
    if not votes:
        return 0, []
    lag = max(votes, key=lambda x: (votes[x], -abs(x)))
    pairs = []                                       # every pair at that lag (a second either way)
    for (ka, a), (kb, b) in itertools.product(MA.items(), MB.items()):
        cs = [c for d in (-1, 0, 1) if (c := corr_at(a, b, lag + d, min_overlap)) is not None]
        if cs and max(cs) >= min_corr:
            pairs.append((ka, kb, max(cs)))
    return lag, sorted(pairs, key=lambda p: -p[2])


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
    that lands on one of A's cows (within rel_thr of her height). -> (share, how many)."""
    byA, byB = collections.defaultdict(list), collections.defaultdict(list)
    for tr in TA.values():
        for s, p in tr.items():
            byA[s].append(p)
    for tr in TB.values():
        for s, p in tr.items():
            byB[s - lag].append(p)
    hit = n = 0
    for s, pb in byB.items():
        pa = byA.get(s)
        if not pa:
            continue
        a, b = np.array(pa), np.array(pb)
        qa = np.c_[undistort(a[:, 0], a[:, 1], kA, arA)]
        qb = apply_h(H, np.c_[undistort(b[:, 0], b[:, 1], kB, arB)])
        inside = (qb[:, 0] > 0.02) & (qb[:, 0] < 0.98) & (qb[:, 1] > 0.02) & (qb[:, 1] < 0.98)
        for q in qb[inside]:
            d = np.hypot(*(qa - q).T) / a[:, 2]
            hit += d.min() < rel_thr
            n += 1
    return (hit / n if n else 0.0), n


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


def draw(path, videos, cam_a, cam_b, TA, TB, H, lag, kA, kB, arA, arB):
    """B's cows of one second drawn onto A's picture of that second (red), A's own (green)."""
    import cv2
    counts = collections.Counter(s for tr in TA.values() for s in tr) + collections.Counter(
        s - lag for tr in TB.values() for s in tr)
    if not counts or cam_a not in videos:
        return None
    s = max(counts, key=counts.get)
    vpath, start = videos[cam_a]
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, s - start) * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--store", required=True, help="a run on recordings (herd.py trial: trial/pass2/herd.sqlite)")
    p.add_argument("--videos", default=None, help="the recordings (for the pictures, and their sizes)")
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
    store = Store(args.store)
    T = {c: load_tracks(store, c) for c in ids}
    videos, aspect = {}, {c: 16 / 9 for c in ids}
    if args.videos:
        recs, _ = recordings(args.videos, ids, cfg["farm"]["timezone_offset_hours"])
        for cam, path, ts in recs:
            if cam not in videos:
                videos[cam] = (path, ts)
                cap = cv2.VideoCapture(path)
                w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                if w and h:
                    aspect[cam] = w / h
    disjoint = {frozenset(x) for x in cfg["mct"].get("disjoint", [])}
    pairs = [tuple(x.split(":")) for x in args.pairs] if args.pairs else \
        [(a, b) for a, b in itertools.combinations(ids, 2) if frozenset((a, b)) not in disjoint]
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    found = {}
    for a, b in pairs:
        if not T.get(a) or not T.get(b):
            print(f"[calib] {a}-{b}: no boxes of {a if not T.get(a) else b} in {args.store}")
            continue
        lag, mv = moving_pairs(T[a], T[b], args.max_lag)
        print(f"[calib] {a}-{b}: {len(mv)} tracks moving together (clock lag {lag:+d} s)", flush=True)
        best = None
        # the pairs that agree, found once with straight lenses; each lens pair of the grid then fitted on them
        H0, inl0, _ = fit_pair(T[a], T[b], mv, lag, 0.0, 0.0, aspect[a], aspect[b], rng, rel_thr=0.9)
        agree = [mv[i] for i in inl0] if H0 is not None else []
        for kA, kB in itertools.product(K_GRID, K_GRID) if len(agree) >= 3 else []:
            corr = pair_points(T[a], T[b], agree, lag, kA, kB, aspect[a], aspect[b])
            H, inl, _ = refit(corr, homography(np.concatenate([c[0] for c in corr]),
                                               np.concatenate([c[1] for c in corr])), list(range(len(corr))))
            share, n = agreement(T[a], T[b], H, lag, kA, kB, aspect[a], aspect[b])
            if best is None or (share, len(inl)) > (best["share"], best["inliers"]):
                best = {"H": H, "inliers": len(inl), "share": share, "checked": n, "kA": kA, "kB": kB, "lag": lag,
                        "moving": agree}
        if best is None or best["inliers"] < 3 or best["share"] < 0.35:
            why = "too few cows moving in both" if best is None else \
                f"best: {best['inliers']} cows, {best['share']:.0%} of the other's cows land on one"
            print(f"[calib] {a}-{b}: no shared floor found ({why})", flush=True)
            continue
        print(f"[calib] {a}-{b}: shared floor - {best['inliers']} moving cows agree; {best['share']:.0%} of {b}'s "
              f"{best['checked']} cow-seconds over {a}'s picture land on one of {a}'s cows; lens {a} k={best['kA']}, "
              f"{b} k={best['kB']}", flush=True)
        found[(a, b)] = best
    if not found:
        sys.exit("[calib] no camera pair with shared floor - nothing written. A longer recording with cows walking "
                 "through the overlap helps (herd.py trial on it first)")
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
