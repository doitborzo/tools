"""Scores for identification - what v14 reports, and how each is counted.

retrieval   query sequences against a gallery of known cows (MultiCamCows test day):
            top-1 / top-5 (the right cow first / among the first five) and mAP (every
            sequence of the right cow ranked high, not only one).
id_prf      the answers a system gives, abstentions allowed (None = "not sure"):
            precision = right answers / answers given
            recall    = right answers / everything that should have been named
            F1        = 2 P R / (P + R)
idf1        multi-camera tracking (the MOT standard): every track is matched to at most
            one cow and every cow to one track so that most detections agree; IDF1 =
            2 IDTP / (2 IDTP + IDFP + IDFN) - "the same cow kept the same id".
album       the cows enrolled at setup against the true cows: purity (share of an album
            cow's observations that are her majority cow), coverage (true cows that are
            some album cow's majority), duplicates (album cows sharing a majority).
continuity  consecutive sightings of one true cow: the track id kept, within one camera
            and when she changes camera (the hand-over).
"""

from __future__ import annotations

import collections

import numpy as np


def assignment(cost):
    """Minimum-cost one-to-one assignment -> (rows, cols). scipy if present, else greedy."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:
        rows, cols, used_r, used_c = [], [], set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = np.unravel_index(flat, cost.shape)
            if r not in used_r and c not in used_c:
                rows.append(r), cols.append(c), used_r.add(r), used_c.add(c)
        return np.array(rows, int), np.array(cols, int)


def best_mapping(pairs):
    """[(a, b)] co-occurrences -> {a: b}, one-to-one, maximising the pairs that agree."""
    counts = collections.Counter(pairs)
    A = sorted({a for a, _ in counts}, key=str)
    B = sorted({b for _, b in counts}, key=str)
    if not A or not B:
        return {}, 0
    M = np.zeros((len(A), len(B)))
    ai, bi = {a: i for i, a in enumerate(A)}, {b: j for j, b in enumerate(B)}
    for (a, b), n in counts.items():
        M[ai[a], bi[b]] = n
    r, c = assignment(-M)
    mapping = {A[i]: B[j] for i, j in zip(r, c) if M[i, j] > 0}
    return mapping, int(sum(M[i, j] for i, j in zip(r, c)))


def retrieval(q_fp, q_ids, g_fp, g_ids, ks=(1, 5)):
    """Queries (n, d) against gallery items (m, d), all L2-normalised -> top-k and mAP."""
    q_fp, g_fp = np.asarray(q_fp, np.float32), np.asarray(g_fp, np.float32)
    q_ids, g_ids = np.asarray(q_ids), np.asarray(g_ids)
    if not len(q_fp) or not len(g_fp):
        return {f"top{k}": None for k in ks} | {"mAP": None, "queries": int(len(q_fp))}
    sim = q_fp @ g_fp.T
    out = {"queries": int(len(q_fp))}
    order = np.argsort(-sim, 1)
    for k in ks:
        hits = []
        for i in range(len(q_fp)):
            seen, cows = set(), []
            for j in order[i]:
                if g_ids[j] not in seen:
                    seen.add(g_ids[j])
                    cows.append(g_ids[j])
                if len(cows) == k:
                    break
            hits.append(q_ids[i] in cows)
        out[f"top{k}"] = float(np.mean(hits))
    aps = []
    for i in range(len(q_fp)):
        rel = (g_ids[order[i]] == q_ids[i]).astype(float)
        if rel.sum() == 0:
            continue
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    out["mAP"] = float(np.mean(aps)) if aps else None
    return out


def id_prf(pred, truth):
    """pred: answers (None = not sure / not found), truth: the right cow per case."""
    n = len(truth)
    answered = [(p, t) for p, t in zip(pred, truth) if p is not None]
    right = sum(1 for p, t in answered if p == t)
    P = right / len(answered) if answered else None
    R = right / n if n else None
    F = (2 * P * R / (P + R)) if P and R else (0.0 if P is not None and R is not None else None)
    return {"cases": n, "answered": len(answered), "right": right, "precision": P, "recall": R, "f1": F,
            "answered_share": len(answered) / n if n else None}


def idf1(gt_ids, tracks, n_pred):
    """gt_ids[i]: the true cow of annotated sighting i; tracks[i]: the track matched to it
    (None = missed); n_pred: every detection the system made in the scored frames."""
    pairs = [(g, t) for g, t in zip(gt_ids, tracks) if t is not None]
    _, idtp = best_mapping(pairs)
    idfn = len(gt_ids) - idtp
    idfp = n_pred - idtp
    d = 2 * idtp + idfp + idfn
    return {"idf1": 2 * idtp / d if d else None, "idtp": idtp, "idfp": idfp, "idfn": idfn,
            "idp": idtp / (idtp + idfp) if idtp + idfp else None,
            "idr": idtp / (idtp + idfn) if idtp + idfn else None}


def album_quality(album_of, gt_of, n_true):
    """Setup sightings: album cow and true cow of each -> purity, coverage, duplicates."""
    by_album = collections.defaultdict(collections.Counter)
    for a, g in zip(album_of, gt_of):
        if a is not None:
            by_album[a][g] += 1
    total = sum(sum(c.values()) for c in by_album.values())
    majority = {a: c.most_common(1)[0][0] for a, c in by_album.items()}
    pure = sum(c.most_common(1)[0][1] for c in by_album.values())
    covered = set(majority.values())
    dup = sum(n - 1 for n in collections.Counter(majority.values()).values() if n > 1)
    mapping, _ = best_mapping([(a, g) for a, g in zip(album_of, gt_of) if a is not None])
    return {"album_cows": len(by_album), "true_cows": n_true, "purity": pure / total if total else None,
            "coverage": len(covered) / n_true if n_true else None, "duplicates": dup,
            "mapping": {str(k): v for k, v in mapping.items()}}


def continuity(sightings):
    """[(true cow, time, camera, track)] -> share of consecutive sightings of a cow that
    kept the track id: overall, within one camera, and across a camera change."""
    by_cow = collections.defaultdict(list)
    for g, t, cam, tr in sightings:
        by_cow[g].append((t, cam, tr))
    same = [0, 0]
    cross = [0, 0]
    for rows in by_cow.values():
        rows.sort(key=lambda r: (r[0], str(r[1])))
        for (t0, c0, k0), (t1, c1, k1) in zip(rows, rows[1:]):
            if k0 is None or k1 is None:
                continue
            bucket = same if c0 == c1 else cross
            bucket[0] += 1
            bucket[1] += int(k0 == k1)
    rate = lambda b: b[1] / b[0] if b[0] else None
    return {"kept_same_camera": rate(same), "kept_camera_change": rate(cross),
            "pairs_same_camera": same[0], "pairs_camera_change": cross[0],
            "kept_all": rate([same[0] + cross[0], same[1] + cross[1]])}
