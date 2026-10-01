"""Metrics.

Three error rates, not one. Posture and activity are separate axes in the
annotation, so a single number hides which of the two the model is bad at -
and they behave very differently: posture is a 2-way call on an easy visual
cue, activity is a 4-way call that needs the scene around the cow.
"""

from __future__ import annotations

import collections
import math

from cbvd import ACTIVITIES, NO_ACTIVITY, POSTURES


def wilson(successes: int, total: int, z: float = 1.96):
    """Wilson score interval. Normal approximation is useless here: drinking
    has ~6 examples in a single clip, and a Wald interval on that runs off
    the end of [0, 1]."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def score(results):
    """results: iterable of dicts with gt_posture, gt_activity, posture, activity."""
    results = list(results)
    n = len(results)

    # A prediction that never arrived is an error, and is also counted on its
    # own line: "the server fell over" and "the model cannot tell a lying cow
    # from a standing one" are different findings.
    def is_failed(r):
        return bool(r.get("error") or r.get("parse_error")
                    or r.get("posture") is None or r.get("activity") is None)

    failed = [r for r in results if is_failed(r)]
    scored = [r for r in results if not is_failed(r)]

    posture_ok = sum(1 for r in scored if r["posture"] == r["gt_posture"])
    activity_ok = sum(1 for r in scored if r["activity"] == r["gt_activity"])
    exact_ok = sum(1 for r in scored
                   if r["posture"] == r["gt_posture"] and r["activity"] == r["gt_activity"])

    def rates(correct):
        errors = n - correct  # failures count against the model
        lo, hi = wilson(errors, n)
        return {"correct": correct, "n": n, "error_rate": errors / n if n else 0.0,
                "ci95": [lo, hi]}

    # The number to beat. With 80% of boxes standing, a constant answer is a
    # strong baseline, and an error rate quoted without it means nothing.
    gt_postures = collections.Counter(r["gt_posture"] for r in results)
    gt_activities = collections.Counter(r["gt_activity"] for r in results)
    maj_posture = gt_postures.most_common(1)[0] if gt_postures else (None, 0)
    maj_activity = gt_activities.most_common(1)[0] if gt_activities else (None, 0)

    per_class = {}
    for axis, values, gt_key, pred_key in (
        ("posture", POSTURES, "gt_posture", "posture"),
        ("activity", ACTIVITIES, "gt_activity", "activity"),
    ):
        for value in values:
            tp = sum(1 for r in scored if r[gt_key] == value and r[pred_key] == value)
            fp = sum(1 for r in scored if r[gt_key] != value and r[pred_key] == value)
            fn = sum(1 for r in results if r[gt_key] == value
                     and (is_failed(r) or r[pred_key] != value))
            per_class[f"{axis}:{value}"] = _prf(tp, fp, fn)

    def confusion(values, gt_key, pred_key):
        matrix = {g: {p: 0 for p in list(values) + ["<failed>"]} for g in values}
        for r in results:
            pred = "<failed>" if is_failed(r) else r[pred_key]
            if r[gt_key] in matrix and pred in matrix[r[gt_key]]:
                matrix[r[gt_key]][pred] += 1
        return matrix

    # Errors turned out not to be spread evenly over the frame, so the report
    # has to say where they are. Box area is the causal variable - how many
    # pixels of cow the model actually got - and it carries over to other
    # camera angles in a way "top half of the frame" does not.
    with_area = [r for r in results if r.get("bbox")]
    by_size = []
    if with_area:
        for r in with_area:
            x1, y1, x2, y2 = r["bbox"]
            r["_area"] = (x2 - x1) * (y2 - y1)
            r["_yc"] = (y1 + y2) / 2
        ranked = sorted(with_area, key=lambda r: r["_area"])
        size = max(1, len(ranked) // 4)
        for q in range(4):
            lo = q * size
            hi = len(ranked) if q == 3 else (q + 1) * size
            band = ranked[lo:hi]
            if not band:
                continue
            errs = sum(1 for r in band
                       if is_failed(r) or r["posture"] != r["gt_posture"]
                       or r["activity"] != r["gt_activity"])
            by_size.append({
                "quartile": q + 1,
                "n": len(band),
                "area_pct_min": 100 * band[0]["_area"],
                "area_pct_max": 100 * band[-1]["_area"],
                "y_centre_median": sorted(r["_yc"] for r in band)[len(band) // 2],
                "lying_share": sum(1 for r in band if r["gt_posture"] == "lying") / len(band),
                "error_rate": errs / len(band),
            })

    # With one clip this is a single row and pointless; across a split it says
    # whether a number is a property of the model or of one camera angle.
    by_clip = []
    clips = collections.OrderedDict()
    for r in results:
        clips.setdefault(r.get("video_id"), []).append(r)
    if len(clips) > 1:
        for clip, rows in clips.items():
            errs = sum(1 for r in rows
                       if is_failed(r) or r["posture"] != r["gt_posture"]
                       or r["activity"] != r["gt_activity"])
            by_clip.append({
                "clip": clip,
                "n": len(rows),
                "keyframes": len({r["timestamp"] for r in rows}),
                "error_rate": errs / len(rows),
            })
        by_clip.sort(key=lambda c: -c["error_rate"])

    return {
        "n_examples": n,
        "n_keyframes": len({(r.get("video_id"), r.get("timestamp")) for r in results}),
        "n_clips": len(clips),
        "by_size": by_size,
        "by_clip": by_clip,
        "n_failed": len(failed),
        "n_missed_by_detector": sum(1 for r in results if r.get("missed_by_detector")),
        "exact_match": rates(exact_ok),
        "posture": rates(posture_ok),
        "activity": rates(activity_ok),
        "baseline": {
            "posture": {"label": maj_posture[0],
                        "error_rate": 1 - maj_posture[1] / n if n else 0.0},
            "activity": {"label": maj_activity[0],
                         "error_rate": 1 - maj_activity[1] / n if n else 0.0},
        },
        "support": {"posture": dict(gt_postures), "activity": dict(gt_activities)},
        # How often each answer was *given*, next to how often it was correct.
        # A class the model never once predicts is the loudest signal in a run
        # and is invisible in an aggregate error rate.
        "predicted": {
            "posture": dict(collections.Counter(r["posture"] for r in scored)),
            "activity": dict(collections.Counter(r["activity"] for r in scored)),
        },
        "per_class": per_class,
        "confusion": {
            "posture": confusion(POSTURES, "gt_posture", "posture"),
            "activity": confusion(ACTIVITIES, "gt_activity", "activity"),
        },
        "failures": [
            {"id": r.get("id"), "reason": r.get("error") or r.get("parse_error")
             or "empty prediction", "raw": (r.get("raw") or "")[:200]}
            for r in failed if not r.get("missed_by_detector")   # counted on their own
        ][:50],
    }


__all__ = ["score", "wilson", "NO_ACTIVITY"]
