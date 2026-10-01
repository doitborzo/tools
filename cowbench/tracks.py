"""Linking the same cow across the keyframes of a clip, and voting over it.

The annotation carries no animal identity: every keyframe is boxed
independently, and nothing says that box 4 at t=2s is the same cow as box 7 at
t=3s. But the camera is fixed and the keyframes are one second apart, so
overlap is enough to link them - and the labels confirm it, being constant
along 98% of the tracks for posture and 91% for activity.

That makes the six answers about one cow six samples of the same question. The
model is asked each of them independently and at temperature 0, so the
disagreements are not sampling noise but genuine per-frame difficulty:
occlusion by a passing animal, a bad moment for the pose. A majority vote over
the track discards exactly those.

Nothing here costs a model call. It is post-processing over results.jsonl.
"""

from __future__ import annotations

import collections


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def build(results, threshold: float = 0.5, min_length: int = 3):
    """Greedy linking from the first keyframe of each clip.

    Greedy rather than Hungarian: the cows barely move between keyframes, boxes
    that overlap by more than half are unambiguous in practice, and a matched
    box is consumed so two tracks cannot claim the same animal. Cows that
    appear only later, or that the overlap test loses, simply stay unlinked -
    they keep their per-frame answer and are reported as uncovered rather than
    silently dropped.
    """
    by_clip = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        by_clip[r.get("video_id")][r.get("timestamp")].append(r)

    # results.jsonl is written in completion order, which concurrent requests
    # shuffle from run to run. Greedy linking is order-sensitive, so without a
    # fixed order two runs over the same boxes could build different tracks
    # (it did: 412 against 411) and the vote would depend on scheduling.
    for by_ts in by_clip.values():
        for stamp in by_ts:
            by_ts[stamp].sort(key=lambda r: (r["bbox"], r["id"]))

    tracks = []
    for _clip, by_ts in sorted(by_clip.items(), key=lambda kv: str(kv[0])):
        stamps = sorted(by_ts)
        if not stamps:
            continue
        taken = {t: set() for t in stamps}
        for seed in by_ts[stamps[0]]:
            track = [seed]
            for t in stamps[1:]:
                best = None
                for j, cand in enumerate(by_ts[t]):
                    if j in taken[t]:
                        continue
                    score = iou(seed["bbox"], cand["bbox"])
                    if score > threshold and (best is None or score > best[0]):
                        best = (score, j, cand)
                if best:
                    taken[t].add(best[1])
                    track.append(best[2])
            if len(track) >= min_length:
                tracks.append(track)
    return tracks


def vote(results, threshold: float = 0.5, min_length: int = 3):
    """Return a copy of `results` with track-majority answers substituted.

    Ties fall back to the box's own answer: a 3-3 split carries no majority,
    and inventing one would be worse than leaving the frame alone.
    """
    tracks = build(results, threshold, min_length)
    replacement = {}
    for track in tracks:
        # A cow the detector missed has no answer to give and gets none: on a
        # farm the tracker links detections, and a missed cow is not one.
        track = [r for r in track if not r.get("missed_by_detector")]
        if not track:
            continue
        postures = collections.Counter(r.get("posture") for r in track)
        activities = collections.Counter(r.get("activity") for r in track)
        top_p = postures.most_common(2)
        top_a = activities.most_common(2)
        for r in track:
            p = r.get("posture")
            a = r.get("activity")
            if len(top_p) < 2 or top_p[0][1] > top_p[1][1]:
                p = top_p[0][0]
            if len(top_a) < 2 or top_a[0][1] > top_a[1][1]:
                a = top_a[0][0]
            replacement[r["id"]] = (p, a)

    out = []
    for r in results:
        if r["id"] in replacement:
            r = dict(r)
            r["posture"], r["activity"] = replacement[r["id"]]
            r["voted"] = True
        out.append(r)
    covered = sum(1 for r in out if r.get("voted"))
    return out, {"tracks": len(tracks), "covered": covered, "total": len(results)}
