"""One track per cow across the barn's cameras, without calibration (stage B of the multi-camera plan).

Every second each camera hands its boxes in the tracking zone (`tracking_mask`, overlapping its
neighbours) with one frame id per box (model.frame_fingerprint: the herd transformer on a burst of
that one frame - the frame's guess of its burst fingerprint). GlobalTracker.update gives each box a global track id (gid), in this order:

  1. the same camera     the gid this camera had there a moment ago: IoU and the frame id
                         (w_iou * IoU + (1 - w_iou) * cos, best assignment); a cow lost for a few
                         seconds (another in front) is taken back near where she was if her frame
                         id still matches (reacquire_*), and a cow lying still whose box comes back
                         at the same place within still_s is hers whatever she looks like
  2. the floor           with calibrated cameras (proj_mat / floor_points, common.floor_mapper):
                         a cow another camera sees within handover_s, standing within handover_dist
                         of this box's floor point, the next cow at least handover_dist_margin
                         further - by place, not look: from another side a cow does not look the same
  2b. a neighbour camera  the cows of the cameras in [mct] neighbours seen in the last handover_s
                         and not by this camera: the one whose frame id is nearest, at least
                         handover_min_sim and handover_margin above the next - the band where
                         two cameras overlap passes a cow on while both still see her. A camera's
                         `handover_zone` polygons, if drawn, are where that may happen
  3. a portal            [[mct.portals]] from -> to with min_s..max_s: a blind stretch between two
                         cameras (a passage); a cow lost by `from` that long ago, by frame id at
                         handover_min_sim + portal_extra_sim; `exit` / `entry` polygons optional
  4. a new gid

The same camera at the same second: two of its gids are two cows (cannot_link) once seen together
for some seconds with boxes apart (CoSeen: one cow with two boxes is not two cows). Across cameras
for pairs in [mct] disjoint - cameras that share no floor - and, calibrated, for cows apart_dist
apart on the floor; without that, one cow in an overlap is seen by two cameras at once.
"""

from __future__ import annotations

import collections
import threading

import numpy as np

from common import floor_mapper, in_polygon, iou

NEG = -1e9


def assign(score):
    """The assignment of rows to columns with the highest summed score; NEG pairs are never made.
    Hungarian (scipy), or greedy best pairs first without it. -> [(row, col)]"""
    if score.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-np.maximum(score, -1e6))
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        pairs, ur, uc = [], set(), set()
        for flat in np.argsort(-score, axis=None):
            r, c = divmod(int(flat), score.shape[1])
            if r not in ur and c not in uc:
                pairs.append((r, c)), ur.add(r), uc.add(c)
    return [(r, c) for r, c in pairs if score[r, c] > NEG / 2]


def _centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _in(box, polys):
    if not polys:
        return True
    polys = polys if isinstance(polys[0][0], (list, tuple)) else [polys]
    cx, cy = _centre(box)
    return any(in_polygon(cx, cy, p) for p in polys)


class CoSeen:
    """Pairs of tracks seen together -> cannot_link, only with evidence: in min_s different moments,
    each time with boxes overlapping less than max_iou (two boxes on one cow, or a track that jumped
    to the cow beside her, are not two cows). MmCows: every same-second pair taken made 3-5% of the
    links wrong, and the nightly grouping split 16 cows into 64."""

    def __init__(self, min_s=5, max_iou=0.3):
        self.min_s, self.max_iou = min_s, max_iou
        self.counts = collections.Counter()
        self.done = set()
        self.lock = threading.Lock()

    def add(self, ids, boxes, weight=1):
        """One camera at one moment: its tracks and their boxes -> the pairs that became two cows now."""
        new = []
        items = sorted(zip(ids, boxes), key=lambda x: x[0])
        with self.lock:
            for i, (a, ba) in enumerate(items):
                for b, bb in items[i + 1:]:
                    k = (a, b)
                    if a == b or k in self.done or iou(ba, bb) >= self.max_iou:
                        continue
                    self.counts[k] += weight
                    if self.counts[k] >= self.min_s:
                        self.done.add(k)
                        del self.counts[k]
                        new.append(k)
        return new


class GlobalTrack:
    __slots__ = ("gid", "cams", "emb", "first", "last", "last_cam", "n", "xy", "xy_ts")

    def __init__(self, gid, ts):
        self.gid, self.cams, self.emb = gid, {}, None
        self.first = self.last = ts
        self.last_cam, self.n = None, 0
        self.xy, self.xy_ts = None, None


class GlobalTracker:
    def __init__(self, cfg, store=None, apart=None):
        m = cfg["mct"]
        self.m = m
        self.cams = {c["id"]: c for c in cfg["cameras"]}
        self.min_iou, self.max_age = cfg["tracker"]["iou"], cfg["tracker"]["max_age_s"]
        self.neighbours = collections.defaultdict(set)
        for a, b in m.get("neighbours", []):
            self.neighbours[a].add(b), self.neighbours[b].add(a)
        self.disjoint = {frozenset(p) for p in m.get("disjoint", [])}
        self.portals = list(m.get("portals", []))
        self.keep_s = max([self.max_age, m["reacquire_s"], m["handover_s"], m.get("still_s", 0)] +
                          [p["max_s"] for p in self.portals])
        self.floor = {cid: floor_mapper(c, m) for cid, c in self.cams.items()}
        self.store, self.apart = store, apart
        self.tracks = {}
        self.history = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.deque(maxlen=120)))
        self.seen = {}                      # cam -> (int second, gids) for the disjoint cameras' links
        self.lock = threading.Lock()
        self.n = 0
        self.stats = collections.Counter()

    # ---------------------------------------------------------------- one camera, one second
    def update(self, cam, ts, dets):
        """dets [{"box": [x1, y1, x2, y2] 0-1, "emb": unit vector or None, "p_lying": 0-1 (optional)}]
        -> a gid per det."""
        f = self.floor.get(cam)
        dets = [dict(d, xy=f(d["box"], d.get("p_lying", 0.5)) if f else None) for d in dets]
        with self.lock:
            out = [None] * len(dets)
            taken = set()
            self._same_camera(cam, ts, dets, out, taken)
            self._floor_handover(cam, ts, dets, out, taken)
            self._handover(cam, ts, dets, out, taken)
            self._portals(cam, ts, dets, out, taken)
            for j, d in enumerate(dets):
                if out[j] is None:
                    self.n += 1
                    t = GlobalTrack(f"g-{int(ts)}-{self.n}", ts)
                    self.tracks[t.gid] = t
                    out[j] = t.gid
                    self.stats["new"] += 1
                self._observe(self.tracks[out[j]], cam, ts, d)
            self._prune(ts)
            links = self._disjoint_links(cam, ts, out, dets)
        if links and self.apart is not None:
            self.apart(links, ts)
        return out

    def _cos(self, t, d):
        if t.emb is None or d.get("emb") is None:
            return None
        return float(np.dot(t.emb, d["emb"]))

    def _same_camera(self, cam, ts, dets, out, taken):
        m = self.m
        hold = max(m["reacquire_s"], m.get("still_s", 0))
        mine = [t for t in self.tracks.values() if cam in t.cams and ts - t.cams[cam]["last"] <= hold]
        if not mine or not dets:
            return
        S = np.full((len(mine), len(dets)), NEG)
        for i, t in enumerate(mine):
            seen = t.cams[cam]
            age = ts - seen["last"]
            for j, d in enumerate(dets):
                v, c = iou(seen["box"], d["box"]), self._cos(t, d)
                if age <= self.max_age and v >= self.min_iou:
                    S[i, j] = m["w_iou"] * v + (1 - m["w_iou"]) * (0.5 if c is None else c)
                elif age <= m.get("still_s", 0) and v >= m.get("still_iou", 1.1):
                    S[i, j] = m["w_iou"] * v                      # lying still: back at the same place
                elif age <= m["reacquire_s"] and c is not None and c >= m["reacquire_sim"]:
                    # lost a moment (hidden, missed): near where she was and still looking like her
                    (ax, ay), (bx, by) = _centre(seen["box"]), _centre(d["box"])
                    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= m["reacquire_dist"]:
                        S[i, j] = (1 - m["w_iou"]) * c
        for i, j in assign(S):
            out[j] = mine[i].gid
            taken.add(mine[i].gid)
            if ts - mine[i].cams[cam]["last"] > self.max_age or iou(mine[i].cams[cam]["box"], dets[j]["box"]) < self.min_iou:
                self.stats["reacquired"] += 1

    def _pick(self, cands, dets, open_, min_sim, out, taken, cam, ts, kind, from_cam):
        """Tracks cands against the open dets by frame id alone: the best assignment, each kept only
        if at least min_sim and handover_margin above the det's next-best candidate."""
        if not cands or not open_:
            return
        S = np.full((len(cands), len(open_)), NEG)
        for i, t in enumerate(cands):
            for k, j in enumerate(open_):
                c = self._cos(t, dets[j])
                if c is not None and c >= min_sim:
                    S[i, k] = c
        for i, k in assign(S):
            col = np.sort(S[:, k])[::-1]
            if len(col) > 1 and col[1] > NEG / 2 and col[0] - col[1] < self.m["handover_margin"]:
                self.stats[f"{kind}_ambiguous"] += 1
                continue
            t, j = cands[i], open_[k]
            out[j] = t.gid
            taken.add(t.gid)
            self.stats[kind] += 1
            src = from_cam(t)
            if self.store is not None:
                self.store.handover(ts, t.gid, src, cam, kind, None, float(S[i, k]))

    def _floor_handover(self, cam, ts, dets, out, taken):
        """By place: the cows other cameras see right now, against this camera's new boxes."""
        m = self.m
        open_ = [j for j, g in enumerate(out) if g is None and dets[j]["xy"] is not None]
        if not open_:
            return
        cands = [t for t in self.tracks.values() if t.gid not in taken and t.xy is not None
                 and ts - t.xy_ts <= m["handover_s"]
                 and not (cam in t.cams and ts - t.cams[cam]["last"] <= self.max_age)]
        if not cands:
            return
        D = np.array([[np.hypot(t.xy[0] - dets[j]["xy"][0], t.xy[1] - dets[j]["xy"][1]) for j in open_]
                      for t in cands])
        S = np.where(D <= m["handover_dist"], -D, NEG)
        for i, k in assign(S):
            others = np.concatenate([np.delete(D[:, k], i), np.delete(D[i, :], k)])
            if others.size and others.min() < D[i, k] + m["handover_dist_margin"]:
                self.stats["floor_ambiguous"] += 1
                continue
            t, j = cands[i], open_[k]
            out[j] = t.gid
            taken.add(t.gid)
            self.stats["floor"] += 1
            if self.store is not None:
                self.store.handover(ts, t.gid, t.last_cam, cam, "floor", float(D[i, k]), None)

    def _handover(self, cam, ts, dets, out, taken):
        m = self.m
        nb = self.neighbours.get(cam)
        zone = self.cams.get(cam, {}).get("handover_zone")
        open_ = [j for j, g in enumerate(out) if g is None and _in(dets[j]["box"], zone)]
        if not nb or not open_:
            return
        recent = lambda t, c: c in t.cams and ts - t.cams[c]["last"] <= m["handover_s"]
        cands = [t for t in self.tracks.values() if t.gid not in taken
                 and any(recent(t, c) for c in nb)
                 and not (cam in t.cams and ts - t.cams[cam]["last"] <= self.max_age)]
        src = lambda t: max((c for c in nb if c in t.cams), key=lambda c: t.cams[c]["last"])
        self._pick(cands, dets, open_, m["handover_min_sim"], out, taken, cam, ts, "overlap", src)

    def _portals(self, cam, ts, dets, out, taken):
        m = self.m
        for p in self.portals:
            if p["to"] != cam:
                continue
            open_ = [j for j, g in enumerate(out) if g is None and _in(dets[j]["box"], p.get("entry"))]
            if not open_:
                continue
            cands = []
            for t in self.tracks.values():
                seen = t.cams.get(p["from"])
                if t.gid in taken or seen is None or t.last_cam != p["from"]:
                    continue
                gone = ts - t.last
                if p["min_s"] <= gone <= p["max_s"] and _in(seen["box"], p.get("exit")):
                    cands.append(t)
            self._pick(cands, dets, open_, m["handover_min_sim"] + m["portal_extra_sim"], out, taken, cam, ts,
                       "portal", lambda t, f=p["from"]: f)

    def _observe(self, t, cam, ts, d):
        t.cams[cam] = {"box": list(d["box"]), "last": ts}
        t.last, t.last_cam, t.n = ts, cam, t.n + 1
        if d.get("xy") is not None:
            if t.xy is not None and t.xy_ts is not None and abs(ts - t.xy_ts) < 1.0:
                t.xy = ((t.xy[0] + d["xy"][0]) / 2, (t.xy[1] + d["xy"][1]) / 2)   # two cameras at once
            else:
                t.xy = d["xy"]
            t.xy_ts = ts
        e = d.get("emb")
        if e is not None:
            e = np.asarray(e, np.float32)
            a = self.m["ema"]
            t.emb = e if t.emb is None else (1 - a) * t.emb + a * e
            t.emb = t.emb / (np.linalg.norm(t.emb) + 1e-12)
        self.history[cam][t.gid].append((ts, list(d["box"])))

    def _prune(self, ts):
        gone = [g for g, t in self.tracks.items() if ts - t.last > self.keep_s]
        for g in gone:
            del self.tracks[g]
        for t in self.tracks.values():
            t.cams = {c: v for c, v in t.cams.items() if ts - v["last"] <= self.keep_s}
        for cam, h in self.history.items():
            for g in [g for g, q in h.items() if q and ts - q[-1][0] > 120]:
                del h[g]

    def _disjoint_links(self, cam, ts, gids, dets=None):
        """Seen in the same second by two cameras with no floor in common, or (calibrated) apart_dist
        apart on the floor: different cows."""
        sec = int(ts)
        here = {g: (d["xy"] if dets else None) for g, d in zip(gids, dets or [{}] * len(gids))}
        self.seen[cam] = (sec, here)
        far = self.m.get("apart_dist", 0)
        if not self.disjoint and not far:
            return []
        links = []
        for other, (s, g2) in self.seen.items():
            if other == cam or s != sec:
                continue
            if frozenset((cam, other)) in self.disjoint:
                links += [(a, b) for a in here for b in g2 if a != b]
            elif far:
                links += [(a, b) for a, pa in here.items() for b, pb in g2.items()
                          if a != b and pa is not None and pb is not None and np.hypot(pa[0] - pb[0], pa[1] - pb[1]) > far]
        return links

    # ---------------------------------------------------------------- bursts
    def track_at(self, cam, ts, box):
        """The gid whose box in this camera near time ts overlaps box most (a burst's chain)."""
        with self.lock:
            best, score = None, 0.0
            for gid, h in self.history.get(cam, {}).items():
                near = min(h, key=lambda x: abs(x[0] - ts), default=None)
                if near and abs(near[0] - ts) <= 2.0:
                    v = iou(near[1], box)
                    if v > score:
                        best, score = gid, v
            return best if score >= self.min_iou else None
