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
  4. a new gid; with late_sim > 0 (an identity model's calibration), once it has a few seconds of
                         frame ids, a cow this camera lost just before it, near, whose own
                         fingerprint (the mean of her frame ids) is clearly the nearest: taken back

The same camera at the same second: two of its gids are two cows (cannot_link) once seen together
for some seconds with boxes apart (CoSeen: one cow with two boxes is not two cows). Across cameras
for pairs in [mct] disjoint - cameras that share no floor - and, calibrated, for cows apart_dist
apart on the floor; without that, one cow in an overlap is seen by two cameras at once.
"""

from __future__ import annotations

import collections
import itertools
import threading
import time

import numpy as np

from common import floor_mapper, in_polygon, iou, same_place

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
    __slots__ = ("gid", "cams", "emb", "first", "last", "last_cam", "n", "xy", "xy_ts", "acc", "acc_n",
                 "born", "late_done")

    def __init__(self, gid, ts):
        self.gid, self.cams, self.emb = gid, {}, None
        self.first = self.last = ts
        self.last_cam, self.n = None, 0
        self.xy, self.xy_ts = None, None
        self.acc, self.acc_n = None, 0            # the sum of its frame ids: the track's own fingerprint
        self.born, self.late_done = None, False    # (camera, first box) where it began; late check made

    def mean(self):
        return None if self.acc is None else self.acc / (np.linalg.norm(self.acc) + 1e-12)


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
        self.keep_s = max([self.max_age, m["reacquire_s"], m["handover_s"], m.get("still_s", 0),
                           m.get("still_long_s", 0)] +
                          [p["max_s"] for p in self.portals] +
                          [m.get("late_window_s", 60.0) + m.get("late_s", 3.0) + 2] * (m.get("late_sim", 0) > 0))
        self.floor = {cid: floor_mapper(c, m) for cid, c in self.cams.items()}
        self.store, self.apart = store, apart
        self.tracks = {}
        self.history = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.deque(maxlen=120)))
        self.seen = {}                      # cam -> (int second, gids) for the disjoint cameras' links
        self.lock = threading.Lock()
        self.n = 0
        self.stats = collections.Counter()
        self.alias = {}                     # gid taken back late -> the gid it continues
        self.late_links = []                # (late gid, earlier gid) not yet handed on (TrackletLinker)

    def root(self, g):
        """The gid a (late taken back) gid continues: what the reports see (store.track_links)."""
        while g in self.alias:
            g = self.alias[g]
        return g

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
                    t.born = (cam, list(d["box"]))
                    self.tracks[t.gid] = t
                    out[j] = t.gid
                    self.stats["new"] += 1
                self._observe(self.tracks[out[j]], cam, ts, d)
            if self.m.get("late_sim", 0) > 0:
                self._late(cam, ts, out)
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
        hold = max(m["reacquire_s"], m.get("still_s", 0), m.get("still_long_s", 0))
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
                elif (age <= m.get("still_s", 0) or (age <= m.get("still_long_s", 0) and self._was_still(seen))) \
                        and self._same_place(seen["box"], d["box"], v):
                    S[i, j] = m["w_iou"] * max(v, 0.3)            # lying still: back at the same place
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

    def _same_place(self, a, b, v):
        m = self.m
        return v >= m.get("still_iou", 1.1) or same_place(a, b, m.get("still_iou", 1.1), m.get("still_contain", 0.0))

    def _was_still(self, seen):
        """She had not moved (still_moved box heights) for still_for_s before this camera lost her."""
        return seen.get("still_since") is not None and seen["last"] - seen["still_since"] >= self.m.get("still_for_s", 30.0)

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
        prev, box = t.cams.get(cam), list(d["box"])
        anchor, since = _centre(box), ts                   # where she has stayed, since when
        if prev is not None and prev.get("anchor") is not None:
            moved = np.hypot(prev["anchor"][0] - anchor[0], prev["anchor"][1] - anchor[1])
            if moved <= self.m.get("still_moved", 0.3) * max(box[3] - box[1], 1e-6):
                anchor, since = prev["anchor"], prev["still_since"]
        t.cams[cam] = {"box": box, "last": ts, "anchor": anchor, "still_since": since}
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
            t.acc = e.copy() if t.acc is None else t.acc + e
            t.acc_n += 1
        self.history[cam][t.gid].append((ts, list(d["box"])))

    def _late(self, cam, ts, out):
        """A cow taken back late, by the track's own fingerprint: a new track of this camera, once it has
        late_n frame ids over late_s seconds, against the tracks this camera lost in the late_window_s
        before it began (near where it began, late_dist): the mean of its frame ids against theirs - a
        few seconds of a cow tell her apart far better than one frame (MmCows, the identity model: 57%
        against 36% from another camera). Taken back when at least late_sim and late_margin above the
        next: the new gid continues the old one (alias, store.track_links, late_links)."""
        m = self.m
        for t in [self.tracks[g] for g in set(out) if g in self.tracks]:
            if t.late_done or t.born is None or t.born[0] != cam:
                continue
            if t.acc_n < m.get("late_n", 3) or ts - t.first < m.get("late_s", 3.0):
                continue
            t.late_done = True
            mine = t.mean()
            (bx, by) = _centre(t.born[1])
            cands = []
            for o in self.tracks.values():
                seen = o.cams.get(cam)
                if o is t or seen is None or o.acc_n < m.get("late_n", 3) or seen["last"] >= t.first:
                    continue
                if t.first - seen["last"] > m.get("late_window_s", 60.0):
                    continue
                (ox, oy) = _centre(seen["box"])
                if ((ox - bx) ** 2 + (oy - by) ** 2) ** 0.5 > m.get("late_dist", 0.3):
                    continue
                cands.append((float(np.dot(mine, o.mean())), o))
            if not cands:
                continue
            cands.sort(key=lambda x: -x[0])
            best, o = cands[0]
            if best < m["late_sim"] or (len(cands) > 1 and best - cands[1][0] < m.get("late_margin", 0.03)):
                self.stats["late_refused"] += 1
                continue
            self._absorb(o, t, cam, ts, out, best)

    def _absorb(self, o, t, cam, ts, out, sim):
        """Track t (late) continues o: o takes over t's state, t's gid becomes an alias of o's."""
        o.cams[cam] = t.cams[cam]
        for c, v in t.cams.items():
            if c not in o.cams or v["last"] > o.cams[c]["last"]:
                o.cams[c] = v
        o.last, o.last_cam, o.n = max(o.last, t.last), t.last_cam, o.n + t.n
        if t.xy is not None:
            o.xy, o.xy_ts = t.xy, t.xy_ts
        o.emb = t.emb if t.emb is not None else o.emb
        o.acc = t.acc if o.acc is None else o.acc + t.acc
        o.acc_n += t.acc_n
        h = self.history[cam]
        if t.gid in h:
            h[o.gid].extend(h.pop(t.gid))
        del self.tracks[t.gid]
        self.alias[t.gid] = o.gid
        self.late_links.append((t.gid, o.gid))
        for j, g in enumerate(out):
            if g == t.gid:
                out[j] = o.gid
        self.stats["late_reacquired"] += 1
        if self.store is not None:
            self.store.link_tracks(ts, t.gid, o.gid)

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


# ------------------------------------------------------------------ all cameras on one floor plan

class FloorTracker:
    """All calibrated cameras on one floor plan ("one canvas"), a second at a time: every box becomes a
    point on the floor (common.floor_mapper), the points of different cameras within fuse_dist of each
    other become one cow (never two points of one camera), and one tracker follows the cows on the
    floor - so a cow seen by three cameras is one point and one track, with no handover at all.

    The spread of the points is what makes it fail, so it is cut where it comes from:
      - a box cut by the frame's edge (a cow half out of view) has its centre in the wrong place:
        it weighs edge_weight in the cow's position and is not fused on its own say;
      - a point off the pen's floor (floor_area, a polygon in floor units) is a phantom - a box on
        two cows, a cow of the next pen - and is left to its camera's own small IoU tracker;
      - the cow's height: a box's centre lands on the floor at another place for a lying (55 cm) and a
        standing (90 cm) cow, and a wrong posture throws the point a metre off. So each point is
        kept at both heights, and two cameras' points are one cow at the height where they agree,
        the posture head only a prior (height_penalty, cm, against the height it does not say).
        MmCows' labels with 30% of postures wrong: 1.18 clusters a cow, 13% mixed - with the
        posture taken as it comes, 1.31 and 20%.
    A cow lost (hidden, missed) keeps her place for still_s, the gate growing with max_speed."""

    def __init__(self, cfg, store=None, apart=None):
        m = cfg["mct"]
        self.m = m
        self.cams = {c["id"]: c for c in cfg["cameras"]}
        self.floor = {cid: floor_mapper(c, m) for cid, c in self.cams.items()}
        self.area = m.get("floor_area") or []
        self.fuse_dist = m.get("fuse_dist", 120.0)
        self.gate = m.get("fuse_gate", 150.0)
        self.speed = m.get("max_speed", 80.0)                 # floor units (cm) a second
        self.keep_s = m.get("still_s", 60.0)
        self.edge = m.get("edge_margin", 0.01)
        self.edge_w = m.get("edge_weight", 0.15)
        self.min_iou = cfg["tracker"]["iou"]
        self.tracks = {}                                       # gid -> {"xy", "v", "last", "n"}
        self.local = collections.defaultdict(dict)             # cam -> gid -> (box, last): off-floor boxes
        self.history = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.deque(maxlen=120)))
        self.store, self.apart = store, apart
        self.n = 0
        self.stats = collections.Counter()
        self.lock = threading.Lock()

    def _new(self, ts, prefix="f"):
        self.n += 1
        return f"{prefix}-{int(ts)}-{self.n}"

    def _point(self, cam, d):
        """-> ({"xy": (lying, standing), "prior": p(lying), "mix": the posture's own point}, weight),
        or (None, 0) off the floor or for an uncalibrated camera."""
        f = self.floor.get(cam)
        if f is None:
            return None, 0.0
        pl = d.get("p_lying", 0.5)
        pt = {"xy": (f(d["box"], 1.0), f(d["box"], 0.0)), "prior": pl, "mix": f(d["box"], pl)}
        if self.area and not any(in_polygon(x, y, self.area) for x, y in pt["xy"]):
            return None, 0.0
        b = d["box"]
        cut = min(b[0], b[1], 1 - b[2], 1 - b[3]) < self.edge
        return pt, (self.edge_w if cut else 1.0)

    def _fuse(self, pts):
        """pts [(cam, j, point, w)] -> clusters [([index into pts], height 0 lying / 1 standing / None)]:
        the closest pairs of different cameras first, at the height where they agree (plus the
        posture prior's penalty); a cluster never holds two points of one camera, nor two points
        fuse_dist * 1.5 apart, and keeps one height."""
        n = len(pts)
        pen = self.m.get("height_penalty", 40.0)
        Dh = []
        for h in (0, 1):
            P = np.array([p[2]["xy"][h] for p in pts], float).reshape(-1, 2)
            Dh.append(np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) if n else np.zeros((0, 0)))
        like = lambda p, h: p[2]["prior"] if h == 0 else 1 - p[2]["prior"]
        cand = sorted((Dh[h][i, k] + pen * (2 - like(pts[i], h) - like(pts[k], h)) / 2, i, k, h)
                      for i in range(n) for k in range(i + 1, n) for h in (0, 1)
                      if pts[i][0] != pts[k][0] and Dh[h][i, k] <= self.fuse_dist)
        groups = {i: ([i], None) for i in range(n)}
        owner = list(range(n))
        for _, i, k, h in cand:
            a, b = owner[i], owner[k]
            if a == b:
                continue
            (ga, ha), (gb, hb) = groups[a], groups[b]
            if (ha is not None and ha != h) or (hb is not None and hb != h):
                continue
            if {pts[x][0] for x in ga} & {pts[x][0] for x in gb}:
                continue
            if max(Dh[h][x, y] for x in ga for y in gb) > self.fuse_dist * 1.5:
                continue
            groups[a] = (ga + gb, h)
            for x in gb:
                owner[x] = a
            del groups[b]
        return list(groups.values())

    def update_all(self, ts, per_cam):
        """per_cam {cam: [{"box", "emb", "p_lying"}]} for one second -> {cam: [gid per box]}."""
        with self.lock:
            out = {cam: [None] * len(ds) for cam, ds in per_cam.items()}
            pts, off = [], []
            for cam, ds in per_cam.items():
                for j, d in enumerate(ds):
                    xy, w = self._point(cam, d)
                    (pts.append((cam, j, xy, w)) if xy is not None else off.append((cam, j)))
            fused = self._fuse(pts)
            clusters = [c for c, _ in fused]
            # a cow's place: the weighted mean of her points at the height her cameras agree on (a box
            # cut by the edge barely counts); seen by one camera, her posture's own point
            pos = []
            for c, h in fused:
                w = np.array([pts[i][3] for i in c])
                xy = np.array([pts[i][2]["xy"][h] if h is not None else pts[i][2]["mix"] for i in c], float)
                pos.append(tuple((xy * w[:, None]).sum(0) / w.sum()))
            live = [g for g, t in self.tracks.items() if ts - t["last"] <= self.keep_s]
            S = np.full((len(live), len(clusters)), NEG)
            for a, g in enumerate(live):
                t = self.tracks[g]
                dt = max(0.0, ts - t["last"])
                px, py = t["xy"][0] + t["v"][0] * min(dt, 2.0), t["xy"][1] + t["v"][1] * min(dt, 2.0)
                gate = self.gate + self.speed * dt
                for b, (x, y) in enumerate(pos):
                    dist = np.hypot(px - x, py - y)
                    if dist <= gate:
                        S[a, b] = -dist
            matched = {}
            for a, b in assign(S):
                matched[b] = live[a]
            for b, (x, y) in enumerate(pos):
                g = matched.get(b)
                if g is None:
                    g = self._new(ts)
                    self.tracks[g] = {"xy": (x, y), "v": (0.0, 0.0), "last": ts, "n": 0}
                    self.stats["new"] += 1
                else:
                    t = self.tracks[g]
                    dt = max(1e-3, ts - t["last"])
                    vx, vy = (x - t["xy"][0]) / dt, (y - t["xy"][1]) / dt
                    t["v"] = (0.5 * t["v"][0] + 0.5 * vx, 0.5 * t["v"][1] + 0.5 * vy) if dt <= 2 else (0.0, 0.0)
                    t["xy"], t["last"] = (x, y), ts
                    self.stats["followed"] += 1
                self.tracks[g]["n"] += 1
                cams_here = {pts[i][0] for i in clusters[b]}
                if len(cams_here) > 1:
                    self.stats["fused"] += 1
                for i in clusters[b]:
                    cam, j = pts[i][0], pts[i][1]
                    out[cam][j] = g
            # off the floor (or an uncalibrated camera): the camera's own IoU tracking, its own ids
            for cam, j in off:
                box = per_cam[cam][j]["box"]
                loc = self.local[cam]
                best = max(((iou(b, box), g) for g, (b, last) in loc.items() if ts - last <= 5.0), default=(0, None))
                g = best[1] if best[0] >= self.min_iou and best[1] not in out[cam] else self._new(ts, f"{cam}-off")
                loc[g] = (box, ts)
                out[cam][j] = g
                self.stats["off_floor"] += 1
            for cam, ds in per_cam.items():
                for g, d in zip(out[cam], ds):
                    self.history[cam][g].append((ts, list(d["box"])))
            self.tracks = {g: t for g, t in self.tracks.items() if ts - t["last"] <= self.keep_s}
            for cam in list(self.local):
                self.local[cam] = {g: v for g, v in self.local[cam].items() if ts - v[1] <= 5.0}
            for cam, h in self.history.items():
                for g in [g for g, q in h.items() if q and ts - q[-1][0] > 120]:
                    del h[g]
            return out

    def track_at(self, cam, ts, box):
        with self.lock:
            best, score = None, 0.0
            for gid, h in self.history.get(cam, {}).items():
                near = min(h, key=lambda x: abs(x[0] - ts), default=None)
                if near and abs(near[0] - ts) <= 2.0:
                    v = iou(near[1], box)
                    if v > score:
                        best, score = gid, v
            return best if score >= self.min_iou else None


class FusedSeconds:
    """The barn's cameras run in threads of their own: each hands in its second here and waits; when every
    camera alive (handed in within alive_s) has, or wait_s has passed, FloorTracker takes them all at
    once and each camera gets its boxes' tracks back. A camera that is down does not hold the others."""

    def __init__(self, tracker, cams, wait_s=0.8, alive_s=10.0):
        self.ft, self.wait, self.alive_s = tracker, wait_s, alive_s
        self.alive = {c: None for c in cams}
        self.pending, self.done = {}, {}
        self.cv = threading.Condition()

    def update(self, cam, ts, dets):
        deadline = time.monotonic() + self.wait
        with self.cv:
            self.alive[cam] = ts
            self.pending.setdefault(ts, {})[cam] = dets
            expect = {c for c, t in self.alive.items() if t is not None and ts - t <= self.alive_s}
            while not (ts in self.done and cam in self.done[ts]):
                ready = set(self.pending.get(ts, {})) >= expect
                if ready or time.monotonic() >= deadline:
                    per = self.pending.pop(ts, {})
                    if per:
                        self.done.setdefault(ts, {}).update(self.ft.update_all(ts, per))
                    self.cv.notify_all()
                    continue
                self.cv.wait(timeout=max(0.01, deadline - time.monotonic()))
            out = self.done[ts][cam]
            for t in [t for t in self.done if t < ts - 30]:
                del self.done[t]
        return out

    def track_at(self, cam, ts, box):
        return self.ft.track_at(cam, ts, box)

    @property
    def stats(self):
        return self.ft.stats


class TrackletLinker:
    """Each camera follows its own cows (GlobalTracker without neighbours: IoU, cows lying still held, the
    frame id to take a hidden cow back) - within a camera boxes are reliable (MmCows: IoU tracks 93%
    pure). Across cameras the tracks are joined by where the cows stand, on evidence gathered over
    time rather than a second at a time: a track of one camera and one of another are one cow when for
    link_min_s seconds they were each other's nearest on the floor (within link_dist, at the height
    where they agree, the posture a prior) and so in most (link_share) of the seconds both were seen.
    Two tracks of one camera seen at the same time are never in one cow. A floor tracker matching
    points a second at a time swapped close cows (MmCows: 33 switches a cow an hour).

    update_all(ts, {cam: dets}) -> {cam: [gid]}: the joined track as known now (online); `root(track)`
    gives the final one - what the reports see, the store keeping the links (track_links)."""

    def __init__(self, cfg, store=None, apart=None):
        m = cfg["mct"]
        self.m = m
        # each camera on its own: no neighbours, and no floor (the joining is done here, on evidence)
        solo = dict(cfg, mct=dict(m, neighbours=[], portals=[], disjoint=[]),
                    cameras=[{k: v for k, v in c.items() if k not in ("proj_mat", "floor_points")}
                             for c in cfg["cameras"]])
        self.cam_tracker = GlobalTracker(solo, None, None)
        self.floor = {c["id"]: floor_mapper(c, m) for c in cfg["cameras"]}
        self.dist = m.get("link_dist", 120.0)
        self.disjoint = {frozenset(p) for p in m.get("disjoint", [])}   # no floor shared: never compared
        self.min_s = m.get("link_min_s", 5)
        self.share = m.get("link_share", 0.6)
        # strict joining: the next cow at least link_margin further than the pair; and no join that would
        # put together two tracks of other cameras seen at the same time for link_contra_s seconds yet
        # near in less than link_contra of them (A=B and B=C must not make A=C when A and C stood apart)
        self.margin = m.get("link_margin", 0.0)
        self.contra_s = m.get("link_contra_s", 10)
        self.contra = m.get("link_contra", 0.0)
        self.pen = m.get("height_penalty", 40.0)
        self.parent = {}
        self.members = collections.defaultdict(set)           # root -> tracks
        self.life = {}                                         # track -> (cam, first, last)
        self.near = collections.Counter()                      # (a, b) -> seconds mutual nearest
        self.both = collections.Counter()                      # (a, b) -> seconds both seen
        self.store, self.apart = store, apart
        self.stats = collections.Counter()
        self.lock = threading.Lock()

    def root(self, t):
        while self.parent.get(t, t) != t:
            self.parent[t] = self.parent.get(self.parent[t], self.parent[t])
            t = self.parent[t]
        return t

    @staticmethod
    def _key(a, b):
        return (a, b) if a <= b else (b, a)

    def _mergeable(self, ra, rb):
        """Never two tracks of one camera that lived at the same time; with link_contra, never two tracks of
        other cameras that were seen together long enough and mostly apart."""
        for a in self.members[ra] or {ra}:
            ca, fa, la = self.life[a]
            for b in self.members[rb] or {rb}:
                cb, fb, lb = self.life[b]
                if ca == cb and fa <= lb and fb <= la:
                    return False
                if self.contra and ca != cb:
                    k = self._key(a, b)
                    if self.both[k] >= self.contra_s and self.near[k] < self.contra * self.both[k]:
                        self.stats["link_contradicted"] += 1
                        return False
        return True

    def _union(self, a, b, ts):
        ra, rb = self.root(a), self.root(b)
        if ra == rb:
            return
        if not self._mergeable(ra, rb):
            self.stats["link_refused"] += 1
            return
        keep, gone = (ra, rb) if self.life[ra][1] <= self.life[rb][1] else (rb, ra)
        self.parent[gone] = keep
        self.members[keep] |= (self.members.pop(gone, set()) or {gone}) | {keep}
        self.stats["linked"] += 1
        if self.store is not None:
            self.store.link_tracks(ts, gone, keep)

    def _points(self, cam, ds):
        f = self.floor.get(cam)
        if f is None:
            return None
        return [(f(d["box"], 1.0), f(d["box"], 0.0), d.get("p_lying", 0.5)) for d in ds]

    def _d(self, p, q):
        best = np.inf
        for h, like in ((0, lambda r: r[2]), (1, lambda r: 1 - r[2])):
            d = np.hypot(p[h][0] - q[h][0], p[h][1] - q[h][1]) + self.pen * (2 - like(p) - like(q)) / 2
            best = min(best, d)
        return best

    def update_all(self, ts, per_cam):
        with self.lock:
            ids = {cam: self.cam_tracker.update(cam, ts, ds) for cam, ds in per_cam.items()}
            for late, earlier in self.cam_tracker.late_links:    # a cow taken back late: before life grows
                if late in self.life and earlier in self.life:
                    self._union(late, earlier, ts)
            self.cam_tracker.late_links.clear()
            for cam, gs in ids.items():
                for g in gs:
                    c, first, _ = self.life.get(g, (cam, ts, ts))
                    self.life[g] = (cam, first, ts)
                    self.members.setdefault(g, set())
            if self.disjoint and self.apart is not None:       # cameras with no floor in common: two cows
                apart = [(self.root(a), self.root(b)) for ca, cb in itertools.combinations(sorted(ids), 2)
                         if frozenset((ca, cb)) in self.disjoint for a in ids[ca] for b in ids[cb]]
                if apart:
                    self.apart([(a, b) for a, b in apart if a != b], ts)
            pts = {cam: self._points(cam, per_cam[cam]) for cam in per_cam}
            cams = [c for c in per_cam if pts[c] is not None and pts[c]]
            for i, ca in enumerate(cams):
                for cb in cams[i + 1:]:
                    if frozenset((ca, cb)) in self.disjoint:
                        continue
                    D = np.array([[self._d(p, q) for q in pts[cb]] for p in pts[ca]])
                    for x, ga in enumerate(ids[ca]):
                        for y, gb in enumerate(ids[cb]):
                            self.both[self._key(ga, gb)] += 1
                    if D.size == 0:
                        continue
                    bx, by = D.argmin(1), D.argmin(0)
                    for x, y in enumerate(bx):
                        if by[y] == x and D[x, y] <= self.dist:
                            if self.margin:
                                row = np.delete(D[x], y)
                                col = np.delete(D[:, y], x)
                                others = np.concatenate([row, col])
                                if others.size and others.min() < D[x, y] + self.margin:
                                    continue
                            k = self._key(ids[ca][x], ids[cb][y])
                            self.near[k] += 1
                            if self.near[k] >= self.min_s and self.near[k] >= self.share * self.both[k]:
                                self._union(k[0], k[1], ts)
            return {cam: [self.root(g) for g in gs] for cam, gs in ids.items()}

    def track_at(self, cam, ts, box):
        t = self.cam_tracker.track_at(cam, ts, box)
        return None if t is None else self.root(t)
