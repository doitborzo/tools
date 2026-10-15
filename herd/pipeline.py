"""The barn, live: IP cameras (RTSP) or recorded files -> SQLite (store.py).

Per camera, every frame goes into a ring buffer of the last few seconds;
nothing is analysed at 25 fps except bursts:

  every 1/fps s (1 s)   detector (RT-DETRv2) -> camera mask -> tracker -> crops
                        -> frame encoder -> posture, activity  -> table seconds
  every burst_every_s   the last burst_seconds of the buffer (7 s, 25 fps):
  (60 s, staggered      detector at burst_detect_fps, linked into one box chain
   over the cameras)    per cow, interpolated to every frame -> 175 crops
                        -> frame encoder -> temporal model -> fingerprint,
                        quality, rumination, posture, activity, lameness
                        (+ the rhythm of the crops, motion.py: chewing)
                        -> ruminating yes / no (heads.json cut-off)
                        -> gallery: confirmed / tentative / unknown  -> table bursts
  nightly (02:00)       gallery: retire, rebuild prototypes, enroll new cows
  Tuesday 10:00         change-over starts (gallery.migration_due)

Cameras see the barn from above and partly overlap: each camera's mask (in
config) covers only the floor it owns, so a cow is counted by one camera.
With [mct] enabled each camera tracks in its wider tracking_mask and one
GlobalTracker (mct.py) gives a cow one track across cameras by her frame id;
seconds and bursts are still written only by the camera whose mask owns her.

    python pipeline.py --config barn.toml                     # live, RTSP urls in config
    python pipeline.py --config barn.toml --start 2026-10-07T06:00:00   # files, as recorded at that time
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import json
import math
import os
import sys
import threading
import time

import numpy as np

from common import (ACTIVITIES, POSTURES, blank_excluded, crop, device, in_mask, in_zone, interpolate_box, iou,
                    load_config)
from motion import motion_features


# ------------------------------------------------------------------ tracker

class Tracker:
    """IoU tracker for 1 fps: cows move little in a second, seen from above."""

    def __init__(self, cam, min_iou=0.3, max_age_s=5):
        self.cam, self.min_iou, self.max_age = cam, min_iou, max_age_s
        self.tracks = {}            # id -> {"box", "last"}
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=120))
        self.n = 0

    def update(self, ts, boxes):
        pairs = sorted(((iou(t["box"], b), tid, j) for tid, t in self.tracks.items()
                        for j, b in enumerate(boxes)), reverse=True)
        used_t, used_b, out = set(), set(), []
        for v, tid, j in pairs:
            if v < self.min_iou or tid in used_t or j in used_b:
                continue
            used_t.add(tid), used_b.add(j)
            self.tracks[tid] = {"box": boxes[j], "last": ts}
            out.append((tid, boxes[j]))
        for j, b in enumerate(boxes):
            if j not in used_b:
                self.n += 1
                tid = f"{self.cam}-{int(ts)}-{self.n}"
                self.tracks[tid] = {"box": b, "last": ts}
                out.append((tid, b))
        self.tracks = {k: v for k, v in self.tracks.items() if ts - v["last"] <= self.max_age}
        for tid, b in out:
            self.history[tid].append((ts, b))
        return out

    def track_at(self, ts, box):
        """The track whose box near time ts overlaps box most (for burst chains)."""
        best, score = None, 0.0
        for tid, h in self.history.items():
            near = min(h, key=lambda x: abs(x[0] - ts), default=None)
            if near and abs(near[0] - ts) <= 2.0:
                v = iou(near[1], box)
                if v > score:
                    best, score = tid, v
        return best if score >= self.min_iou else None


# ------------------------------------------------------------------- models

class GpuLock:
    """One GPU, many camera threads. The once-a-second work (urgent) goes before
    a waiting burst batch, so a big burst never holds up the 1 fps ticks for
    longer than one batch. Counts the time it is held: the GPU's busy share."""

    def __init__(self):
        self.cv = threading.Condition()
        self.busy, self.urgent_waiting = False, 0
        self.held = collections.Counter()           # kind -> seconds held

    @contextlib.contextmanager
    def __call__(self, urgent=False, kind="other"):
        with self.cv:
            self.urgent_waiting += urgent
            while self.busy or (not urgent and self.urgent_waiting):
                self.cv.wait()
            self.urgent_waiting -= urgent
            self.busy = True
        t = time.perf_counter()
        try:
            yield
        finally:
            dt_ = time.perf_counter() - t
            with self.cv:
                self.held[kind] += dt_
                self.busy = False
                self.cv.notify_all()


class Models:
    """Detector, frame encoder and herd model, shared by all cameras (one GPU)."""

    def __init__(self, cfg, detector=None):
        import torch
        from model import FrameEncoder, HerdModel
        self.gpu = GpuLock()
        self.dev = device()
        ck_path = cfg["model"]["checkpoint"]
        self.model, ck = HerdModel.load(ck_path, map_location=self.dev)
        self.model.to(self.dev).eval()
        feat = ck.get("features", {})
        self.features_meta = {k: feat.get(k, d) for k, d in (("encoder", cfg["model"]["encoder"]),
                              ("grid", cfg["model"]["grid"]), ("crop", cfg["model"]["crop"]), ("margin", 0.1),
                              ("dim", None))}
        self.crop_size = feat.get("crop", cfg["model"]["crop"])
        self.margin = feat.get("margin", 0.1)          # the context the model was trained with
        self.encoder = FrameEncoder(feat.get("encoder", cfg["model"]["encoder"]),
                                    feat.get("grid", cfg["model"]["grid"])).to(self.dev).eval()
        if detector is not None:
            self.detector = detector
        else:
            import detector as det_mod
            self.detector = det_mod.Live(cfg["detector"]["weights"], cfg["detector"]["threshold"],
                                         tiles=cfg["detector"].get("tiles"))
        self.lameness = cfg["lameness"]["enabled"]
        self.motion_dim = self.model.motion_dim
        # rumination.py's model, when the config names one: it reads the same frame vectors
        # (it must have been trained on this encoder's) and decides rumination in place of the head
        self.rum_model = None
        rp = cfg["model"].get("rumination")
        if rp:
            from model import TemporalModel
            rk = torch.load(rp, map_location=self.dev, weights_only=False)
            want = {k: rk["features"].get(k) for k in ("encoder", "grid", "crop", "margin")}
            have = {k: self.features_meta.get(k) for k in want}
            if want != have:
                raise SystemExit(f"{rp} was trained on frame vectors {want}, the herd model's are {have}")
            self.rum_model = TemporalModel(rk["in_dim"], **rk["kw"]).to(self.dev).eval()
            self.rum_model.load_state_dict(rk["state"])
            self.rum_model_thr = float(rk["threshold"])
            self.motion_dim = max(self.motion_dim, rk["kw"].get("motion_dim", 0))
        # reid.py's identity model, when the config names one: the fingerprint (bursts, and one frame a
        # second for mct.py) comes from it - taught on cows with ids - and the gallery takes its abstain.json
        self.id_model, self.id_abstain = None, None
        ip = cfg["model"].get("identity")
        if ip:
            from reid import load_identity
            self.id_model, self.id_abstain = load_identity(ip, self.features_meta, self.dev)
        # The cut-off calibrated with the model: "ruminating" per burst, summed
        # into minutes by the reports (the one that keeps the share true).
        try:
            h = json.load(open(os.path.join(os.path.dirname(os.path.abspath(ck_path)), "heads.json"), encoding="utf-8"))
            self.rum_thr = float(h.get("rumination_threshold_time", h["rumination_threshold"]))
        except (OSError, KeyError, ValueError):
            self.rum_thr = float(ck.get("rum_threshold") or 0.5)
        self.torch = torch

    def _sync(self):
        # the lock must cover the GPU work itself, not just queueing it
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize()

    def detect(self, img, urgent=False, kind="detect"):
        return self.detect_many([img], urgent=urgent, kind=kind)[0]

    def detect_many(self, imgs, batch=8, urgent=False, kind="detect"):
        from PIL import Image
        out = []
        many = getattr(self.detector, "many", None)
        step = batch if many else 1
        for i in range(0, len(imgs), step):
            pils = [Image.fromarray(x) for x in imgs[i:i + step]]
            with self.gpu(urgent, kind):
                res = many(pils) if many else [self.detector(pils[0])]
            out += [[c["bbox"] for c in r] for r in res]
        return out

    def encode(self, crops, batch=256, urgent=False, kind="encode"):
        torch = self.torch
        out = []
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(np.ascontiguousarray(np.stack(crops[i:i + batch])))
            with self.gpu(urgent, kind), torch.no_grad():
                x = x.to(self.dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                    out.append(self.encoder(x).float())
                self._sync()
        return torch.cat(out) if out else torch.zeros((0, self.encoder.out_dim), device=self.dev)

    def frame_heads(self, feats, boxes, urgent=True, kind="heads", with_id=False):
        """Posture and activity per box; with_id also the frame id (model.frame_fingerprint)."""
        from model import box_pos
        with self.gpu(urgent, kind), self.torch.no_grad():
            pos = self.torch.tensor([box_pos(b) for b in boxes], device=self.dev).float()
            o = self.model.frame(feats, pos)
            pp, pa = o["posture"].softmax(-1).cpu().numpy(), o["activity"].softmax(-1).cpu().numpy()
            if not with_id:
                return pp, pa
            if self.id_model is not None:
                x = feats.float().unsqueeze(1)
                o = self.id_model(x, x.new_zeros(len(x), 1), self.torch.ones(len(x), 1, dtype=self.torch.bool,
                                                                               device=self.dev), pos)
                return pp, pa, o["fingerprint"].float().cpu().numpy()
            return pp, pa, self.model.frame_fingerprint(feats.float(), pos).float().cpu().numpy()

    def burst(self, feats, times, valid, boxes, motion=None, urgent=False, kind="temporal"):
        from model import box_pos
        torch = self.torch
        with self.gpu(urgent, kind), torch.no_grad():
            pos = torch.tensor([box_pos(b) for b in boxes], device=self.dev).float().unsqueeze(0)
            mot = None if motion is None else torch.as_tensor(motion, device=self.dev).float().unsqueeze(0)
            o = self.model.temporal(feats.unsqueeze(0), torch.as_tensor(times, device=self.dev).unsqueeze(0).float(),
                                    torch.as_tensor(valid, device=self.dev).unsqueeze(0), pos, mot)
            out = {"fingerprint": o["fingerprint"][0].cpu().numpy(),
                   "quality_max": float(o["weights"][0].max()),
                   "quality_mean": float(o["quality"][0][torch.as_tensor(valid, device=self.dev)].mean()),
                   "rumination": float(o["rumination"][0].sigmoid()),
                   "posture": POSTURES[int(o["posture"][0].argmax())],
                   "activity": ACTIVITIES[int(o["activity"][0].argmax())],
                   "lameness": float(o["lameness"][0]) if self.lameness else None}
            if self.rum_model is not None:
                r = self.rum_model(feats.unsqueeze(0).float(), torch.as_tensor(times, device=self.dev).unsqueeze(0).float(),
                                   torch.as_tensor(valid, device=self.dev).unsqueeze(0),
                                   motion=mot if self.rum_model.motion_dim else None)
                out["rumination"] = float(r["rumination"][0].sigmoid())
            if self.id_model is not None:
                i = self.id_model(feats.unsqueeze(0).float(), torch.as_tensor(times, device=self.dev).unsqueeze(0).float(),
                                  torch.as_tensor(valid, device=self.dev).unsqueeze(0), pos)
                out["fingerprint"] = i["fingerprint"][0].cpu().numpy()
                out["quality_max"] = float(i["weights"][0].max())
                out["quality_mean"] = float(i["quality"][0][torch.as_tensor(valid, device=self.dev)].mean())
        if self.rum_model is not None:
            out["ruminating"] = bool(out["rumination"] >= self.rum_model_thr)
            return out
        # a cow that feeds or drinks is not ruminating, whatever the head says
        out["ruminating"] = bool(out["rumination"] >= self.rum_thr and out["activity"] == "none")
        return out


# ------------------------------------------------------------------- camera

def link_chains(stamped, iou_thr):
    """A burst's detections [(t, [boxes])] in time order -> chains [{t: box}]: each box goes to the
    chain whose last box overlaps it most (IoU >= iou_thr), the rest start chains of their own."""
    chains = []
    for ts, boxes in stamped:
        used = set()
        for ch in chains:
            last_t = max(ch)
            cands = [(iou(ch[last_t], b), j) for j, b in enumerate(boxes) if j not in used]
            if cands:
                v, j = max(cands)
                if v >= iou_thr:
                    ch[ts] = boxes[j]
                    used.add(j)
        chains += [{ts: b} for j, b in enumerate(boxes) if j not in used]
    return chains


_CROP_POOL = None


def cpu_pool():
    global _CROP_POOL
    if _CROP_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _CROP_POOL = ThreadPoolExecutor(max(1, min(8, (os.cpu_count() or 2) - 1)), thread_name_prefix="crop")
    return _CROP_POOL


def crops_of(pairs, size, margin):
    """[(frame, box)] -> crops, in threads: PIL's resize lets go of the GIL, and a
    burst is ~175 crops a cow - serial, the slowest part of it. Same pixels."""
    cpu_pool()
    if len(pairs) < 8:
        return [crop(img, b, size, margin) for img, b in pairs]
    return list(_CROP_POOL.map(lambda p: crop(p[0], p[1], size, margin), pairs, chunksize=16))


class Camera:
    """One camera: frames into a ring buffer (push, the reader's thread); the
    1 fps path (step_seconds) and the bursts (step_burst) each in a thread of
    their own when live, so a burst never stops the seconds."""

    def __init__(self, cam, index, n_cams, cfg, models, gallery, store, start_ts):
        s = cfg["sampling"]
        self.cam, self.cfg, self.models, self.gallery, self.store = cam, cfg, models, gallery, store
        self.id = cam["id"]
        self.mask = cam.get("mask", [])
        self.fps_out = s["fps"]
        self.burst_s, self.burst_every, self.burst_det_fps = s["burst_seconds"], s["burst_every_s"], s["burst_detect_fps"]
        self.width = s["frame_width"]
        self.buffer = collections.deque()
        self.buffer_s = self.burst_s + 3
        self.buf_lock, self.trk_lock = threading.Lock(), threading.Lock()
        self.tracker = Tracker(self.id, cfg["tracker"]["iou"], cfg["tracker"]["max_age_s"])
        self.next_second = start_ts
        # Cameras burst in turn, not all at once: one GPU, an even load.
        self.next_burst = start_ts + self.burst_every * index / max(1, n_cams)
        self.stats = collections.Counter()
        self.timings = None          # a list to record every tick and burst into (stress.py)
        self.cache = None            # barn_dataset.TrainingCache, when [training_cache] enabled
        self.glob = None             # mct.GlobalTracker, when [mct] enabled: one track per cow across cameras

    def push(self, ts, frame_rgb):
        h, w = frame_rgb.shape[:2]
        if w > self.width:
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        with self.buf_lock:
            self.buffer.append((ts, frame_rgb))
            while self.buffer and ts - self.buffer[0][0] > self.buffer_s:
                self.buffer.popleft()

    def blank(self, img):
        """The far zone greyed out before the detector (camera `exclude`)."""
        return blank_excluded(img, self.cam.get("exclude")) if self.cam.get("blank_excluded", True) else img

    def zone(self, boxes, img):
        """Boxes in the camera's working zone; the rest (far, too small, another
        camera's floor) are dropped here and never tracked or identified."""
        h, w = img.shape[:2]
        keep = [b for b in boxes if in_zone(b, self.cam, w, h, tracking=self.glob is not None)]
        self.stats["boxes_outside_zone"] += len(boxes) - len(keep)
        return keep

    def frames(self):
        with self.buf_lock:
            return list(self.buffer)

    def _time(self, kind, ts, **kw):
        if self.timings is not None:
            self.timings.append({"cam": self.id, "kind": kind, "ts": ts, **kw})

    def step(self, now):
        """Files: one thread does both, on the recording's clock."""
        self.step_seconds(now)
        self.step_burst(now)

    def step_seconds(self, now):
        while now >= self.next_second:
            t = self.next_second
            self.next_second += 1.0 / self.fps_out
            frame = min(self.frames(), key=lambda x: abs(x[0] - t), default=None)
            if frame and abs(frame[0] - t) <= 0.5:
                self.second(t, frame[1])
            else:                                         # fell behind past the buffer, or no stream
                self.stats["seconds_missed"] += 1
                self._time("second", t, missed=True)

    def step_burst(self, now):
        if now < self.next_burst + self.burst_s:
            return
        t0 = self.next_burst
        self.next_burst += self.burst_every
        frames = [f for f in self.frames() if t0 <= f[0] < t0 + self.burst_s]
        if len(frames) >= 10 and frames[-1][0] - frames[0][0] >= 0.8 * self.burst_s:
            self.burst(t0, frames)
        else:
            self.stats["bursts_missed"] += 1
            self._time("burst", t0, missed=True, frames=len(frames))

    # one frame a second
    def second(self, ts, img):
        start = time.time()
        boxes = self.zone(self.models.detect(self.blank(img), urgent=True, kind="second_detect"), img)
        if self.glob is not None:
            return self._second_global(ts, img, boxes, start)
        with self.trk_lock:
            tracked = self.tracker.update(ts, boxes)
        if len(tracked) > 1 and hasattr(self.gallery, "together"):
            self.gallery.together([tid for tid, _ in tracked], ts, [b for _, b in tracked])   # two cows, with evidence
        if tracked:
            size = self.models.crop_size
            feats = self.models.encode(crops_of([(img, b) for _, b in tracked], size, self.models.margin),
                                       urgent=True, kind="second_encode")
            pp, pa = self.models.frame_heads(feats, [b for _, b in tracked])
            rows = [(self.id, tid, ts, *b, POSTURES[int(p.argmax())], ACTIVITIES[int(a.argmax())],
                     float(p[POSTURES.index("lying")])) for (tid, b), p, a in zip(tracked, pp, pa)]
            self.store.seconds(rows)
        self.stats["seconds"] += 1
        self.stats["cow_seconds"] += len(tracked)
        self._time("second", ts, start=start, end=time.time(), cows=len(tracked))

    def _second_global(self, ts, img, boxes, start):
        """The second with mct.py: every box in the tracking zone is encoded first, its frame id
        carries the cow from camera to camera; only the boxes this camera owns (its mask) are
        counted, so a cow in an overlap is one row, under one track, whichever camera sees her."""
        rows, owned_n = [], 0
        if boxes:
            feats = self.models.encode(crops_of([(img, b) for b in boxes], self.models.crop_size, self.models.margin),
                                       urgent=True, kind="second_encode")
            pp, pa, ids = self.models.frame_heads(feats, boxes, with_id=True)
            lying = POSTURES.index("lying")
            gids = self.glob.update(self.id, ts, [{"box": b, "emb": e, "p_lying": float(p[lying])}
                                                  for b, e, p in zip(boxes, ids, pp)])
            if len(gids) > 1 and hasattr(self.gallery, "together"):
                self.gallery.together(gids, ts, boxes)              # two cows, with evidence (mct.CoSeen)
            for gid, b, p, a in zip(gids, boxes, pp, pa):
                if in_mask(b, self.mask):
                    rows.append((self.id, gid, ts, *b, POSTURES[int(p.argmax())], ACTIVITIES[int(a.argmax())],
                                 float(p[POSTURES.index("lying")])))
            owned_n = len(rows)
            if rows:
                self.store.seconds(rows)
        self.stats["seconds"] += 1
        self.stats["cow_seconds"] += owned_n
        self.stats["cow_seconds_tracked_not_owned"] += len(boxes) - owned_n
        self._time("second", ts, start=start, end=time.time(), cows=owned_n)

    def _track_of(self, ts, box):
        if self.glob is not None:
            return self.glob.track_at(self.id, ts, box)
        with self.trk_lock:
            return self.tracker.track_at(ts, box)

    # 7 s at 25 fps
    def burst(self, t0, frames):
        start = time.time()
        part = collections.Counter()
        step = max(1, int(round(len(frames) / (self.burst_s * self.burst_det_fps))))
        sampled = frames[::step]
        t = time.perf_counter()
        detections = self.models.detect_many([self.blank(img) for _, img in sampled], kind="burst_detect")
        part["detect"] += time.perf_counter() - t
        chains = link_chains([(ts, self.zone(found, img)) for (ts, img), found in zip(sampled, detections)],
                             self.cfg["tracker"]["iou"])
        size = self.models.crop_size
        cows = crops_n = rum_yes = 0
        rum_p = []
        done = []                                   # the burst's cows, decided together below
        for ch in chains:
            if len(ch) < max(2, 0.5 * len(sampled)):      # seen in too little of the burst
                continue
            times = sorted(ch)
            mid = times[len(times) // 2]
            if self.glob is not None and not in_mask(ch[mid], self.mask):
                continue                                  # tracked here, counted by the camera that owns it
            tid = self._track_of(mid, ch[mid]) or f"{self.id}-burst-{int(t0)}-{times[0]:.0f}"
            valid = np.array([times[0] - 0.5 <= ts <= times[-1] + 0.5 for ts, _ in frames])
            boxes = [interpolate_box(ch, ts) for ts, _ in frames]
            t = time.perf_counter()
            crops = crops_of([(img, b) for (_, img), b in zip(frames, boxes)], size, self.models.margin)
            part["crop"] += time.perf_counter() - t
            rel = [ts - t0 for ts, _ in frames]
            # the rhythm on the CPU while the GPU encodes the same crops
            fut = (cpu_pool().submit(motion_features, np.stack(crops), rel, valid)
                   if self.models.motion_dim else None)
            t = time.perf_counter()
            feats = self.models.encode(crops, kind="burst_encode")
            part["encode"] += time.perf_counter() - t
            t = time.perf_counter()
            motion = None
            if fut is not None:
                m, ok = fut.result()
                motion = m if ok else None
            part["motion_wait"] += time.perf_counter() - t
            t = time.perf_counter()
            out = self.models.burst(feats, rel, valid, boxes, motion)
            part["temporal"] += time.perf_counter() - t
            area = float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in ch.values()]))
            # kept until the burst's decision: only what the training cache needs, not every crop
            keep_feats = feats.float().cpu().numpy() if self.cache is not None else None
            done.append((tid, out, area, keep_feats, rel, valid, motion, boxes, crops[len(crops) // 2]))
            cows += 1
            crops_n += len(crops)
            rum_p.append(round(out["rumination"], 3))
            rum_yes += out["ruminating"]
        # the burst's cows are decided together: they were all there at once, so no cow twice
        t = time.perf_counter()
        if len(done) > 1 and hasattr(self.gallery, "together"):
            # seven seconds side by side is evidence enough, unless the two chains are one cow's
            self.gallery.together([d[0] for d in done], t0, [d[7][len(d[7]) // 2] for d in done],
                                  weight=self.cfg["identity"].get("cannot_link_min_s", 5))
        decisions = (self.gallery.decide_batch([(d[1], d[2], d[0]) for d in done], t0)
                     if hasattr(self.gallery, "decide_batch") else
                     [self.gallery.decide(d[1], d[2], t0, d[0]) for d in done])
        for (tid, out, area, feats, rel, valid, motion, boxes, mid_crop), decision in zip(done, decisions):
            self.store.burst((self.id, tid, t0, decision["state"], decision["cow"], decision["sim"],
                              decision["margin"], decision["p"], out["rumination"], out["posture"],
                              out["activity"], out["lameness"], out["quality_max"], int(valid.sum()),
                              int(out["ruminating"])))
            if self.cache is not None:
                from model import box_pos
                self.cache.add(self.id, tid, t0, feats, rel, valid, motion,
                               np.mean([box_pos(b) for b in boxes], 0), mid_crop,
                               {"state": decision["state"], "cow": decision["cow"], "sim": decision["sim"],
                                "p": decision["p"], "area": area})
            self.stats[f"id_{decision['state']}"] += 1
        part["gallery"] += time.perf_counter() - t
        self.stats["bursts"] += 1
        self._time("burst", t0, start=start, end=time.time(), ready=t0 + self.burst_s, frames=len(frames),
                   detected=len(sampled), chains=len(chains), cows=cows, crops=crops_n,
                   ruminating=rum_yes, rumination_p=rum_p,
                   parts={k: round(v, 4) for k, v in part.items()})


class SharedGallery:
    """The gallery behind a lock, with the nightly job and the weekly change-over.

    It also keeps who-was-where-when: tracks seen at the same moment are two cows (cannot_link,
    from each camera's tracker and from the chains of one burst), and a cow confirmed on one
    track is that track's for claim_s - another track seen at the same time cannot be the same
    cow unless its burst is clearly surer (claim_steal_margin); otherwise that cow is ruled out
    for it and the next one is tried."""

    def __init__(self, gallery, store, cfg, start_ts):
        self.g, self.store, self.cfg = gallery, store, cfg
        self.lock = threading.Lock()
        self.next_nightly = self._next_hour(start_ts, cfg["farm"]["nightly_hour"])
        self.cannot_link = set(store.cannot_link_pairs(start_ts - cfg["identity"]["history_days"] * 86400)) \
            if hasattr(store, "cannot_link_pairs") else set()
        self._pending_links, self._last_flush = [], start_ts
        from mct import CoSeen
        self.co_seen = CoSeen(cfg["identity"].get("cannot_link_min_s", 5), cfg["identity"].get("cannot_link_max_iou", 0.3))
        self.claims = {}                         # cow -> (track, ts, p or sim)

    def _next_hour(self, ts, hour):
        tz = dt.timezone(dt.timedelta(hours=self.cfg["farm"]["timezone_offset_hours"]))
        t = dt.datetime.fromtimestamp(ts, tz).replace(hour=hour, minute=0, second=0, microsecond=0)
        if t.timestamp() <= ts:
            t += dt.timedelta(days=1)
        return t.timestamp()

    def together(self, tracks, ts, boxes=None, weight=1):
        """Tracks seen at the same moment by one camera. With their boxes: two different cows once
        seen so in cannot_link_min_s moments with boxes apart (mct.CoSeen; weight: moments this
        sighting counts for). Without boxes: every pair at once."""
        if boxes is not None:
            pairs = self.co_seen.add(tracks, boxes, weight)
        else:
            tracks = sorted(set(tracks))
            pairs = [(a, b) for i, a in enumerate(tracks) for b in tracks[i + 1:]]
        with self.lock:
            for k in pairs:
                if k not in self.cannot_link:
                    self.cannot_link.add(k)
                    self._pending_links.append(k)
            if self._pending_links and (len(self._pending_links) >= 500 or ts - self._last_flush >= 60):
                self.store.add_cannot_link(self._pending_links, ts)
                self._pending_links, self._last_flush = [], ts

    def separate(self, pairs, ts):
        """Pairs of tracks seen at the same moment (cameras with no floor in common): two cows each."""
        with self.lock:
            for a, b in pairs:
                k = (min(a, b), max(a, b))
                if a != b and k not in self.cannot_link:
                    self.cannot_link.add(k)
                    self._pending_links.append(k)

    def apart(self, a, b):
        return a != b and (min(a, b), max(a, b)) in self.cannot_link

    def _maintain(self, ts):
        if self.g.migration_due(ts):
            self.store.event(ts, "change-over", "weekly cow change-over started")
        if ts >= self.next_nightly:
            res = self.g.nightly(ts, self.cannot_link)
            self.store.event(ts, "nightly", json.dumps(res))
            self.next_nightly = self._next_hour(ts, self.cfg["farm"]["nightly_hour"])
        hold = self.cfg["identity"].get("claim_s", 120.0)
        self.claims = {c: v for c, v in self.claims.items() if ts - v[1] <= hold}

    def decide_batch(self, items, ts):
        """items [(out, area, track)] seen at the same moment (one burst) -> a decision each."""
        steal = self.cfg["identity"].get("claim_steal_margin", 0.08)
        conf = lambda d: d["p"] if d.get("p") is not None else (d.get("sim") or 0.0)
        with self.lock:
            self._maintain(ts)
            q = [{"fingerprint": o["fingerprint"], "quality_max": o["quality_max"], "quality_mean": o["quality_mean"],
                  "area": a} for o, a, _ in items]
            ds = self.g.match_batch(q)
            for k, (o, a, tr) in enumerate(items):
                d = ds[k]
                if d["state"] != "confirmed":
                    continue
                cow = d["cow"]
                held = self.claims.get(cow)
                if held and self.apart(tr, held[0]) and conf(d) <= held[2] + steal:
                    # that cow is on a track seen at the same time, with a burst at least as sure:
                    # not her - the next cow, with every cow held elsewhere ruled out
                    ex = {c for c, (t2, _, _) in self.claims.items() if self.apart(tr, t2)}
                    ex |= {x["cow"] for j, x in enumerate(ds) if j != k and x.get("cow")}
                    d = ds[k] = self.g.match(o["fingerprint"], o["quality_max"], o["quality_mean"], a, exclude=ex)
                    self.store.event(ts, "claim-kept", json.dumps({"cow": cow, "held_by": held[0], "track": tr,
                                                                   "now": d.get("cow"), "state": d["state"]}))
                if d["state"] == "confirmed":
                    self.claims[d["cow"]] = (tr, ts, conf(d))
            for (o, a, tr), d in zip(items, ds):
                self.g.observe(d, o["fingerprint"], ts, tr)
            return ds

    def decide(self, out, area, ts, track):
        return self.decide_batch([(out, area, track)], ts)[0]

    def save(self):
        with self.lock:
            self.g.save()


# -------------------------------------------------------------------- input

def frames_from(url, start_ts=None):
    """(ts, rgb) from a file (its own clock, from start_ts) or RTSP (wall clock)."""
    import cv2
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {url}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    live = start_ts is None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield (time.time() if live else start_ts + i / fps), cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def run_file(camera, url, start_ts):
    for ts, img in frames_from(url, start_ts):
        camera.push(ts, img)
        camera.step(ts)


def run_live(camera, source, stop):
    """Three threads per camera: the reader fills the buffer at the camera's
    rate, the 1 fps path and the bursts run on the wall clock beside it, so a
    slow burst neither drops frames nor holds up the seconds. source() gives
    (ts, rgb) frames: frames_from(url) for RTSP, or a paced file (stress.py)."""
    def reader():
        while not stop.is_set():
            try:
                for ts, img in source():
                    camera.push(ts, img)
                    if stop.is_set():
                        return
            except RuntimeError as e:
                print(f"[{camera.id}] {e}; reconnecting in 5 s", flush=True)
            stop.wait(5)

    def loop(step):
        while not stop.is_set():
            try:
                step(time.time())
            except Exception as e:                   # one bad frame must not stop the camera
                import traceback
                traceback.print_exc()
                print(f"[{camera.id}] {e.__class__.__name__}: {e}", flush=True)
            stop.wait(0.05)

    threads = [threading.Thread(target=f, daemon=True, name=f"{camera.id}-{n}")
               for n, f in (("reader", reader), ("seconds", lambda: loop(camera.step_seconds)),
                            ("bursts", lambda: loop(camera.step_burst)))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()


def run(cfg, start=None, models=None, max_wall_s=None):
    import gallery as gallery_mod
    from store import Store
    store = Store(cfg["store"]["path"])
    ab_path = os.path.join(os.path.dirname(os.path.abspath(cfg["model"]["checkpoint"])), "abstain.json")
    abstain = json.load(open(ab_path, encoding="utf-8")) if os.path.exists(ab_path) else None
    folder = os.path.join(os.path.dirname(os.path.abspath(cfg["store"]["path"])), "gallery")
    start_ts = start if start is not None else time.time()
    models = models or Models(cfg)
    if getattr(models, "id_model", None) is not None:
        abstain = models.id_abstain          # the NaN model fitted on the identity model's fingerprints
    shared = SharedGallery(gallery_mod.Gallery(folder, cfg, abstain), store, cfg, start_ts)
    cams = [Camera(c, i, len(cfg["cameras"]), cfg, models, shared, store, start_ts)
            for i, c in enumerate(cfg["cameras"])]
    if cfg["mct"]["enabled"]:
        from mct import GlobalTracker
        glob = GlobalTracker(cfg, store, shared.separate)
        for cam in cams:
            cam.glob = glob
    if cfg["training_cache"]["enabled"]:
        from barn_dataset import TrainingCache
        meta = dict(models.features_meta, dim=models.features_meta.get("dim") or models.encoder.out_dim)
        cache = TrainingCache(cfg, meta)
        for cam in cams:
            cam.cache = cache
        print(f"[run] keeping every {cache.every_n}th burst for training in {cache.folder}", flush=True)
    stop = threading.Event()
    threads = []
    for cam, c in zip(cams, cfg["cameras"]):
        target = (lambda cam=cam, c=c: run_file(cam, c["url"], start_ts)) if start is not None else \
                 (lambda cam=cam, c=c: run_live(cam, lambda: frames_from(c["url"]), stop))
        th = threading.Thread(target=target, daemon=True)
        th.start()
        threads.append(th)
    t0 = time.time()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            if max_wall_s and time.time() - t0 > max_wall_s:
                stop.set()
                break
    except KeyboardInterrupt:
        stop.set()
    shared.save()
    return {c.id: dict(c.stats) for c in cams}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--start", default=None, help="files: wall-clock time the recordings start (ISO)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    start = dt.datetime.fromisoformat(args.start).timestamp() if args.start else None
    stats = run(cfg, start)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    sys.exit(main())
