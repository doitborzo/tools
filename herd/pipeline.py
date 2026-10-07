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
                        -> gallery: confirmed / tentative / unknown  -> table bursts
  nightly (02:00)       gallery: retire, rebuild prototypes, enroll new cows
  Tuesday 10:00         change-over starts (gallery.migration_due)

Cameras see the barn from above and partly overlap: each camera's mask (in
config) covers only the floor it owns, so a cow is counted by one camera.

    python pipeline.py --config barn.toml                     # live, RTSP urls in config
    python pipeline.py --config barn.toml --start 2026-10-07T06:00:00   # files, as recorded at that time
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import sys
import threading
import time

import numpy as np

from common import ACTIVITIES, POSTURES, crop, device, in_mask, interpolate_box, iou, load_config


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

class Models:
    """Detector, frame encoder and herd model, shared by all cameras (one GPU)."""

    def __init__(self, cfg, detector=None):
        import torch
        from model import FrameEncoder, HerdModel
        self.lock = threading.Lock()
        self.dev = device()
        ck_path = cfg["model"]["checkpoint"]
        self.model, ck = HerdModel.load(ck_path, map_location=self.dev)
        self.model.to(self.dev).eval()
        feat = ck.get("features", {})
        self.crop_size = feat.get("crop", cfg["model"]["crop"])
        self.encoder = FrameEncoder(feat.get("encoder", cfg["model"]["encoder"]),
                                    feat.get("grid", cfg["model"]["grid"])).to(self.dev).eval()
        if detector is not None:
            self.detector = detector
        else:
            import detector as det_mod
            self.detector = det_mod.Live(cfg["detector"]["weights"], cfg["detector"]["threshold"])
        self.lameness = cfg["lameness"]["enabled"]
        self.torch = torch

    def detect(self, img):
        from PIL import Image
        with self.lock:
            return [c["bbox"] for c in self.detector(Image.fromarray(img))]

    def encode(self, crops, batch=256):
        torch = self.torch
        out = []
        with self.lock, torch.no_grad():
            for i in range(0, len(crops), batch):
                x = torch.from_numpy(np.ascontiguousarray(np.stack(crops[i:i + batch]))).to(self.dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                    out.append(self.encoder(x).float())
        return torch.cat(out) if out else torch.zeros((0, self.encoder.out_dim), device=self.dev)

    def frame_heads(self, feats):
        with self.lock, self.torch.no_grad():
            o = self.model.frame(feats)
            return o["posture"].softmax(-1).cpu().numpy(), o["activity"].softmax(-1).cpu().numpy()

    def burst(self, feats, times, valid):
        torch = self.torch
        with self.lock, torch.no_grad():
            o = self.model.temporal(feats.unsqueeze(0), torch.as_tensor(times, device=self.dev).unsqueeze(0).float(),
                                    torch.as_tensor(valid, device=self.dev).unsqueeze(0))
            return {"fingerprint": o["fingerprint"][0].cpu().numpy(),
                    "quality_max": float(o["weights"][0].max()),
                    "quality_mean": float(o["quality"][0][torch.as_tensor(valid, device=self.dev)].mean()),
                    "rumination": float(o["rumination"][0].sigmoid()),
                    "posture": POSTURES[int(o["posture"][0].argmax())],
                    "activity": ACTIVITIES[int(o["activity"][0].argmax())],
                    "lameness": float(o["lameness"][0]) if self.lameness else None}


# ------------------------------------------------------------------- camera

class Camera:
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
        self.tracker = Tracker(self.id, cfg["tracker"]["iou"], cfg["tracker"]["max_age_s"])
        self.next_second = start_ts
        # Cameras burst in turn, not all at once: one GPU, an even load.
        self.next_burst = start_ts + self.burst_every * index / max(1, n_cams)
        self.stats = collections.Counter()

    def push(self, ts, frame_rgb):
        h, w = frame_rgb.shape[:2]
        if w > self.width:
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        self.buffer.append((ts, frame_rgb))
        while self.buffer and ts - self.buffer[0][0] > self.buffer_s:
            self.buffer.popleft()

    def step(self, now):
        while now >= self.next_second:
            frame = min(self.buffer, key=lambda x: abs(x[0] - self.next_second), default=None)
            if frame and abs(frame[0] - self.next_second) <= 0.5:
                self.second(self.next_second, frame[1])
            self.next_second += 1.0 / self.fps_out
        if now >= self.next_burst + self.burst_s:
            frames = [f for f in self.buffer if self.next_burst <= f[0] < self.next_burst + self.burst_s]
            if len(frames) >= 10 and frames[-1][0] - frames[0][0] >= 0.8 * self.burst_s:
                self.burst(self.next_burst, frames)
            self.next_burst += self.burst_every

    # one frame a second
    def second(self, ts, img):
        boxes = [b for b in self.models.detect(img) if in_mask(b, self.mask)]
        tracked = self.tracker.update(ts, boxes)
        if not tracked:
            return
        size = self.models.crop_size
        feats = self.models.encode([crop(img, b, size) for _, b in tracked])
        pp, pa = self.models.frame_heads(feats)
        rows = [(self.id, tid, ts, *b, POSTURES[int(p.argmax())], ACTIVITIES[int(a.argmax())],
                 float(p[POSTURES.index("lying")])) for (tid, b), p, a in zip(tracked, pp, pa)]
        self.store.seconds(rows)
        self.stats["seconds"] += 1
        self.stats["cow_seconds"] += len(rows)

    # 7 s at 25 fps
    def burst(self, t0, frames):
        step = max(1, int(round(len(frames) / (self.burst_s * self.burst_det_fps))))
        sampled = frames[::step]
        chains = []                                       # [{t: box}]
        for ts, img in sampled:
            boxes = [b for b in self.models.detect(img) if in_mask(b, self.mask)]
            used = set()
            for ch in chains:
                last_t = max(ch)
                cands = [(iou(ch[last_t], b), j) for j, b in enumerate(boxes) if j not in used]
                if cands:
                    v, j = max(cands)
                    if v >= self.cfg["tracker"]["iou"]:
                        ch[ts] = boxes[j]
                        used.add(j)
            chains += [{ts: b} for j, b in enumerate(boxes) if j not in used]
        size = self.models.crop_size
        for ch in chains:
            if len(ch) < max(2, 0.5 * len(sampled)):      # seen in too little of the burst
                continue
            times = sorted(ch)
            mid = times[len(times) // 2]
            tid = self.tracker.track_at(mid, ch[mid]) or f"{self.id}-burst-{int(t0)}-{times[0]:.0f}"
            valid = np.array([times[0] - 0.5 <= ts <= times[-1] + 0.5 for ts, _ in frames])
            crops = [crop(img, interpolate_box(ch, ts), size) for ts, img in frames]
            feats = self.models.encode(crops)
            out = self.models.burst(feats, [ts - t0 for ts, _ in frames], valid)
            area = float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in ch.values()]))
            decision = self.gallery.decide(out, area, t0, tid)
            self.store.burst((self.id, tid, t0, decision["state"], decision["cow"], decision["sim"],
                              decision["margin"], decision["p"], out["rumination"], out["posture"],
                              out["activity"], out["lameness"], out["quality_max"], int(valid.sum())))
            self.stats["bursts"] += 1
            self.stats[f"id_{decision['state']}"] += 1


class SharedGallery:
    """The gallery behind a lock, with the nightly job and the weekly change-over."""

    def __init__(self, gallery, store, cfg, start_ts):
        self.g, self.store, self.cfg = gallery, store, cfg
        self.lock = threading.Lock()
        self.next_nightly = self._next_hour(start_ts, cfg["farm"]["nightly_hour"])

    def _next_hour(self, ts, hour):
        tz = dt.timezone(dt.timedelta(hours=self.cfg["farm"]["timezone_offset_hours"]))
        t = dt.datetime.fromtimestamp(ts, tz).replace(hour=hour, minute=0, second=0, microsecond=0)
        if t.timestamp() <= ts:
            t += dt.timedelta(days=1)
        return t.timestamp()

    def decide(self, out, area, ts, track):
        with self.lock:
            if self.g.migration_due(ts):
                self.store.event(ts, "change-over", "weekly cow change-over started")
            if ts >= self.next_nightly:
                res = self.g.nightly(ts)
                self.store.event(ts, "nightly", json.dumps(res))
                self.next_nightly = self._next_hour(ts, self.cfg["farm"]["nightly_hour"])
            d = self.g.match(out["fingerprint"], out["quality_max"], out["quality_mean"], area)
            self.g.observe(d, out["fingerprint"], ts, track)
            return d

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


def run_live(camera, url, stop):
    """Reader thread fills the buffer at the camera's rate; this thread analyses
    on the wall clock, so a slow burst never makes the reader drop frames."""
    def reader():
        while not stop.is_set():
            try:
                for ts, img in frames_from(url):
                    camera.push(ts, img)
                    if stop.is_set():
                        return
            except RuntimeError as e:
                print(f"[{camera.id}] {e}; reconnecting in 5 s", flush=True)
            time.sleep(5)
    threading.Thread(target=reader, daemon=True).start()
    while not stop.is_set():
        camera.step(time.time())
        time.sleep(0.05)


def run(cfg, start=None, models=None, max_wall_s=None):
    import gallery as gallery_mod
    from store import Store
    store = Store(cfg["store"]["path"])
    ab_path = os.path.join(os.path.dirname(os.path.abspath(cfg["model"]["checkpoint"])), "abstain.json")
    abstain = json.load(open(ab_path, encoding="utf-8")) if os.path.exists(ab_path) else None
    folder = os.path.join(os.path.dirname(os.path.abspath(cfg["store"]["path"])), "gallery")
    start_ts = start if start is not None else time.time()
    shared = SharedGallery(gallery_mod.Gallery(folder, cfg, abstain), store, cfg, start_ts)
    models = models or Models(cfg)
    cams = [Camera(c, i, len(cfg["cameras"]), cfg, models, shared, store, start_ts)
            for i, c in enumerate(cfg["cameras"])]
    stop = threading.Event()
    threads = []
    for cam, c in zip(cams, cfg["cameras"]):
        target = (lambda cam=cam, c=c: run_file(cam, c["url"], start_ts)) if start is not None else \
                 (lambda cam=cam, c=c: run_live(cam, c["url"], stop))
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
