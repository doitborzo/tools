#!/usr/bin/env python3
"""Can one GPU run the barn? Two tests, on CBVD-5 val videos and the real models
(RT-DETRv2 detector, DINOv2-S encoder, the herd checkpoint):

1. burst alone - one 7 s, 25 fps burst at a time, nothing else running: how long
   a burst takes (detect / crop / encode / temporal / gallery), how many cows and
   crops; and one 1 fps tick alone.
2. live - N cameras at once (5 by default), each a video played at its real 25
   fps through the same code as the barn (pipeline.run_live: reader, 1 fps path
   and bursts in threads of their own), every camera a tick a second and a 7 s
   burst a minute, bursts staggered over the cameras. Measured after a warm-up:
   ticks done and their lag (frame time -> written), bursts done and how late,
   the GPU's busy share, memory, CPU.

    python herd/herd.py stress --model /workspace/herd/run4/model.pt \\
        --detector /workspace/lora-runs/detector/best --root /workspace/cbvd5 --out /workspace/herd/stress

Keeps up when: >= 98% of the ticks are done, tick lag p99 <= --max-lag s, every
burst is done, and each burst ends before the camera's next one is due.
Clips with the most cows go first, so the load is on the heavy side. A camera
loops its clips; a burst across two clips just loses the cows that jump.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, write_json  # noqa: E402


def q(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else None


def dist(xs, ps=(50, 90, 99)):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    out = {f"p{p}": round(q(xs, p), 3) for p in ps}
    out.update(mean=round(float(np.mean(xs)), 3), max=round(float(max(xs)), 3), n=len(xs))
    return out


# ------------------------------------------------------------------ videos

def busiest_clips(root, split="val"):
    """Val clips with a video, the most cows in a keyframe first."""
    import cbvd
    boxes = cbvd.load_boxes(os.path.join(root, "annotations", f"ava_{split}_v2.1.csv"))
    per_key = collections.Counter((b.video_id, b.timestamp) for b in boxes)
    most = collections.Counter()
    for (vid, _), n in per_key.items():
        most[vid] = max(most[vid], n)
    out = []
    for vid, n in most.most_common():
        try:
            out.append((cbvd.video_path(root, vid), n))
        except FileNotFoundError:
            pass
    return out


class Paced:
    """A camera from files: its clips one after another, forever, at the clips'
    own frame rate on the wall clock (frame time = time.time(), as from RTSP)."""

    def __init__(self, paths, stop):
        self.paths, self.stop = paths, stop
        self.frames = 0
        self.late_resets = 0

    def __call__(self):
        import cv2
        t_next = time.time()
        while not self.stop.is_set():
            for path in self.paths:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                while not self.stop.is_set():
                    ok, f = cap.read()
                    if not ok:
                        break
                    wait = t_next - time.time()
                    if wait > 0:
                        self.stop.wait(wait)
                    elif wait < -1.0:              # decoding fell behind: a real camera would drop frames
                        t_next = time.time()
                        self.late_resets += 1
                    t_next += 1.0 / fps
                    self.frames += 1
                    yield time.time(), cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                cap.release()
                if self.stop.is_set():
                    return


def first_seconds(path, seconds, t0):
    import pipeline
    frames = []
    for ts, img in pipeline.frames_from(path, t0):
        if ts - t0 >= seconds:
            break
        frames.append((ts, img))
    return frames


# --------------------------------------------------------------------- GPU

class GpuSampler(threading.Thread):
    """nvidia-smi once a second: memory used, utilisation."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        self.index = vis if vis.isdigit() else "0"

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--id={self.index}", "--query-gpu=name,memory.used,memory.total,"
                                      "utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                name, used, total, util = [x.strip() for x in out.split(",")]
                self.rows.append((time.time(), name, float(used), float(total), float(util)))
            except Exception:
                return                              # no nvidia-smi: CPU run or a locked-down box
            self.stop.wait(1.0)

    def summary(self, t_from, t_to):
        rows = [r for r in self.rows if t_from <= r[0] <= t_to]
        if not rows:
            return {}
        return {"name": rows[0][1], "memory_total_mib": rows[0][3],
                "memory_used_mib_max": max(r[2] for r in rows),
                "utilization_pct": dist([r[4] for r in rows], (50, 90))}


# ---------------------------------------------------------------- the test

def build(args):
    import gallery as gallery_mod
    import pipeline
    from store import Store
    cfg = copy.deepcopy(load_config(args.config))
    cfg["model"]["checkpoint"] = args.model or cfg["model"]["checkpoint"]
    cfg["detector"]["weights"] = args.detector or cfg["detector"]["weights"]
    if args.threshold is not None:
        cfg["detector"]["threshold"] = args.threshold
    s = cfg["sampling"]
    s["burst_every_s"] = args.burst_every or s["burst_every_s"]
    s["burst_seconds"] = args.burst_seconds or s["burst_seconds"]
    s["fps"] = args.fps or s["fps"]
    if not cfg["model"]["checkpoint"] or not cfg["detector"]["weights"]:
        sys.exit("need --model (herd checkpoint) and --detector (RT-DETRv2 best/ folder)")
    cfg["cameras"] = [{"id": f"cam{i + 1}", "mask": []} for i in range(args.cameras)]
    if os.path.exists(args.out) and not args.keep:
        for name in ("stress.sqlite", "stress.sqlite-wal", "stress.sqlite-shm", "gallery"):
            p = os.path.join(args.out, name)
            shutil.rmtree(p) if os.path.isdir(p) else (os.remove(p) if os.path.exists(p) else None)
    os.makedirs(args.out, exist_ok=True)
    cfg["store"]["path"] = os.path.join(args.out, "stress.sqlite")
    store = Store(cfg["store"]["path"])
    ab_path = os.path.join(os.path.dirname(os.path.abspath(cfg["model"]["checkpoint"])), "abstain.json")
    abstain = json.load(open(ab_path, encoding="utf-8")) if os.path.exists(ab_path) else None
    t = time.time()
    models = pipeline.Models(cfg)
    print(f"[stress] models loaded in {time.time() - t:.0f} s; detector on {getattr(models.detector, 'device', '?')}",
          flush=True)
    shared = pipeline.SharedGallery(gallery_mod.Gallery(os.path.join(args.out, "gallery"), cfg, abstain),
                                    store, cfg, time.time())
    return cfg, models, shared, store


def burst_alone(args, cfg, models, shared, store, clips):
    """Bursts one by one, then 1 fps ticks one by one: the cost of each, uncontended."""
    import pipeline
    cam = pipeline.Camera({"id": "alone", "mask": []}, 0, 1, cfg, models, shared, store, time.time())
    cam.timings = []
    bs = cfg["sampling"]["burst_seconds"]
    picks = [p for p, _ in clips[:args.alone + 1]]
    print(f"[stress] burst alone: {len(picks) - 1} bursts (+1 warm-up)", flush=True)
    for k, path in enumerate(picks):
        t0 = time.time() - 20
        frames = first_seconds(path, bs, t0)
        if k == 0:
            cam.burst(t0, frames)                      # warm-up: CUDA kernels, allocator
            cam.timings.clear()
            continue
        cam.burst(t0, frames)
        r = cam.timings[-1]
        print(f"  {os.path.basename(path)}: {r['end'] - r['start']:.2f} s, {r['cows']} cows, {r['crops']} crops  "
              f"{r['parts']}", flush=True)
        for j in range(0, len(frames), max(1, len(frames) // 3)):   # a few 1 fps ticks on the same clip
            cam.second(frames[j][0], frames[j][1])
    bursts = [r for r in cam.timings if r["kind"] == "burst" and not r.get("missed")]
    ticks = [r for r in cam.timings if r["kind"] == "second" and not r.get("missed")]
    parts = collections.defaultdict(list)
    for r in bursts:
        for k, v in r["parts"].items():
            parts[k].append(v)
    return {
        "bursts": len(bursts),
        "burst_s": dist([r["end"] - r["start"] for r in bursts], (50, 90)),
        "burst_parts_s": {k: dist(v, (50,)) for k, v in parts.items()},
        "cows_per_burst": dist([r["cows"] for r in bursts], (50,)),
        "crops_per_burst": dist([r["crops"] for r in bursts], (50,)),
        "crops_per_s_encode": round(sum(r["crops"] for r in bursts) / max(1e-9, sum(parts["encode"])), 1),
        "frames_per_burst": dist([r["frames"] for r in bursts], (50,)),
        "tick_s": dist([r["end"] - r["start"] for r in ticks], (50, 90)),
        "cows_per_tick": dist([r["cows"] for r in ticks], (50,)),
        "runs": bursts,
    }


def live(args, cfg, models, shared, store, clips):
    import pipeline
    n = args.cameras
    per_cam = [[p for i, (p, _) in enumerate(clips) if i % n == c] or [clips[c % len(clips)][0]] for c in range(n)]
    stop = threading.Event()
    start_ts = time.time() + 3                     # the buffers fill first
    cams, sources = [], []
    timings = []
    for i, cam_cfg in enumerate(cfg["cameras"]):
        cam = pipeline.Camera(cam_cfg, i, n, cfg, models, shared, store, start_ts)
        cam.timings = timings
        cams.append(cam)
        sources.append(Paced(per_cam[i], stop))
    gpu = GpuSampler()
    gpu.start()
    threads = [threading.Thread(target=pipeline.run_live, args=(c, s, stop), daemon=True)
               for c, s in zip(cams, sources)]
    for th in threads:
        th.start()
    t_from = start_ts + args.warmup
    t_to = t_from + args.duration
    bs, every = cfg["sampling"]["burst_seconds"], cfg["sampling"]["burst_every_s"]
    print(f"[stress] live: {n} cameras, tick every {1 / cfg['sampling']['fps']:g} s, burst {bs:g} s every {every:g} s; "
          f"warm-up {args.warmup:g} s, measuring {args.duration:g} s", flush=True)
    held0 = cpu0 = None
    last = time.time()
    while time.time() < t_to:
        time.sleep(0.5)
        now = time.time()
        if held0 is None and now >= t_from:
            with models.gpu.cv:
                held0 = dict(models.gpu.held)
            cpu0, wall0 = time.process_time(), now
            frames0 = [src.frames for src in sources]
            if hasattr(models.torch, "cuda") and models.dev.type == "cuda":
                models.torch.cuda.reset_peak_memory_stats()
        if now - last >= 30:
            last = now
            done = [r for r in timings if r["kind"] == "second" and not r.get("missed") and r["ts"] >= start_ts]
            lag = [r["end"] - r["ts"] for r in done[-200:]]
            nb = sum(1 for r in timings if r["kind"] == "burst" and not r.get("missed"))
            print(f"  {now - start_ts:5.0f} s: ticks {len(done)}, lag p90 {q(lag, 90) or 0:.2f} s, bursts {nb}, "
                  f"camera fps " + " ".join(f"{s.frames / max(1e-9, now - start_ts + 3):.0f}" for s in sources),
                  flush=True)
    with models.gpu.cv:
        held1 = dict(models.gpu.held)
    cpu1, wall1 = time.process_time(), time.time()
    frames1 = [src.frames for src in sources]
    # let the bursts due in the window finish (they are counted, not the ones after)
    due = [(c.id, t0) for c in cams for t0 in burst_times(c, start_ts, every, n, t_from, t_to - bs)]
    deadline = time.time() + every
    while time.time() < deadline:
        got = {(r["cam"], round(r["ts"], 3)) for r in timings if r["kind"] == "burst"}
        if all((cid, round(t0, 3)) in got for cid, t0 in due):
            break
        time.sleep(0.5)
    stop.set()
    gpu.stop.set()
    for th in threads:
        th.join(timeout=30)
    peak = models.torch.cuda.max_memory_allocated() / 2 ** 20 if models.dev.type == "cuda" else None

    fps = cfg["sampling"]["fps"]
    expected_ticks = int(round(args.duration * fps))
    per = {}
    all_lag, all_burst_late = [], []
    for c, src, f0, f1 in zip(cams, sources, frames0, frames1):
        ticks = [r for r in timings if r["cam"] == c.id and r["kind"] == "second" and t_from <= r["ts"] < t_to]
        ok = [r for r in ticks if not r.get("missed")]
        lag = [r["end"] - r["ts"] for r in ok]
        mine = [t0 for cid, t0 in due if cid == c.id]
        recs = {round(r["ts"], 3): r for r in timings if r["cam"] == c.id and r["kind"] == "burst"}
        bursts = [recs.get(round(t0, 3)) for t0 in mine]
        done = [r for r in bursts if r and not r.get("missed")]
        late = [r["end"] - r["ready"] for r in done]
        all_lag += lag
        all_burst_late += late
        per[c.id] = {
            "clips": len(per_cam[cams.index(c)]),
            "camera_fps": round((f1 - f0) / max(1e-9, wall1 - wall0), 1),
            "decode_fell_behind": src.late_resets,
            "ticks_expected": expected_ticks, "ticks_done": len(ok),
            "ticks_missed": sum(1 for r in ticks if r.get("missed")),
            "tick_lag_s": dist(lag),
            "tick_work_s": dist([r["end"] - r["start"] for r in ok], (50, 90)),
            "cows_per_tick": dist([r["cows"] for r in ok], (50,)),
            "bursts_due": len(mine), "bursts_done": len(done),
            "bursts_missed": sum(1 for r in bursts if r is None or r.get("missed")),
            "burst_late_s": dist(late, (50, 90)),
            "burst_work_s": dist([r["end"] - r["start"] for r in done], (50, 90)),
            "cows_per_burst": dist([r["cows"] for r in done], (50,)),
        }
    window = wall1 - wall0 if held0 is not None else args.duration
    busy = {k: (held1.get(k, 0) - (held0 or {}).get(k, 0)) / window for k in held1}
    busy_total = sum(busy.values())
    tick_done = sum(p["ticks_done"] for p in per.values())
    bursts_due = sum(p["bursts_due"] for p in per.values())
    bursts_done = sum(p["bursts_done"] for p in per.values())
    checks = {
        "ticks_done_98pct": tick_done >= 0.98 * expected_ticks * n,
        f"tick_lag_p99_le_{args.max_lag:g}s": (q(all_lag, 99) or 1e9) <= args.max_lag,
        "all_bursts_done": bursts_done == bursts_due and bursts_due > 0,
        "bursts_end_before_next": (max(all_burst_late) if all_burst_late else 1e9) < every,
    }
    return {
        "cameras": n, "duration_s": args.duration, "warmup_s": args.warmup,
        "fps": fps, "burst_seconds": bs, "burst_every_s": every,
        "ticks_expected": expected_ticks * n, "ticks_done": tick_done,
        "tick_lag_s": dist(all_lag),
        "bursts_due": bursts_due, "bursts_done": bursts_done,
        "burst_late_s": dist(all_burst_late, (50, 90)),
        "gpu_busy_share": round(busy_total, 3),
        "gpu_busy_by_kind": {k: round(v, 3) for k, v in sorted(busy.items())},
        # GPU-bound estimate with 20% kept free; CPU (decode, crops) may stop it sooner
        "cameras_max_estimate": int(n * 0.8 / busy_total) if busy_total > 0 else None,
        "gpu": gpu.summary(t_from, t_to),
        "torch_peak_mib": round(peak) if peak else None,
        "cpu_cores_used": round((cpu1 - cpu0) / window, 2) if cpu0 is not None else None,
        "cpu_cores": os.cpu_count(),
        "checks": checks,
        "keeps_up": all(checks.values()),
        "per_camera": per,
    }


def burst_times(cam, start_ts, every, n, t_from, t_last):
    """The burst start times of a camera in [t_from, t_last]."""
    i = int(cam.id.replace("cam", "")) - 1
    t = start_ts + every * i / max(1, n)
    out = []
    while t <= t_last:
        if t >= t_from:
            out.append(t)
        t += every
    return out


# ------------------------------------------------------------------ report

def fmt(d, k="p50", unit="", dd=2):
    v = (d or {}).get(k)
    return "-" if v is None else f"{v:.{dd}f}{unit}"


def markdown(res):
    a, L = res.get("alone"), res.get("live")
    out = [f"# herd: one GPU, {L['cameras'] if L else '-'} cameras", "",
           f"- model: `{res['model']}`", f"- detector: `{res['detector']}` on {res['detector_device']}",
           f"- GPU: {res.get('gpu_name') or '-'}", ""]
    if a:
        out += ["## Burst alone", "",
                f"{a['bursts']} bursts of {res['burst_seconds']:g} s at 25 fps, one at a time, nothing else running.", "",
                "| | p50 | p90 | max |", "|---|---|---|---|",
                f"| burst, s | {fmt(a['burst_s'])} | {fmt(a['burst_s'], 'p90')} | {fmt(a['burst_s'], 'max')} |",
                f"| cows in a burst | {fmt(a['cows_per_burst'], unit='', dd=0)} | | {fmt(a['cows_per_burst'], 'max', '', 0)} |",
                f"| crops in a burst | {fmt(a['crops_per_burst'], unit='', dd=0)} | | {fmt(a['crops_per_burst'], 'max', '', 0)} |",
                f"| 1 fps tick, s | {fmt(a['tick_s'])} | {fmt(a['tick_s'], 'p90')} | {fmt(a['tick_s'], 'max')} |",
                f"| cows in a tick | {fmt(a['cows_per_tick'], unit='', dd=0)} | | {fmt(a['cows_per_tick'], 'max', '', 0)} |",
                "", "Burst parts, p50 s: " + ", ".join(f"{k} {fmt(v)}" for k, v in a["burst_parts_s"].items())
                + f"; encoder {a['crops_per_s_encode']:.0f} crops/s.", ""]
    if L:
        g = L.get("gpu") or {}
        out += [f"## Live: {L['cameras']} cameras at once", "",
                f"Every camera: 25 fps in, a tick every {1 / L['fps']:g} s, a {L['burst_seconds']:g} s burst every "
                f"{L['burst_every_s']:g} s (cameras in turn); {L['duration_s']:g} s measured after "
                f"{L['warmup_s']:g} s warm-up.", "",
                f"**Keeps up: {'yes' if L['keeps_up'] else 'no'}** - "
                + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in L["checks"].items()), "",
                "| | |", "|---|---|",
                f"| ticks done | {L['ticks_done']} of {L['ticks_expected']} |",
                f"| tick lag (frame -> written) p50 / p90 / p99 / max | {fmt(L['tick_lag_s'])} / "
                f"{fmt(L['tick_lag_s'], 'p90')} / {fmt(L['tick_lag_s'], 'p99')} / {fmt(L['tick_lag_s'], 'max')} |",
                f"| bursts done | {L['bursts_done']} of {L['bursts_due']} |",
                f"| burst ready -> done p50 / p90 / max | {fmt(L['burst_late_s'])} / {fmt(L['burst_late_s'], 'p90')} / "
                f"{fmt(L['burst_late_s'], 'max')} |",
                f"| GPU busy (model work) | {L['gpu_busy_share']:.0%} |",
                f"| GPU utilisation (nvidia-smi) p50 / p90 | {fmt(g.get('utilization_pct'), unit='%', dd=0)} / "
                f"{fmt(g.get('utilization_pct'), 'p90', '%', 0)} |",
                f"| GPU memory, max used / total | {g.get('memory_used_mib_max', '-')} / {g.get('memory_total_mib', '-')} MiB "
                f"(torch peak {L['torch_peak_mib'] or '-'} MiB) |",
                f"| CPU cores used | {L['cpu_cores_used']} of {L['cpu_cores']} |",
                f"| cameras one GPU could take (estimate, 20% kept free) | "
                f"{('at most ' if L['gpu_busy_share'] > 0.9 else '') + str(L['cameras_max_estimate'])} |", "",
                "GPU busy by part: " + ", ".join(f"{k} {v:.1%}" for k, v in L["gpu_busy_by_kind"].items()), "",
                "| camera | fps in | ticks done | missed | lag p50 / p99, s | cows a tick | bursts | "
                "burst work p50, s | ready -> done max, s | cows a burst |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for cid, p in L["per_camera"].items():
            out.append(f"| {cid} | {p['camera_fps']} | {p['ticks_done']}/{p['ticks_expected']} | {p['ticks_missed']} | "
                       f"{fmt(p['tick_lag_s'])} / {fmt(p['tick_lag_s'], 'p99')} | {fmt(p['cows_per_tick'], unit='', dd=1)} | "
                       f"{p['bursts_done']}/{p['bursts_due']} | {fmt(p['burst_work_s'])} | "
                       f"{fmt(p['burst_late_s'], 'max')} | {fmt(p['cows_per_burst'], 'mean', '', 1)} |")
        out += ["", "Tick lag: from the moment of the frame to its rows in the store. Burst ready -> done: from the "
                "end of the 7 s window to its rows in the store. CBVD-5 clips are side views with fewer cows than "
                "a 12-cow top-view camera; the burst cost grows with cows (one 175-crop pass each), the tick cost "
                "hardly at all.", ""]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="herd checkpoint (model.pt)")
    p.add_argument("--detector", default=None, help="RT-DETRv2 best/ folder")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--config", default=None, help="a barn TOML (optional): sampling, model, detector")
    p.add_argument("--root", default="/workspace/cbvd5")
    p.add_argument("--cameras", type=int, default=5)
    p.add_argument("--duration", type=float, default=300, help="seconds measured")
    p.add_argument("--warmup", type=float, default=30)
    p.add_argument("--fps", type=float, default=None, help="ticks a second (default from config: 1)")
    p.add_argument("--burst-every", type=float, default=None, help="s (default 60)")
    p.add_argument("--burst-seconds", type=float, default=None, help="s (default 7)")
    p.add_argument("--alone", type=int, default=5, help="bursts in the burst-alone test (0: skip)")
    p.add_argument("--no-live", action="store_true", help="only the burst-alone test")
    p.add_argument("--max-lag", type=float, default=2.0, help="s: tick lag p99 allowed")
    p.add_argument("--out", default="/workspace/herd/stress")
    p.add_argument("--keep", action="store_true", help="keep the store and gallery of an earlier test")
    args = p.parse_args(argv)

    clips = busiest_clips(args.root)
    if not clips:
        sys.exit(f"no CBVD-5 val videos under {args.root}")
    print(f"[stress] {len(clips)} val clips with video, up to {clips[0][1]} cows in a keyframe", flush=True)
    cfg, models, shared, store = build(args)
    res = {"model": cfg["model"]["checkpoint"], "detector": cfg["detector"]["weights"],
           "detector_device": getattr(models.detector, "device", "?"),
           "burst_seconds": cfg["sampling"]["burst_seconds"], "clips": len(clips)}
    if models.dev.type == "cuda":
        res["gpu_name"] = models.torch.cuda.get_device_name()
    if args.alone:
        res["alone"] = burst_alone(args, cfg, models, shared, store, clips)
    if not args.no_live:
        res["live"] = live(args, cfg, models, shared, store, clips)
    shared.save()
    json_path = os.path.join(args.out, f"stress_herd_{args.cameras}cam.json")
    md_path = json_path[:-5] + ".md"
    write_json(json_path, res)
    text = markdown(res)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("\n" + text)
    print(f"[stress] -> {md_path}")
    return 0 if not res.get("live") or res["live"]["keeps_up"] else 1


if __name__ == "__main__":
    sys.exit(main())
