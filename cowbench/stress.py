"""Many cameras at once: can one vLLM server keep up with N streams?

Each stream stands for one camera: it gets its own val clip and replays that
clip's keyframes for --duration seconds. A frame is done when every question
about it is answered:

    classify  one request per annotated cow, the cows of a frame in parallel -
              the bench's own per-cow question (with a LoRA: --answer-now)
    frame     one request per frame about every annotated cow on it (frame.py)
    detect    one request per frame, detect.py's "find every cow"

--interval 0 sends the next frame as soon as the last is done: the most the
server gives each camera. --interval S paces every camera at one frame per
S seconds: then what matters is whether frames finish inside S, or pile up.

Images are rendered and encoded before the clock starts, so what is timed is
the server, not JPEG encoding on this machine. The first --warmup seconds
are run but not counted: CUDA graphs and caches settle there.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import threading
import time

import cbvd
import detect as detect_mod
import frame as frame_mod
import render as render_mod


def _pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def run(args, client_mod, jsonl_read):
    manifest = jsonl_read(os.path.join(args.out, "manifest.jsonl"))
    if not manifest:
        raise SystemExit(f"no manifest in {args.out} - run `plan` first")
    with open(os.path.join(args.out, "plan_meta.json"), encoding="utf-8") as fh:
        plan_meta = json.load(fh)
    root = args.root or plan_meta["root"]
    frames = detect_mod.frames_of(manifest)

    # The busiest clips: a stress test should not be flattered by empty stalls.
    by_clip = {}
    for (vid, ts), cows in frames.items():
        by_clip.setdefault(vid, []).append((ts, cows))
    clips = sorted(by_clip, key=lambda v: (-sum(len(c) for _, c in by_clip[v]), int(v)))
    streams = [clips[i % len(clips)] for i in range(args.streams)]
    print(f"{args.streams} streams, task={args.task}, "
          f"{'as fast as possible' if not args.interval else f'one frame per {args.interval:g} s each'}, "
          f"{args.duration:g} s (+{args.warmup:g} s warm-up)")
    print("  clips: " + ", ".join(f"{v} ({sum(len(c) for _, c in by_clip[v]) // len(by_clip[v])} cows/frame)"
                                  for v in streams))

    t = time.perf_counter()
    prepared = {}
    for vid in dict.fromkeys(streams):
        for ts, cows in sorted(by_clip[vid]):
            path = cbvd.frame_path(root, cbvd.Box(vid, ts, 0, 0, 0, 0, "1", ()))
            if args.task == "detect":
                img = render_mod.render(path, None, mode="plain", max_width=args.max_width)
                prepared[(vid, ts)] = [(render_mod.to_data_url(img, args.jpeg_quality), img.size)]
            elif args.task == "frame":
                ordered = frame_mod.order(cows)
                img = frame_mod.render(path, ordered, args.max_width)
                prepared[(vid, ts)] = [(render_mod.to_data_url(img, args.jpeg_quality), ordered)]
            else:
                prepared[(vid, ts)] = [
                    (render_mod.to_data_url(render_mod.render(path, c["bbox"], mode="marked",
                                                              max_width=args.max_width),
                                            args.jpeg_quality), None)
                    for c in cows]
    n_img = sum(len(v) for v in prepared.values())
    print(f"  {n_img} images prepared in {time.perf_counter() - t:.0f} s")

    local = threading.local()

    def client():
        if not hasattr(local, "cli"):
            local.cli = client_mod.MuseClient(
                base_url=args.base_url, model=args.model, api_key=args.api_key,
                temperature=args.temperature, max_tokens=args.max_tokens,
                timeout=args.timeout, retries=1, answer_now=args.answer_now)
        return local.cli

    def ask(item):
        url, size = item
        t0 = time.perf_counter()
        try:
            if args.task == "detect":
                out = client().ask([url], detect_mod.PROMPT.format(w=size[0], h=size[1]),
                                   detect_mod.SCHEMA, "cows_in_frame")
            elif args.task == "frame":
                cows = size
                if args.answer_now:
                    out = client().complete([url], frame_mod.prompt(cows),
                                            prefill=frame_mod.ANSWER_PREFILL,
                                            max_tokens=24 * len(cows) + 64)
                else:
                    out = client().complete([url], frame_mod.prompt(cows), schema=frame_mod.SCHEMA,
                                            name="cows_in_frame")
                got = frame_mod.parse(out["raw"], len(cows))
                return {"seconds": time.perf_counter() - t0, "usage": out.get("usage") or {},
                        "ok": len(got) == len(cows)}
            else:
                out = client().classify([url])
            return {"seconds": time.perf_counter() - t0, "usage": out.get("usage") or {},
                    "ok": "parse_error" not in out}
        except Exception as exc:
            return {"seconds": time.perf_counter() - t0, "usage": {}, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    info = client().server_info()
    frames_log = []
    lock = threading.Lock()
    width = max(len(v) for v in prepared.values())
    start = time.perf_counter()
    count_from = start + args.warmup
    deadline = count_from + args.duration

    def stream(k, vid):
        keys = sorted(ts for ts, _ in by_clip[vid])
        with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
            i = 0
            while True:
                due = start + i * args.interval if args.interval else time.perf_counter()
                if due >= deadline:
                    break
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                t0 = time.perf_counter()
                ts = keys[i % len(keys)]
                answers = list(pool.map(ask, prepared[(vid, ts)]))
                t1 = time.perf_counter()
                with lock:
                    frames_log.append({
                        "stream": k, "clip": vid, "timestamp": ts,
                        "start": round(t0 - start, 3), "end": round(t1 - start, 3),
                        "latency": round(t1 - t0, 3), "late_by": round(max(0.0, t0 - due), 3),
                        "requests": len(answers), "failed": sum(not a["ok"] for a in answers),
                        "request_seconds": [round(a["seconds"], 3) for a in answers],
                        "prompt_tokens": sum(a["usage"].get("prompt_tokens", 0) for a in answers),
                        "completion_tokens": sum(a["usage"].get("completion_tokens", 0)
                                                 for a in answers),
                        "errors": [a["error"] for a in answers if "error" in a][:3],
                    })
                i += 1

    def progress():
        while time.perf_counter() < deadline:
            time.sleep(5)
            with lock:
                n = sum(f["end"] >= args.warmup for f in frames_log)
            phase = "warm-up" if time.perf_counter() < count_from else "measuring"
            print(f"\r  {time.perf_counter() - start:5.0f} s  {phase}  "
                  f"{n} frames counted   ", end="", flush=True)

    threads = [threading.Thread(target=stream, args=(k, v), daemon=True)
               for k, v in enumerate(streams)]
    threading.Thread(target=progress, daemon=True).start()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print()

    # Frames that finished inside the measured window, wherever they began:
    # throughput is completions per second. Requiring the start inside the
    # window too counted nothing when a frame takes longer than the window
    # minus one frame - detect with reasoning, ~3 min a frame at 12 cameras.
    counted = [f for f in frames_log
               if args.warmup <= f["end"] <= args.warmup + args.duration]
    window = args.duration
    lat = [f["latency"] for f in counted]
    req = [s for f in counted for s in f["request_seconds"]]
    per_stream = [sum(f["stream"] == k for f in counted) / window for k in range(args.streams)]
    summary = {
        "run_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model, "vllm_version": info.get("vllm_version"), "server": info,
        "task": args.task, "answer_now": args.answer_now, "streams": args.streams,
        "clips": streams, "interval": args.interval, "duration": args.duration,
        "warmup": args.warmup, "max_width": args.max_width,
        "frames": len(counted), "requests": len(req),
        "failed_requests": sum(f["failed"] for f in counted),
        "frames_per_s_total": len(counted) / window,
        "frames_per_s_per_stream": {"min": min(per_stream), "mean": sum(per_stream) / len(per_stream),
                                    "max": max(per_stream)},
        "requests_per_s": len(req) / window,
        "frame_latency_s": {q: _pct(lat, int(q[1:])) for q in ("p50", "p90", "p99")}
                           | {"max": max(lat) if lat else None},
        "request_latency_s": {q: _pct(req, int(q[1:])) for q in ("p50", "p90", "p99")},
        "prompt_tokens_per_s": sum(f["prompt_tokens"] for f in counted) / window,
        "completion_tokens_per_s": sum(f["completion_tokens"] for f in counted) / window,
    }
    if args.interval:
        late = [f["late_by"] for f in counted]
        summary["paced"] = {
            "frames_started_late": sum(x > 0.5 for x in late),
            "late_by_s": {"p50": _pct(late, 50), "max": max(late) if late else None},
            "frames_over_interval": sum(x > args.interval for x in lat),
            # Keeping up: frames finish within their slot, so none starts late.
            "keeps_up": bool(counted) and max(late) <= 0.5 * args.interval,
        }

    tag = f"stress_{args.task}_{args.streams}x" + (f"_{args.interval:g}s" if args.interval else "_max")
    with open(os.path.join(args.out, tag + "_frames.jsonl"), "w", encoding="utf-8") as fh:
        for f in frames_log:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    report = render_report(summary)
    with open(os.path.join(args.out, tag + ".md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    if len(counted) < 2 * args.streams:
        print(f"!! only {len(counted)} frames finished in the window - a frame takes about "
              f"{_pct(lat, 50) or 0:.0f} s here; a longer --duration gives steadier numbers")
    print(f"-> {os.path.join(args.out, tag + '.md')}")


def render_report(s) -> str:
    f = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"
    mode = ("as fast as possible (next frame as soon as the last is answered)"
            if not s["interval"] else f"paced, one frame per {s['interval']:g} s per camera")
    per_frame = {"classify": "one request per annotated cow",
                 "frame": "one request per frame about every annotated cow on it",
                 "detect": "one request per frame, finding the cows"}[s["task"]]
    lines = [
        f"# {s['streams']} cameras at once", "",
        f"Model `{s['model']}` on vLLM {s.get('vllm_version') or '?'}, {s['run_date']}. "
        f"Task `{s['task']}` ({per_frame}"
        + (", answer prefilled, no reasoning" if s["answer_now"] else "") + f"), "
        f"max width {s['max_width']} px. {mode}. Measured {s['duration']:g} s after "
        f"{s['warmup']:g} s warm-up.", "",
        "| | |", "|---|---|",
        f"| Frames answered | {s['frames']} ({s['requests']} requests"
        + (f", **{s['failed_requests']} failed**" if s["failed_requests"] else "") + ") |",
        f"| Frames per second, all cameras | {f(s['frames_per_s_total'], 2)} |",
        f"| Frames per second, per camera | {f(s['frames_per_s_per_stream']['mean'], 3)} "
        f"(min {f(s['frames_per_s_per_stream']['min'], 3)}, max {f(s['frames_per_s_per_stream']['max'], 3)}) |",
        f"| Requests per second | {f(s['requests_per_s'], 2)} |",
        f"| Frame latency p50 / p90 / p99 / max, s | {f(s['frame_latency_s']['p50'])} / "
        f"{f(s['frame_latency_s']['p90'])} / {f(s['frame_latency_s']['p99'])} / {f(s['frame_latency_s']['max'])} |",
        f"| Request latency p50 / p90 / p99, s | {f(s['request_latency_s']['p50'])} / "
        f"{f(s['request_latency_s']['p90'])} / {f(s['request_latency_s']['p99'])} |",
        f"| Tokens per second, prompt / generated | {f(s['prompt_tokens_per_s'], 0)} / "
        f"{f(s['completion_tokens_per_s'], 0)} |",
    ]
    if s.get("paced"):
        p = s["paced"]
        lines += [
            f"| Frames longer than the {s['interval']:g} s slot | {p['frames_over_interval']} |",
            f"| Frames started late (>0.5 s) | {p['frames_started_late']}, "
            f"late by p50 {f(p['late_by_s']['p50'])} s, max {f(p['late_by_s']['max'])} s |",
            f"| **Keeps up** | **{'yes' if p['keeps_up'] else 'no'}** |",
        ]
    return "\n".join(lines) + "\n"
