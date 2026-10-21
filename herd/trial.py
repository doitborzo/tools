#!/usr/bin/env python3
"""A dress rehearsal on recordings: the barn's whole system with its own config - zones, detector, tracker,
posture and activity, the gallery - on the files, in a store of its own, and what it made drawn back.

A real barn's gallery fills over its first hours (a cow needs new_cow_min_bursts bursts before she gets an
id, and new cows are made at night); a quarter of an hour of recordings would end with every cow NaN.
So the recordings are played twice:

  1. the first pass - a burst every --burst-every s - and then the night's maintenance at its end: the cows
     seen often enough (--min-bursts) get ids (new-<day>-NN)
  2. the second pass with that gallery, as the next day would: who is who
  3. show-ids for every camera (OUT/show_<cam>/index.html, ids.mp4) and OUT/trial.json: per camera the
     seconds counted, the tracks, the share of boxes given a cow, posture and activity; the cows made

    python herd/herd.py trial --config barn.toml --videos /srv/herd/rec --out /srv/herd/trial
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import itertools
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_config, recordings, sessions  # noqa: E402


def play(cfg, groups, store_path, models):
    import pipeline
    for k, g in enumerate(groups, 1):
        c = copy.deepcopy(cfg)
        cams = {cam["id"]: cam for cam in c["cameras"]}
        c["cameras"] = [dict(cams[cam], url=path, start_ts=ts) for cam, path, ts in g]
        c["store"]["path"] = store_path
        start = min(ts for _, _, ts in g)
        print(f"[trial]   session {k}/{len(groups)}: {', '.join(cam for cam, _, _ in g)}", flush=True)
        models = models or pipeline.Models(c)
        status = {}
        pipeline.run(c, start, models=models, status=status)
        if status.get("interrupted"):
            sys.exit("[trial] stopped (Ctrl+C)")
    return models


def summary(store_path, cfg, gallery_folder):
    import report
    from store import Store
    store = Store(store_path)
    tm = report.track_map(store)
    out = {}
    for cam, n, tracks in store.query("SELECT cam, COUNT(*), COUNT(DISTINCT track) FROM seconds GROUP BY cam"):
        rows = store.query("SELECT track, posture, activity FROM seconds WHERE cam = ?", (cam,))
        named = sum(1 for t, _, _ in rows if tm.get(t))
        states = dict(store.query("SELECT state, COUNT(*) FROM bursts WHERE cam = ? GROUP BY state", (cam,)))
        out[cam] = {"cow_seconds": n, "tracks": tracks, "with_a_cow": named / max(1, n),
                    "cows": len({tm[t] for t, _, _ in rows if tm.get(t)}),
                    "posture": dict(collections.Counter(p for _, p, _ in rows)),
                    "activity": dict(collections.Counter(a for _, _, a in rows)), "bursts": states}
    meta = os.path.join(gallery_folder, "gallery.json")
    cows = len(json.load(open(meta)).get("cows", {})) if os.path.exists(meta) else 0
    return out, cows, impossible(store, tm, cfg)


def impossible(store, tm, cfg):
    """What cannot be: one cow at the same second on two tracks of one camera, or on two cameras that share
    no floor ([mct] disjoint) - two cows given one id. -> counts, per cow and over all, of named cow-seconds."""
    disjoint = {frozenset(p) for p in cfg["mct"].get("disjoint", [])}
    at = collections.defaultdict(lambda: collections.defaultdict(set))       # second -> cow -> {(cam, track)}
    named = 0
    for cam, track, ts in store.query("SELECT cam, track, ts FROM seconds"):
        cow = tm.get(track)
        if cow:
            at[int(ts)][cow].add((cam, track))
            named += 1
    twice, apart, per_cow = 0, 0, collections.Counter()
    for cows in at.values():
        for cow, seen in cows.items():
            cams = collections.Counter(c for c, _ in seen)
            bad = sum(n - 1 for n in cams.values())
            far = sum(1 for a, b in itertools.combinations(sorted(cams), 2) if frozenset((a, b)) in disjoint)
            twice, apart = twice + bad, apart + far
            if bad or far:
                per_cow[cow] += 1
    return {"named_cow_seconds": named, "one_cow_twice_on_one_camera": twice,
            "one_cow_on_cameras_apart": apart, "cows_seen_where_they_cannot_be": dict(per_cow.most_common())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--videos", required=True, help="recordings named <camera>..<date and time>")
    p.add_argument("--out", required=True)
    p.add_argument("--burst-every", type=float, default=20.0, help="the first pass: a burst every this many s")
    p.add_argument("--min-bursts", type=int, default=6, help="bursts a cow needs in the first pass to get an id")
    p.add_argument("--show-minutes", type=float, default=10.0, help="minutes of every camera drawn (show-ids)")
    p.add_argument("--gap", type=float, default=300.0)
    args = p.parse_args(argv)
    import gallery as gallery_mod
    import show_ids
    from store import Store
    cfg = load_config(args.config)
    tz = dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))
    recs, bad = recordings(args.videos, [c["id"] for c in cfg["cameras"]], cfg["farm"]["timezone_offset_hours"])
    if not recs:
        sys.exit(f"no recordings under {args.videos} named <camera>..<YYYY-MM-DD_HH-MM-SS>")
    groups = sessions(recs, args.gap)
    out = os.path.abspath(args.out.rstrip("\\"))
    for d in ("pass1", "pass2"):
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
        os.makedirs(os.path.join(out, d))
    cfg["training_cache"]["enabled"] = False
    cfg["identity"]["new_cow_min_bursts"] = args.min_bursts
    first = copy.deepcopy(cfg)
    first["sampling"]["burst_every_s"] = args.burst_every
    print(f"[trial] 1/3 the first pass ({len(recs)} recordings, a burst every {args.burst_every:g} s): the gallery "
          f"collects the cows", flush=True)
    models = play(first, groups, os.path.join(out, "pass1", "herd.sqlite"), None)
    # the night after it: the cows seen often enough get ids
    abstain = getattr(models, "id_abstain", None)
    folder = os.path.join(out, "pass1", "gallery")
    from reid import apply_thresholds
    from common import DEFAULTS
    apply_thresholds(first, getattr(models, "id_thresholds", {}) or {}, DEFAULTS, None)
    g = gallery_mod.Gallery(folder, first, abstain)
    last = max(ts for _, _, ts in recs) + 3600
    res = g.nightly(last, Store(os.path.join(out, "pass1", "herd.sqlite")).cannot_link_pairs())
    print(f"[trial] the night: {len(res['enrolled'])} cows enrolled ({res['pool']} bursts left unknown"
          + (f"; {res['held_back_herd_full']} more held back: max_cows reached" if res.get("held_back_herd_full") else "")
          + ")", flush=True)
    shutil.copytree(folder, os.path.join(out, "pass2", "gallery"))
    print("[trial] 2/3 the second pass, with those cows: who is who", flush=True)
    play(cfg, groups, os.path.join(out, "pass2", "herd.sqlite"), models)
    print("[trial] 3/3 drawn back", flush=True)
    store2 = os.path.join(out, "pass2", "herd.sqlite")
    shown = {}
    for cam, path, ts in recs:
        if cam in shown:
            continue
        dest = os.path.join(out, f"show_{cam}")
        try:
            show_ids.main(["--config", args.config, "--store", store2, "--cam", cam, "--video", path,
                           "--start", dt.datetime.fromtimestamp(ts, tz).isoformat(), "--minutes", str(args.show_minutes),
                           "--out", dest])
            shown[cam] = os.path.join(dest, "index.html")
        except SystemExit as e:
            print(f"[trial] {cam}: nothing drawn ({e})", flush=True)
    per_cam, cows, odd = summary(store2, cfg, os.path.join(out, "pass2", "gallery"))
    report = {"cows_enrolled": res["enrolled"], "cows_in_gallery": cows, "cameras": per_cam, "impossible": odd,
              "held_back_herd_full": res.get("held_back_herd_full", 0), "show": shown}
    with open(os.path.join(out, "trial.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[trial] {len(res['enrolled'])} cows made from the first pass")
    for cam, r in sorted(per_cam.items()):
        b = r["bursts"]
        print(f"  {cam}: {r['cow_seconds']} cow-seconds on {r['tracks']} tracks; {r['with_a_cow']:.0%} of them given a "
              f"cow ({r['cows']} cows); bursts confirmed {b.get('confirmed', 0)}, tentative {b.get('tentative', 0)}, "
              f"unknown {b.get('unknown', 0)}")
    n = max(1, odd["named_cow_seconds"])
    print(f"  what cannot be: one cow twice on one camera at once {odd['one_cow_twice_on_one_camera']} s, on two "
          f"cameras that share no floor at once {odd['one_cow_on_cameras_apart']} s "
          f"({(odd['one_cow_twice_on_one_camera'] + odd['one_cow_on_cameras_apart']) / n:.1%} of the named "
          f"cow-seconds) - two cows given one id; {len(odd['cows_seen_where_they_cannot_be'])} cows so")
    if res.get("held_back_herd_full"):
        print(f"  !! [identity] max_cows = {cfg['identity'].get('max_cows')}: the gallery was full, "
              f"{res['held_back_herd_full']} more cows got no id - set it to the cows these cameras see (0: no "
              "limit)")
    print(f"-> {os.path.join(out, 'trial.json')}; the videos: " + ", ".join(shown.values()))


if __name__ == "__main__":
    sys.exit(main())
