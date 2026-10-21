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

With the boxes you labelled with cow ids on frames of the same recordings (herd.py boxes: frames.json and
the labels.json the page, or the ids gallery, saved) the second pass is scored against them - IDF1, the
share of your boxes that got the right cow (the system's names matched one to one to yours), and IDP /
IDR beside it; per camera too, and which of its cows took two of yours (merged) or the other way (split):

    python herd/herd.py trial ... --boxes /srv/herd/boxes --labels /srv/herd/labels.json
    python herd/herd.py trial --score-only --config barn.toml --out /srv/herd/trial --boxes ... --labels ...
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import iou, load_config, recordings, sessions  # noqa: E402


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


def idf1(co, n_true, n_pred):
    """co Counter((your id, its cow)) over the boxes matched -> IDF1, IDP, IDR, the one-to-one mapping."""
    from scipy.optimize import linear_sum_assignment
    if not co:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0, "mapping": {}}
    ts = sorted({t for t, _ in co})
    ps = sorted({p for _, p in co})
    M = np.zeros((len(ts), len(ps)))
    for (t, p), n in co.items():
        M[ts.index(t), ps.index(p)] = n
    r, c = linear_sum_assignment(-M)
    tp = float(M[r, c].sum())
    return {"idf1": 2 * tp / max(1, n_true + n_pred), "idp": tp / max(1, n_pred), "idr": tp / max(1, n_true),
            "mapping": {ts[i]: ps[j] for i, j in zip(r, c) if M[i, j] > 0}}


def score(store_path, cfg, boxes_dir, labels_path, max_dt=0.6, min_iou=0.3, page_dir=None, cameras=None):
    """Your labelled boxes (status done, with an id) against the second pass at the same camera and second.

    Only boxes this camera counts (its mask, zone) are scored - one in another camera's part of an overlap
    is that camera's. Your box and its box matched by overlap (min_iou, one to one); the system's cow of a
    box is its track's (report.track_map). A box of yours it did not find, or found without a cow, counts
    against IDR; a cow it gave where you labelled none counts against IDP."""
    import report
    from box_dataset import read_labels
    from common import in_zone
    from scipy.optimize import linear_sum_assignment
    from store import Store
    store = Store(store_path)
    tm = report.track_map(store)
    frames = {f["file"]: f for f in json.load(open(os.path.join(boxes_dir, "frames.json"), encoding="utf-8"))["frames"]}
    cams = {c["id"]: c for c in cfg["cameras"] if not cameras or c["id"] in cameras}
    labels = read_labels(labels_path)
    shift = clock_shift(store, [frames[k] for k, v in labels.items() if v.get("status") == "done" and k in frames
                                and frames[k]["cam"] in cams])
    co, cam_co = collections.Counter(), collections.defaultdict(collections.Counter)
    n = collections.Counter()                      # (cam, what) -> boxes
    for file, lab in labels.items():
        f = frames.get(file)
        if lab.get("status") != "done" or not f or f["cam"] not in cams:
            continue
        f = dict(f, ts=f["ts"] + shift)
        cam = f["cam"]
        truth = [(b[:4], str(b[4]).strip()) for b in lab.get("boxes", []) if len(b) >= 5 and str(b[4]).strip()
                 and str(b[4]).strip().lower() != "x"]
        truth = [t for t in truth if in_zone(t[0], cams[cam], f.get("w", 1920), f.get("h", 1080))]
        rows = store.query("SELECT track, ts, x1, y1, x2, y2 FROM seconds WHERE cam = ? AND ts BETWEEN ? AND ?",
                           (cam, f["ts"] - max_dt, f["ts"] + max_dt))
        if not rows:
            n[(cam, "frames_not_played")] += 1
            continue
        n[(cam, "frames")] += 1
        sec = min({r[1] for r in rows}, key=lambda t: abs(t - f["ts"]))
        pred = [(list(r[2:6]), tm.get(r[0])) for r in rows if r[1] == sec]
        n[(cam, "your_boxes")] += len(truth)
        n[(cam, "its_named_boxes")] += sum(1 for _, cow in pred if cow)
        if truth and pred:
            M = np.array([[iou(t[0], q[0]) for q in pred] for t in truth])
            r, c = linear_sum_assignment(-M)
            hit = [(i, j) for i, j in zip(r, c) if M[i, j] >= min_iou]
        else:
            hit = []
        n[(cam, "found")] += len(hit)
        n[(cam, "found_with_a_cow")] += sum(1 for _, j in hit if pred[j][1])
        for i, j in hit:
            if pred[j][1]:
                co[(truth[i][1], pred[j][1])] += 1
                cam_co[cam][(truth[i][1], pred[j][1])] += 1
    allc = sorted({c for c, _ in n})
    tot = lambda what: sum(n[(c, what)] for c in allc)
    out = {"all": idf1(co, tot("your_boxes"), tot("its_named_boxes")), "cameras": {}}
    out["all"].update({k: tot(k) for k in ("frames", "frames_not_played", "your_boxes", "found", "found_with_a_cow",
                                           "its_named_boxes")})
    for c in allc:
        r = idf1(cam_co[c], n[(c, "your_boxes")], n[(c, "its_named_boxes")])
        r.update({k: n[(c, k)] for k in ("frames", "frames_not_played", "your_boxes", "found", "found_with_a_cow",
                                         "its_named_boxes")})
        r.pop("mapping")
        out["cameras"][c] = r
    by_pred, by_true = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for (t, p), k in co.items():
        by_pred[p][t] += k
        by_true[t][p] += k
    big = lambda cnt: [x for x, k in cnt.items() if k >= max(3, 0.2 * sum(cnt.values()))]
    out["merged"] = {p: dict(cnt.most_common()) for p, cnt in by_pred.items() if len(big(cnt)) > 1}
    out["split"] = {t: dict(cnt.most_common()) for t, cnt in by_true.items() if len(big(cnt)) > 1}
    out["your_cows"], out["its_cows"] = len(by_true), len(by_pred)
    out["clock_shift_s"] = shift
    if page_dir and out["merged"]:
        out["same_cow_page"] = same_cow_page(page_dir, boxes_dir, labels, frames, out["merged"])
    return out


def same_cow_page(out, boxes_dir, labels, frames, merged, per_cam=2):
    """OUT/same_cow.html: for each of its cows that took several of your ids, your ids side by side (crops
    from your frames, each camera) - one click says they are one cow (or not); Save gives labels.json with
    those ids made one (the cow a second camera sees under another of your names)."""
    from PIL import Image
    folder = os.path.join(out, "same_cow")
    os.makedirs(folder, exist_ok=True)
    want = {i for ids in merged.values() for i in ids}
    seen = collections.defaultdict(list)                       # your id -> [(cam, file, box)]
    for file, lab in labels.items():
        f = frames.get(file)
        if lab.get("status") != "done" or not f:
            continue
        for b in lab.get("boxes", []):
            if len(b) >= 5 and str(b[4]).strip() in want:
                seen[str(b[4]).strip()].append((f["cam"], file, b[:4], f["ts"]))
    crops = {}
    for cid, items in seen.items():
        out_c = []
        for cam in sorted({c for c, _, _, _ in items}):
            mine = sorted((x for x in items if x[0] == cam), key=lambda x: x[3])
            for k in sorted({int(round(i)) for i in np.linspace(0, len(mine) - 1, min(per_cam, len(mine)))}):
                _, file, b, _ = mine[k]
                with Image.open(os.path.join(boxes_dir, file)) as im:
                    w, h = im.size
                    pad = 0.08
                    c = im.convert("RGB").crop((int(max(0, (b[0] - pad * (b[2] - b[0])) * w)),
                                                int(max(0, (b[1] - pad * (b[3] - b[1])) * h)),
                                                int(min(w, (b[2] + pad * (b[2] - b[0])) * w)),
                                                int(min(h, (b[3] + pad * (b[3] - b[1])) * h))))
                    c.thumbnail((200, 200))
                    name = f"{len(crops) * 10 + len(out_c)}_{cam}.jpg"
                    c.save(os.path.join(folder, name), quality=88)
                out_c.append({"src": f"same_cow/{name}", "cam": cam})
        crops[cid] = {"crops": out_c, "cams": sorted({c for c, _, _, _ in items}), "boxes": len(items)}
    cards = [{"cow": cow, "ids": [{"id": i, "n": n, **crops.get(i, {"crops": [], "cams": [], "boxes": 0})}
                                  for i, n in ids.items()]}
             for cow, ids in sorted(merged.items(), key=lambda kv: -sum(kv[1].values()))]
    data = {"cards": cards, "labels": {"frames": labels}}
    path = os.path.join(out, "same_cow.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(SAME_COW.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    return path


SAME_COW = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>One cow, two names?</title>
<style>
:root { --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --line: #d9d9d4; --card: #fff; --accent: #c2410c;
  --yes: #15803d; --no: #b91c1c; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecece8; --muted: #9a9a94; --line: #34342f;
  --card: #1f1f1d; --accent: #fb923c; --yes: #4ade80; --no: #f87171; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line);
  padding: 10px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
h1 { font-size: 17px; margin: 0 8px 0 0; }
button { font: inherit; padding: 5px 12px; border: 1px solid var(--line); border-radius: 6px; background: var(--card);
  color: var(--fg); cursor: pointer; }
button.main { background: var(--accent); border-color: var(--accent); color: #fff; }
button.on.yes { background: var(--yes); border-color: var(--yes); color: #fff; }
button.on.no { background: var(--no); border-color: var(--no); color: #fff; }
.help { padding: 8px 16px; color: var(--muted); max-width: 1000px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 12px 16px; padding: 10px 12px; }
.card h2 { font-size: 15px; margin: 0 0 8px; color: var(--muted); font-weight: normal; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 6px 0; border-top: 1px solid var(--line); }
.row:first-of-type { border-top: 0; }
.name { min-width: 120px; }
.name b { font-size: 18px; }
.name small { display: block; color: var(--muted); }
.crops { display: flex; gap: 6px; flex-wrap: wrap; }
.crops figure { margin: 0; text-align: center; font-size: 12px; color: var(--muted); }
.crops img { height: 120px; border-radius: 4px; display: block; }
.ask { margin-left: auto; display: flex; gap: 6px; }
</style></head><body>
<header><h1>One cow, two names?</h1><button class="main" id="save">Save labels.json</button>
<span id="state" style="color: var(--muted)"></span></header>
<div class="help">The system gave one cow to boxes you named differently. Often it is one cow that two cameras
see, named once on each - then say <b>same cow</b> and the second name becomes the first. If they are two cows,
say <b>different</b>: the system was wrong there. Save gives labels.json with the names made one; score again
with it (and use it for training).</div>
<main id="cards"></main>
<script>
const DATA = __DATA__;
const KEY = "same-cow:" + DATA.cards.map(c => c.cow).join(",");
let ans = {};
try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ans = {}; }
const keep = () => { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} };
function render() {
  const main = document.getElementById("cards");
  main.innerHTML = "";
  let asked = 0, done = 0;
  DATA.cards.forEach(c => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h2>The system's ${c.cow}</h2>`;
    const first = c.ids[0].id;
    c.ids.forEach((x, k) => {
      const row = document.createElement("div");
      row.className = "row";
      const figs = x.crops.map(p => `<figure><img src="${p.src}" alt="${x.id}"><figcaption>${p.cam}</figcaption></figure>`).join("");
      row.innerHTML = `<div class="name"><b>${x.id}</b><small>${x.n} boxes with it here; ${(x.cams || []).join(", ")}</small></div>
        <div class="crops">${figs}</div>`;
      if (k > 0) {
        asked++;
        const key = c.cow + "|" + x.id, a = ans[key];
        if (a) done++;
        const ask = document.createElement("div");
        ask.className = "ask";
        ask.innerHTML = `<button class="yes ${a === "same" ? "on" : ""}">same cow as ${first}</button>
          <button class="no ${a === "diff" ? "on" : ""}">different</button>`;
        ask.querySelector(".yes").onclick = () => { ans[key] = "same"; keep(); render(); };
        ask.querySelector(".no").onclick = () => { ans[key] = "diff"; keep(); render(); };
        row.appendChild(ask);
      }
      card.appendChild(row);
    });
    main.appendChild(card);
  });
  document.getElementById("state").textContent = done + " of " + asked + " answered";
}
function renames() {
  const parent = {};
  const root = x => { while (parent[x] && parent[x] !== x) x = parent[x]; return x; };
  DATA.cards.forEach(c => c.ids.slice(1).forEach(x => {
    if (ans[c.cow + "|" + x.id] === "same") {
      const a = root(c.ids[0].id), b = root(x.id);
      if (a !== b) parent[b] = a;
    }
  }));
  const out = {};
  Object.keys(parent).forEach(k => { const r = root(k); if (r !== k) out[k] = r; });
  return out;
}
document.getElementById("save").onclick = () => {
  const ren = renames(), lab = JSON.parse(JSON.stringify(DATA.labels));
  Object.values(lab.frames).forEach(f => (f.boxes || []).forEach(b => {
    if (b.length >= 5 && ren[String(b[4]).trim()] !== undefined) b[4] = ren[String(b[4]).trim()];
  }));
  const blob = new Blob([JSON.stringify(lab)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "labels.json"; a.click();
};
render();
</script></body></html>
"""


def clock_shift(store, frames):
    """Whole hours to add to the frames' clock to land in this run: frames cut with another time zone
    (timezone_offset_hours changed since) are as many hours off. 0 when they land as they are."""
    span = {cam: (a, b) for cam, a, b in store.query("SELECT cam, MIN(ts), MAX(ts) FROM seconds GROUP BY cam")}
    hits = lambda off: sum(1 for f in frames if f["cam"] in span and span[f["cam"]][0] - 1 <= f["ts"] + off
                           <= span[f["cam"]][1] + 1)
    best = max(range(-14, 15), key=lambda h: (hits(h * 3600), -abs(h)))
    if hits(best * 3600) <= hits(0):
        if frames and not hits(0):
            fr = sorted(f["ts"] for f in frames)
            print(f"[score] none of your {len(frames)} frames is in this run: they are from "
                  f"{dt.datetime.utcfromtimestamp(fr[0]):%Y-%m-%d %H:%M} to {dt.datetime.utcfromtimestamp(fr[-1]):%H:%M} "
                  f"UTC, the run from {dt.datetime.utcfromtimestamp(min(a for a, _ in span.values())):%Y-%m-%d %H:%M}"
                  f" to {dt.datetime.utcfromtimestamp(max(b for _, b in span.values())):%H:%M} UTC", flush=True)
        return 0.0
    print(f"[score] your frames' clock is {best:+d} h off this run's: cut with another timezone_offset_hours than "
          f"the config has now - taken into account ({hits(best * 3600)} of {len(frames)} frames in the run)", flush=True)
    return best * 3600.0


def print_score(sc):
    a = sc["all"]
    if sc.get("clock_shift_s"):
        print(f"[score] (your frames' clock moved {sc['clock_shift_s'] / 3600:+.0f} h to this run's)")
    print(f"\n[score] against your labels: {a['your_boxes']} boxes of {sc['your_cows']} cows on {a['frames']} frames"
          + (f" ({a['frames_not_played']} frames not in this run)" if a["frames_not_played"] else ""))
    print(f"  found {a['found'] / max(1, a['your_boxes']):.0%} of your boxes, {a['found_with_a_cow'] / max(1, a['your_boxes']):.0%}"
          f" with a cow; IDF1 {a['idf1']:.1%} (IDP {a['idp']:.1%}: of its cows given, the right one; IDR "
          f"{a['idr']:.1%}: of your boxes, the right cow)")
    for c, r in sorted(sc["cameras"].items()):
        print(f"  {c}: IDF1 {r['idf1']:.1%} (IDP {r['idp']:.1%}, IDR {r['idr']:.1%}) on {r['your_boxes']} boxes")
    print(f"  its cows: {sc['its_cows']} for your {sc['your_cows']}; one of its cows on several of yours (merged): "
          f"{len(sc['merged'])}; one of yours under several of its (split): {len(sc['split'])}")
    if sc.get("same_cow_page"):
        print(f"  one cow two names? {sc['same_cow_page']} - your names side by side: say which are one cow, save "
              "labels.json, score again with it")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--videos", default=None, help="recordings named <camera>..<date and time>")
    p.add_argument("--out", required=True)
    p.add_argument("--burst-every", type=float, default=20.0, help="the first pass: a burst every this many s")
    p.add_argument("--min-bursts", type=int, default=6, help="bursts a cow needs in the first pass to get an id")
    p.add_argument("--show-minutes", type=float, default=10.0, help="minutes of every camera drawn (show-ids)")
    p.add_argument("--gap", type=float, default=300.0)
    p.add_argument("--boxes", default=None, help="your labelled frames (herd.py boxes: the folder with frames.json)")
    p.add_argument("--labels", default=None, help="their labels.json with cow ids: the second pass scored (IDF1)")
    p.add_argument("--score-only", action="store_true", help="score the trial already in --out, play nothing")
    p.add_argument("--cameras", nargs="*", default=None, help="score these cameras only (one id across them)")
    args = p.parse_args(argv)
    if args.score_only:
        if not (args.boxes and args.labels):
            p.error("--score-only needs --boxes and --labels")
        cfg = load_config(args.config)
        out = os.path.abspath(args.out.rstrip("\\"))
        sc = score(os.path.join(out, "pass2", "herd.sqlite"), cfg, args.boxes, args.labels, page_dir=out,
                   cameras=args.cameras)
        name = "score.json" if not args.cameras else f"score_{'_'.join(args.cameras)}.json"
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(sc, fh, indent=2)
        print_score(sc)
        print(f"-> {os.path.join(out, name)}")
        return
    if not args.videos:
        p.error("--videos is needed")
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
    if args.boxes and args.labels:
        sc = score(store2, cfg, args.boxes, args.labels, page_dir=out, cameras=args.cameras)
        with open(os.path.join(out, "score.json"), "w", encoding="utf-8") as fh:
            json.dump(sc, fh, indent=2)
        print_score(sc)
    print(f"-> {os.path.join(out, 'trial.json')}; the videos: " + ", ".join(shown.values()))


if __name__ == "__main__":
    sys.exit(main())
