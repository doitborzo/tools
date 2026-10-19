#!/usr/bin/env python3
"""The barn's own footage as training data for identity - the part CBVD-5 cannot teach.

CBVD-5 has no cow ids and one 10 s clip per cow, so the fingerprint only
learns "the same cow 3.5 s later". What the brief wants - gait, the same cow
lying and walking, clean and dirty, Monday and Thursday - needs bursts of one
cow far apart in time. The barn makes them by itself:

  1. cache   (pipeline.py, [training_cache] enabled = true) every N-th burst's
             frame vectors (the encoder's output, 175 x 1920 numbers), its
             track, the gallery's guess and a thumbnail of the cow are kept
             under training_cache/<day>/<camera>/.
  2. sheet   the labelling page (tracks.html, opened in a browser): per day one
             row per track - thumbnails over the track, camera, time, the
             gallery's guess - and a field for her name (ear tag). One name on
             many tracks makes them one cow; x leaves a track out. "Save
             merges.csv" writes  identity,track  per line.
  3. export  -> herd_barn_feat_<date>.tgz for the identity model (reid.py, the
             pod's reid_pod.sh picks it up): the named tracks' bursts, a sixth of
             the cows held out whole. Unnamed tracks are not exported.
  4. build   (the herd model, CBVD-5 format) -> a features folder, every burst
             with an "identity"; train.py --extra-train <folder>/train trains the
             fingerprint so that all bursts of one identity are positives;
             train.py eval --extra-val <folder>/val scores re-identification
             across tracks (and days, with --val-from). Without merges.csv a
             track is a cow (the tracker keeps her for minutes to hours).

    python herd/herd.py barn sheet  --cache /srv/herd/barn/training_cache --out tracks.html \\
        [--merges merges.csv] [--names /srv/herd/barn/names.json]
    python herd/herd.py barn export --cache /srv/herd/barn/training_cache --merges merges.csv
    python herd/barn_dataset.py build --cache /workspace/herd/barn/training_cache \\
        --out /workspace/herd/features_barn --merges merges.csv --val-from 2026-11-01
    python herd/herd.py train --features /workspace/herd/features --extra-train /workspace/herd/features_barn/train ...
    python herd/herd.py eval  --features /workspace/herd/features --out RUN --extra-val /workspace/herd/features_barn/val

--gallery-labels 1 also joins tracks the gallery confirmed as one cow (2/3 of
their bursts): self-training - more data, but the gallery's mistakes are
learnt too; use it only once the NaN cut-off is strict.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_FRAMES, FPS = 175, 25.0
CHUNK = 200                       # bursts per pseudo-clip file
FIRST_CLIP = 900000               # barn pseudo-clip ids, clear of CBVD-5's


# ------------------------------------------------------------------ cache

class TrainingCache:
    """Written by the pipeline: every every_n-th burst's frame vectors + its
    decision + a thumbnail, until max_gb is used. Thread-safe."""

    def __init__(self, cfg, features_meta):
        c = cfg["training_cache"]
        self.folder = c["folder"] if os.path.isabs(c["folder"]) else os.path.join(
            os.path.dirname(os.path.abspath(cfg["store"]["path"])), c["folder"])
        self.every_n, self.max_bytes = max(1, int(c["every_n"])), float(c["max_gb"]) * 2 ** 30
        self.tz = dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))
        self.lock = threading.Lock()
        self.n = 0
        os.makedirs(self.folder, exist_ok=True)
        with open(os.path.join(self.folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(features_meta, fh, indent=2)
        self.used = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(self.folder) for f in fs)
        self.full = False

    def add(self, cam, track, ts, feats, times, valid, motion, pos, thumb, row):
        with self.lock:
            self.n += 1
            if self.n % self.every_n or self.full:
                return
            if self.used >= self.max_bytes:
                self.full = True
                print(f"[cache] {self.folder} reached {self.max_bytes / 2 ** 30:.0f} GB - no more bursts kept",
                      flush=True)
                return
            day = dt.datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d")
            folder = os.path.join(self.folder, day, cam)
            os.makedirs(folder, exist_ok=True)
            name = f"{ts:.0f}_{track.replace('/', '_')}"
            path = os.path.join(folder, name + ".npz")
            np.savez(path, feats=np.asarray(feats, np.float16), times=np.asarray(times, np.float32),
                     valid=np.asarray(valid, bool), pos=np.asarray(pos, np.float32),
                     motion=np.asarray(motion if motion is not None else [], np.float32))
            from PIL import Image
            Image.fromarray(thumb).resize((112, 112)).save(os.path.join(folder, name + ".jpg"), quality=80)
            rec = {"file": os.path.relpath(path, self.folder), "thumb": os.path.relpath(path[:-4] + ".jpg", self.folder),
                   "day": day, "cam": cam, "track": track, "ts": ts, **row}
            with open(os.path.join(self.folder, day, "index.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            self.used += os.path.getsize(path) + 6000


def read_cache(folder):
    rows = []
    for day in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        p = os.path.join(folder, day, "index.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh if line.strip()]
    return rows


# --------------------------------------------------------------- identities

DROP = {"x", "-", "skip", "bad"}  # merges.csv names of tracks left out: not one cow, or no cow at all


def identities(rows, merges=None, gallery_labels=False):
    """track -> identity: a hand-made merge first, then (optionally) the
    gallery's confirmed majority, else the track itself; None for a track
    marked x (two cows swapped in it, or no cow) - it is left out."""
    ident = {r["track"]: f"track:{r['track']}" for r in rows}
    if gallery_labels:
        votes = collections.defaultdict(collections.Counter)
        n = collections.Counter()
        for r in rows:
            n[r["track"]] += 1
            if r.get("state") == "confirmed" and r.get("cow"):
                votes[r["track"]][r["cow"]] += 1
        for track, v in votes.items():
            cow, k = v.most_common(1)[0]
            if k >= 2 / 3 * n[track]:
                ident[track] = f"gallery:{cow}"
    for track, name in (merges or {}).items():
        if track in ident:
            ident[track] = None if name.lower() in DROP else f"label:{name}"
    return ident


def read_merges(path):
    """merges.csv: identity,track per line (a header line is skipped)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].strip() and row[0].strip().lower() != "identity":
                out[row[1].strip()] = row[0].strip()
    return out


def resample(times, n=N_FRAMES, fps=FPS):
    """Frame indices of a burst at the camera's own rate -> n frames on a 25 fps grid."""
    times = np.asarray(times, float)
    grid = np.arange(n) / fps
    return np.clip(np.searchsorted(times, grid), 0, len(times) - 1)


def write_split(rows, ident, cache, out_dir):
    meta = json.load(open(os.path.join(cache, "meta.json"), encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for c, start in enumerate(range(0, len(rows), CHUNK)):
        clip = str(FIRST_CLIP + c)
        chunk = rows[start:start + CHUNK]
        feats, times, motion, ok = [], [], [], []
        for i, r in enumerate(chunk):
            z = np.load(os.path.join(cache, r["file"]))
            idx = resample(z["times"])
            feats.append(z["feats"][idx])
            times.append((np.arange(N_FRAMES) / FPS).astype(np.float32))
            m = z["motion"]
            if len(m):
                motion.append(m)
                ok.append(True)
            else:
                motion.append(None)
                ok.append(False)
            index.append({"kind": "burst", "clip": clip, "i": i, "track": r["track"], "identity": ident[r["track"]],
                          "posture": -1, "activity": -1, "rumination": -1, "n_keyframes": 0,
                          "area": float(r.get("area", 0.05)), "start": 0.0, "uids": [],
                          "pos": [float(v) for v in z["pos"]], "day": r["day"], "cam": r["cam"]})
        np.savez(os.path.join(out_dir, f"{clip}.npz"), burst_feats=np.stack(feats), burst_times=np.stack(times),
                 key_feats=np.zeros((0, meta["dim"]), np.float16))
        dims = [len(m) for m in motion if m is not None]
        if dims:
            D = dims[0]
            np.savez(os.path.join(out_dir, f"{clip}.motion.npz"),
                     motion=np.stack([m if m is not None else np.zeros(D, np.float32) for m in motion]),
                     ok=np.array(ok, bool))
        index.append({"kind": "clip_done", "clip": clip})
    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as fh:
        for r in index:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({**meta, "burst_seconds": N_FRAMES / FPS, "fps": FPS, "source": "barn"}, fh, indent=2)


def cmd_build(args):
    rows = read_cache(args.cache)
    if not rows:
        sys.exit(f"no cached bursts under {args.cache} - is [training_cache] enabled in the barn config?")
    ident = identities(rows, read_merges(args.merges) if args.merges else None, bool(args.gallery_labels))
    rows = [r for r in rows if ident[r["track"]]]
    count = collections.Counter(ident[r["track"]] for r in rows)
    rows = [r for r in rows if count[ident[r["track"]]] >= args.min_bursts]
    parts = {"train": [r for r in rows if not args.val_from or r["day"] < args.val_from],
             "val": [r for r in rows if args.val_from and r["day"] >= args.val_from]}
    for name, part in parts.items():
        if part:
            write_split(part, ident, args.cache, os.path.join(args.out, name))
            ids = collections.Counter(ident[r["track"]] for r in part)
            print(f"[barn] {name}: {len(part)} bursts, {len(ids)} identities "
                  f"({sum(1 for k in ids if k.startswith('label:'))} labelled by hand, "
                  f"{sum(1 for k in ids if k.startswith('gallery:'))} from the gallery), "
                  f"days {min(r['day'] for r in part)} - {max(r['day'] for r in part)} -> {os.path.join(args.out, name)}")


def track_table(rows, cache, page_dir, names=None, thumbs=12):
    """The sheet's rows: per (day, track) its camera, time, bursts, the gallery's guess (with the ear tag
    names.json gives it) and up to `thumbs` thumbnails spread over the track."""
    by_track = collections.defaultdict(list)
    for r in rows:
        by_track[(r["day"], r["track"])].append(r)
    out = []
    for (day, track), rs in sorted(by_track.items(), key=lambda kv: (kv[0][0], kv[1][0]["ts"], kv[0][1])):
        rs.sort(key=lambda r: r["ts"])
        guess = collections.Counter(r.get("cow") for r in rs if r.get("state") == "confirmed" and r.get("cow"))
        g = guess.most_common(1)[0][0] if guess else ""
        pick = sorted({int(round(i)) for i in np.linspace(0, len(rs) - 1, min(thumbs, len(rs)))})
        out.append({"day": day, "track": track, "cam": "/".join(sorted({r["cam"] for r in rs})), "n": len(rs),
                    "t0": dt.datetime.fromtimestamp(rs[0]["ts"]).strftime("%H:%M"),
                    "t1": dt.datetime.fromtimestamp(rs[-1]["ts"]).strftime("%H:%M"),
                    "guess": g, "guess_name": (names or {}).get(g, ""),
                    "thumbs": [os.path.relpath(os.path.join(cache, rs[i]["thumb"]), page_dir).replace(os.sep, "/")
                               for i in pick]})
    return out


def cmd_sheet(args):
    """The labelling page: per day, one row per track - thumbnails, camera, time, the gallery's guess - and
    a field for the cow's name (her ear tag). The names stay in the browser while you work; "Save
    merges.csv" writes them out (identity,track per line, x = leave the track out) for barn export."""
    rows = read_cache(args.cache)
    if args.day:
        rows = [r for r in rows if r["day"] == args.day]
    names = json.load(open(args.names, encoding="utf-8")) if args.names else {}
    page_dir = os.path.dirname(os.path.abspath(args.out))
    tracks = track_table(rows, args.cache, page_dir, names, args.thumbs)
    labels = read_merges(args.merges) if args.merges else {}
    data = {"cache": os.path.abspath(args.cache), "tracks": tracks, "labels": labels}
    html = SHEET.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[barn] {len(tracks)} tracks on {len({t['day'] for t in tracks})} days"
          + (f", {sum(1 for t in tracks if t['track'] in labels)} named in {args.merges}" if labels else "")
          + f" -> {args.out}  (open it in a browser; the thumbnails are read from {args.cache})")


SHEET = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Barn track labels</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1da;--accent:#2f6f4f;--warn:#a33;--done:#e8f3ec;--x:#f6e7e7}
@media (prefers-color-scheme: dark){:root{--bg:#171816;--card:#20221f;--fg:#ecebe6;--mute:#9a9a93;--line:#34362f;--accent:#7fc29c;--warn:#e88;--done:#1f3127;--x:#3a2424}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:17px;margin:0 0 6px}.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button,select,input{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.main{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.stat{color:var(--mute)}main{padding:8px 16px 40px}
h2{font-size:15px;margin:18px 0 6px}
.row{display:grid;grid-template-columns:230px 1fr;gap:10px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;margin:6px 0}
.row.named{background:var(--done)}.row.dropped{background:var(--x)}
.info b{word-break:break-all}.info .mute{color:var(--mute)}.info input{width:100%;margin:6px 0 4px}
.btns{display:flex;flex-wrap:wrap;gap:4px}.btns button{padding:2px 6px;font-size:12px}
.thumbs{display:flex;flex-wrap:wrap;gap:3px;align-content:flex-start}.thumbs img{width:84px;height:84px;object-fit:cover;border-radius:4px;background:var(--line)}
.cows{margin:6px 0 0;color:var(--mute);font-size:12px}.cows span{display:inline-block;margin-right:10px}
.help{color:var(--mute);font-size:13px;margin:4px 0}
@media (max-width:640px){header{position:static}.row{grid-template-columns:1fr}.thumbs img{width:64px;height:64px}}
</style></head><body>
<header>
<h1>Barn tracks: who is who</h1>
<p class="help">Type each track's cow (her ear tag) and press Enter for the next track. One name on many tracks
makes them one cow. <b>x</b> leaves a track out (two cows in it, or no cow). Names are kept in this browser;
<b>Save merges.csv</b> writes them out for <code>herd.py barn export</code>.</p>
<div class="bar">
<select id="day"></select><select id="cam"></select>
<label><input type="checkbox" id="todo"> only unnamed</label>
<input id="find" placeholder="find a name" size="12">
<button class="main" id="save">Save merges.csv</button>
<label><button id="loadbtn" type="button">Load merges.csv</button><input type="file" id="load" accept=".csv,text/csv" hidden></label>
<span class="stat" id="stat"></span>
</div>
<div class="cows" id="cows"></div>
</header>
<main id="list"></main>
<datalist id="known"></datalist>
<script>
const DATA = __DATA__;
const KEY = "herd-barn-labels:" + DATA.cache;
const PAGE = 150;
let labels = Object.assign({}, DATA.labels);
try { Object.assign(labels, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
let shown = PAGE, lastName = "";
const $ = id => document.getElementById(id);
const isDrop = n => ["x", "-", "skip", "bad"].includes((n || "").toLowerCase());
function store() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function options(sel, values, all) {
  sel.innerHTML = "";
  [all, ...values].forEach(v => { const o = document.createElement("option"); o.value = v === all ? "" : v; o.textContent = v; sel.appendChild(o); });
}
options($("day"), [...new Set(DATA.tracks.map(t => t.day))], "all days");
options($("cam"), [...new Set(DATA.tracks.flatMap(t => t.cam.split("/")))].sort(), "all cameras");
function visible() {
  const d = $("day").value, c = $("cam").value, todo = $("todo").checked, f = $("find").value.trim().toLowerCase();
  return DATA.tracks.filter(t => (!d || t.day === d) && (!c || t.cam.split("/").includes(c)) &&
    (!todo || !labels[t.track]) && (!f || (labels[t.track] || "").toLowerCase() === f));
}
function summary() {
  const cows = {};
  DATA.tracks.forEach(t => { const n = labels[t.track]; if (!n || isDrop(n)) return;
    (cows[n] = cows[n] || {tracks: 0, days: new Set(), bursts: 0}); cows[n].tracks++; cows[n].days.add(t.day); cows[n].bursts += t.n; });
  const named = DATA.tracks.filter(t => labels[t.track]).length;
  const dropped = DATA.tracks.filter(t => isDrop(labels[t.track])).length;
  $("stat").textContent = `${named} of ${DATA.tracks.length} tracks done (${dropped} left out), ${Object.keys(cows).length} cows`;
  const names = Object.keys(cows).sort();
  $("known").innerHTML = names.map(n => `<option value="${esc(n)}">`).join("");
  $("cows").innerHTML = names.map(n => `<span title="${cows[n].bursts} bursts">${esc(n)}: ${cows[n].tracks} tracks, ${cows[n].days.size} days</span>`).join("");
}
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }
function setLabel(t, name, row) {
  name = name.trim();
  if (name) labels[t.track] = name; else delete labels[t.track];
  if (name && !isDrop(name)) lastName = name;
  row.classList.toggle("named", !!name && !isDrop(name));
  row.classList.toggle("dropped", isDrop(name));
  row.querySelector("input").value = name;
  store(); summary();
}
function render() {
  const list = $("list"), ts = visible();
  list.innerHTML = "";
  let day = null;
  ts.slice(0, shown).forEach(t => {
    if (t.day !== day) { day = t.day; const h = document.createElement("h2"); h.textContent = day; list.appendChild(h); }
    const row = document.createElement("div");
    row.className = "row";
    const guess = t.guess ? `${t.guess}${t.guess_name ? " = " + t.guess_name : ""}` : "-";
    row.innerHTML = `<div class="info"><b>${esc(t.track)}</b><div class="mute">${esc(t.cam)} ${t.t0}-${t.t1}, ${t.n} bursts</div>
      <div class="mute">gallery: ${esc(guess)}</div><input list="known" placeholder="ear tag / name">
      <div class="btns"><button data-a="last" title="the last name typed">= last</button>
      ${t.guess ? `<button data-a="guess" title="the gallery's guess">= ${esc(t.guess_name || "gallery")}</button>` : ""}
      <button data-a="x" title="two cows in it, or no cow">x</button><button data-a="clear">clear</button></div></div>
      <div class="thumbs">${t.thumbs.map(s => `<img loading="lazy" src="${esc(s)}" alt="">`).join("")}</div>`;
    const inp = row.querySelector("input");
    inp.value = labels[t.track] || "";
    row.classList.toggle("named", !!inp.value && !isDrop(inp.value));
    row.classList.toggle("dropped", isDrop(inp.value));
    inp.addEventListener("change", () => setLabel(t, inp.value, row));
    inp.addEventListener("keydown", e => { if (e.key === "Enter") { setLabel(t, inp.value, row);
      const all = [...list.querySelectorAll(".row input")]; const nx = all[all.indexOf(inp) + 1]; if (nx) nx.focus(); } });
    row.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      const a = b.dataset.a;
      setLabel(t, a === "last" ? lastName : a === "guess" ? (t.guess_name || t.guess) : a === "x" ? "x" : "", row);
    }));
    list.appendChild(row);
  });
  if (ts.length > shown) {
    const more = document.createElement("button");
    more.textContent = `show ${Math.min(PAGE, ts.length - shown)} more of ${ts.length - shown}`;
    more.onclick = () => { shown += PAGE; render(); };
    list.appendChild(more);
  }
  if (!ts.length) list.innerHTML = "<p class='stat'>No tracks here.</p>";
}
["day", "cam", "todo", "find"].forEach(id => $(id).addEventListener(id === "find" ? "input" : "change", () => { shown = PAGE; render(); }));
$("save").onclick = () => {
  const lines = ["identity,track", ...DATA.tracks.filter(t => labels[t.track]).map(t => `${labels[t.track].replace(/,/g, " ")},${t.track}`)];
  const kept = new Set(DATA.tracks.map(t => t.track));
  Object.keys(labels).filter(k => !kept.has(k)).forEach(k => lines.push(`${labels[k].replace(/,/g, " ")},${k}`));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  a.download = "merges.csv"; a.click();
};
$("loadbtn").onclick = () => $("load").click();
$("load").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  f.text().then(txt => { txt.split(/\r?\n/).forEach(line => { const i = line.indexOf(",");
    if (i < 1) return; const n = line.slice(0, i).trim(), tr = line.slice(i + 1).trim();
    if (n && tr && n.toLowerCase() !== "identity") labels[tr] = n; });
    store(); summary(); render(); });
};
summary(); render();
</script></body></html>
"""


def cmd_export(args):
    """Named tracks -> the identity model's training set (reid.py's feat/barn): each burst's valid frames
    (up to --frames, spread over its 7 s), each named cow an identity across tracks, cameras and days; a
    sixth of the cows held out whole (never trained on: they score it and set its thresholds). Unnamed
    tracks are not exported - one could be a held-out cow."""
    import random
    import tarfile
    import tempfile
    rows = read_cache(args.cache)
    if not rows:
        sys.exit(f"no cached bursts under {args.cache} - is [training_cache] enabled in the barn config?")
    merges = read_merges(args.merges)
    named = {t: n for t, n in merges.items() if n.lower() not in DROP}
    rows = [r for r in rows if r["track"] in named]
    if not rows:
        sys.exit(f"none of the tracks in {args.merges} is in {args.cache}")
    count = collections.Counter(named[r["track"]] for r in rows)
    few = sorted(n for n, k in count.items() if k < args.min_bursts)
    rows = [r for r in rows if count[named[r["track"]]] >= args.min_bursts]
    cows = sorted({named[r["track"]] for r in rows})
    if len(cows) < 2:
        sys.exit(f"{len(cows)} named cow(s) with {args.min_bursts}+ bursts: name more tracks")
    k = max(1, round(len(cows) * args.val_share)) if args.val_share > 0 else 0
    val = set(random.Random(args.seed).sample(cows, min(k, len(cows) - 1)))
    meta = json.load(open(os.path.join(args.cache, "meta.json"), encoding="utf-8"))
    feats, bursts, n = [], [], 0
    for r in sorted(rows, key=lambda r: (named[r["track"]], r["ts"])):
        z = np.load(os.path.join(args.cache, r["file"]))
        f = z["feats"]
        valid = z["valid"][:len(f)] if len(z["valid"]) else np.ones(len(f), bool)
        idx = np.flatnonzero(valid)
        if len(idx) < 2:
            continue
        idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, min(args.frames, len(idx)))).astype(int))]
        feats.append(np.asarray(f[idx], np.float16))
        cow = named[r["track"]]
        bursts.append({"cow": cow, "day": r["day"], "cam": r["cam"], "track": r["track"], "ts": r["ts"],
                       "frames": list(range(n, n + len(idx))), "split": "val" if cow in val else "train",
                       "identity": f"barn:{cow}"})
        n += len(idx)
    split = collections.defaultdict(set)
    for b in bursts:
        split[b["cow"]].add(b["split"])
    assert all(len(v) == 1 for v in split.values()), "a cow both in training and held out"
    if not feats:
        sys.exit("no burst with two or more frames of its cow")
    feats = np.concatenate(feats)
    tmp = tempfile.mkdtemp()
    folder = os.path.join(tmp, "reid", "feat", "barn")
    os.makedirs(folder)
    np.savez(os.path.join(folder, "crops.npz"), feats=feats)
    keep = {k: meta.get(k) for k in ("encoder", "grid", "crop", "margin")}
    with open(os.path.join(folder, "crops.npz.json"), "w") as fh:
        json.dump(dict(keep, crops=int(len(feats))), fh)
    with open(os.path.join(folder, "bursts.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(b) + "\n" for b in bursts)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(keep, dim=int(feats.shape[1]), source="barn"), fh, indent=2)
    with open(os.path.join(folder, "merges.csv"), "w", encoding="utf-8") as fh:
        fh.write("identity,track\n" + "".join(f"{v},{t}\n" for t, v in merges.items()))
    out = args.out or f"herd_barn_feat_{dt.datetime.now():%Y%m%d_%H%M}.tgz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(os.path.join(tmp, "reid"), arcname="reid")
    import shutil
    shutil.rmtree(tmp)
    days = collections.defaultdict(set)
    for b in bursts:
        days[b["cow"]].add(b["day"])
    by = collections.Counter(b["split"] for b in bursts)
    one_day = sum(1 for c in cows if len(days[c]) < 2)
    print(f"[barn] {len(cows)} cows from {len(named)} named tracks ({len(merges) - len(named)} left out): "
          f"{by['train']} train bursts of {len(cows) - len(val)} cows, {by['val']} held-out bursts of {len(val)} "
          f"cows ({', '.join(sorted(val))}); {len(feats)} frames -> {out} ({os.path.getsize(out) / 2 ** 20:.0f} MB)")
    if few:
        print(f"[barn] fewer than {args.min_bursts} bursts, left out: {', '.join(few)}")
    if one_day:
        print(f"[barn] {one_day} of {len(cows)} cows are named on one day only: the model learns a cow across days "
              f"from the others - name the same cows on more days")


def cmd_video(args):
    """Recordings of the barn's cameras -> the training cache, as the live barn would fill it - but every
    burst kept, a burst every --burst-every s, into a store of its own (the barn's database untouched).
    The files are played in sessions (the cameras that started together); a file done is not played again."""
    import copy
    import shutil
    from common import load_config, recordings, sessions
    cfg = load_config(args.config)
    tz = cfg["farm"]["timezone_offset_hours"]
    ids = [c["id"] for c in cfg["cameras"]]
    if args.video:
        if not args.start:
            sys.exit("--video needs --start (when the recordings began, local time)")
        t0 = dt.datetime.fromisoformat(args.start)
        t0 = (t0 if t0.tzinfo else t0.replace(tzinfo=dt.timezone(dt.timedelta(hours=tz)))).timestamp()
        recs = []
        for v in args.video:
            cam, _, path = v.partition("=")
            if cam not in ids or not os.path.exists(path):
                sys.exit(f"--video {v}: want <camera id>=<file>, a camera of {args.config} ({', '.join(ids)})")
            recs.append((cam, path, t0))
        groups = [recs]
    else:
        recs, bad = recordings(args.videos, ids, tz)
        if bad:
            print(f"[video] {len(bad)} files without a camera id ({', '.join(ids)}) or a time in their name - "
                  f"skipped, e.g. {bad[0]}", flush=True)
        if not recs:
            sys.exit(f"no recordings under {args.videos} named <camera>..<YYYY-MM-DD_HH-MM-SS> (or --video cam=file "
                     f"--start ISO)")
        groups = sessions(recs, args.gap)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    done_path = os.path.join(out, "videos_done.txt")
    done = set(open(done_path, encoding="utf-8").read().split("\n")) if os.path.exists(done_path) else set()
    if args.gallery and not os.path.exists(os.path.join(out, "gallery")):
        shutil.copytree(args.gallery, os.path.join(out, "gallery"))      # its guesses help naming the cows
        print(f"[video] the gallery copied from {args.gallery}", flush=True)
    cfg["store"]["path"] = os.path.join(out, "herd.sqlite")
    cfg["training_cache"].update(enabled=True, every_n=1, folder=os.path.join(out, "training_cache"))
    cfg["sampling"]["burst_every_s"] = args.burst_every
    todo = [g for g in groups if not all(p in done for _, p, _ in g)]
    print(f"[video] {sum(len(g) for g in groups)} recordings in {len(groups)} sessions, {len(todo)} to play "
          f"(a burst every {args.burst_every:g} s a camera, every one kept) -> {out}", flush=True)
    import pipeline
    models = None
    for k, g in enumerate(todo, 1):
        c = copy.deepcopy(cfg)
        cams = {cam["id"]: cam for cam in c["cameras"]}
        c["cameras"] = [dict(cams[cam], url=path, start_ts=ts) for cam, path, ts in g]
        start = min(ts for _, _, ts in g)
        when = dt.datetime.fromtimestamp(start, dt.timezone(dt.timedelta(hours=tz))).strftime("%Y-%m-%d %H:%M")
        print(f"[video] session {k}/{len(todo)}: {when}, {', '.join(cam for cam, _, _ in g)}", flush=True)
        models = models or pipeline.Models(c)
        t = dt.datetime.now()
        stats = pipeline.run(c, start, models=models)
        print(f"[video]   done in {(dt.datetime.now() - t).total_seconds() / 60:.0f} min, "
              f"{sum(v.get('bursts_missed', 0) for v in stats.values())} bursts missed; "
              f"{json.dumps({cam: {k: v for k, v in s.items() if k.startswith('id_')} for cam, s in stats.items()})}",
              flush=True)
        with open(done_path, "a", encoding="utf-8") as fh:
            fh.writelines(p + "\n" for _, p, _ in g)
    rows = read_cache(cfg["training_cache"]["folder"])
    print(f"[video] the cache: {len(rows)} bursts of {len({r['track'] for r in rows})} tracks on "
          f"{len({r['day'] for r in rows})} days -> next: herd.py barn sheet --cache {cfg['training_cache']['folder']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--merges", default=None, help="identity,track per line")
    b.add_argument("--gallery-labels", type=int, default=0)
    b.add_argument("--min-bursts", type=int, default=2, help="identities with fewer bursts are left out")
    b.add_argument("--val-from", default=None, help="YYYY-MM-DD: days from it on go to val (across-day re-ID)")
    s = sub.add_parser("sheet", help="the labelling page: name each track's cow -> merges.csv")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", default="tracks.html")
    s.add_argument("--day", default=None)
    s.add_argument("--merges", default=None, help="a merges.csv saved earlier: its names filled in")
    s.add_argument("--names", default=None, help="the barn's names.json: the gallery's guess shown as the ear tag")
    s.add_argument("--thumbs", type=int, default=12, help="thumbnails a track")
    v = sub.add_parser("video", help="recorded camera files -> the training cache (every burst), a store of its own")
    v.add_argument("--config", required=True, help="the barn's config: its cameras (ids, masks), models")
    v.add_argument("--videos", default=None, help="a folder of recordings named <camera>..<date and time>")
    v.add_argument("--video", action="append", default=[], help="cam1=file.mp4 (with --start), instead of --videos")
    v.add_argument("--start", default=None, help="--video: when the recordings began, local time (ISO)")
    v.add_argument("--out", required=True, help="its store, gallery and training_cache/")
    v.add_argument("--burst-every", type=float, default=20.0, help="a burst every this many s a camera")
    v.add_argument("--gap", type=float, default=300.0, help="recordings starting within this many s play together")
    v.add_argument("--gallery", default=None, help="the barn's gallery/ copied in first: its guesses on the page")
    x = sub.add_parser("export", help="named tracks -> herd_barn_feat_*.tgz for the identity model (reid_pod.sh)")
    x.add_argument("--cache", required=True)
    x.add_argument("--merges", required=True, help="identity,track per line (the page's Save merges.csv)")
    x.add_argument("--out", default=None, help="default herd_barn_feat_<date>.tgz here")
    x.add_argument("--frames", type=int, default=35, help="frames a burst at most (35: 5 a second)")
    x.add_argument("--min-bursts", type=int, default=4, help="cows with fewer bursts are left out")
    x.add_argument("--val-share", type=float, default=1 / 6, help="cows held out (never trained on)")
    x.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "video" and not (args.videos or args.video):
        p.error("video: --videos DIR or --video cam=file")
    {"build": cmd_build, "sheet": cmd_sheet, "export": cmd_export, "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
