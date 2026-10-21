#!/usr/bin/env python3
"""The cameras' zones drawn on their own pictures instead of typed as numbers.

  page   a still of every camera (from its recording, or --image) with the detector's cows on it ->
         OUT/zones.html: draw each camera's mask (the floor it counts), exclude (greyed before the
         detector: neighbouring pens, the yard), tracking_mask (with [mct]) and floor points (with [mct]:
         marks on the floor, X and Y measured in cm); the cows are coloured live by the rule the barn
         uses (green kept, red in the excluded zone, orange too small, dashed another camera's floor).
         "Save cameras.toml" writes the [[cameras]] blocks.
  apply  cameras.toml -> barn.toml: its [[cameras]] blocks replaced (barn.toml.bak kept).

    python herd/herd.py zones page  --config barn.toml --videos /srv/herd/rec --out /srv/herd/zones
    python herd/herd.py zones page  --config barn.toml --image cam1=cam1.jpg --image cam2=cam2.jpg --out zones
    python herd/herd.py zones apply --config barn.toml --cameras cameras.toml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402,F401  (puts cowbench on the path)
from common import load_config, recordings  # noqa: E402

KEYS = ("mask", "exclude", "tracking_mask", "floor_points")


def still(path, at, width):
    """One frame of a recording, at `at` s (or its middle when shorter), as the pipeline sizes it."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(at * fps), max(0, n // 2)))
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return img


def cmd_page(args):
    import numpy as np
    from PIL import Image
    cfg = load_config(args.config)
    by_id = {c["id"]: c for c in cfg["cameras"]}
    for cid in args.cameras or []:               # cameras not in the config yet: they get a block too
        by_id.setdefault(cid, {"id": cid})
    if args.cameras:                             # only these, in this order
        by_id = {cid: by_id[cid] for cid in args.cameras}
    pictures = {}
    for v in args.image:
        cam, _, path = v.partition("=")
        pictures[cam] = np.asarray(Image.open(path).convert("RGB"))
    if args.videos:
        recs, bad = recordings(args.videos, list(by_id), cfg["farm"]["timezone_offset_hours"])
        for cam, path, _ in recs:
            if cam not in pictures:
                img = still(path, args.at, cfg["sampling"]["frame_width"])
                if img is not None:
                    pictures[cam] = img
        if bad:
            print(f"[zones] {len(bad)} files without a camera id of the config or a time in their name: "
                  f"{', '.join(os.path.basename(b) for b in bad[:4])}")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "stills"), exist_ok=True)
    live = None
    if args.detector != "none" and pictures:
        import detector as det_mod
        weights = args.detector or cfg["detector"]["weights"]
        if not (weights and os.path.exists(os.path.join(weights, "det_train_meta.json"))):
            import glob
            found = sorted(glob.glob(os.path.join(os.environ.get("HERD_MODELS", "/models"), "**", "best",
                                                  "det_train_meta.json"), recursive=True), key=os.path.getmtime)
            print(f"[zones] [detector] weights = \"{weights}\" in {args.config}: no detector there (a best/ folder "
                  f"with det_train_meta.json)" + (f" - this one is used: {os.path.dirname(found[-1])}; put it in "
                                                  f"the config too" if found else " - the page shows no cows"))
            weights = os.path.dirname(found[-1]) if found else None
        if weights:
            live = det_mod.Live(weights, threshold=cfg["detector"]["threshold"], tiles=cfg["detector"].get("tiles"))
    cams = []
    ids = list(by_id) + sorted(c for c in pictures if c not in by_id)
    for cid in ids:
        c = by_id.get(cid, {"id": cid})
        rec = {"id": cid, "url": c.get("url", ""), "min_box_pct": c.get("min_box_pct") or 0.0,
               **{k: c.get(k) or [] for k in KEYS}, "still": None, "w": 16, "h": 9, "boxes": []}
        img = pictures.get(cid)
        if img is not None:
            rel = f"stills/{cid}.jpg"
            if live is not None:      # the detector sees the excluded zone grey; the page shows the picture
                found = live(Image.fromarray(img))
                rec["boxes"] = [[round(v, 4) for v in d["bbox"]] + [round(float(d["det_score"]), 3)] for d in found]
            Image.fromarray(img).save(os.path.join(out, rel), quality=90)
            rec.update(still=rel, w=int(img.shape[1]), h=int(img.shape[0]))
        cams.append(rec)
    data = {"key": out, "cams": cams}
    with open(os.path.join(out, "zones.html"), "w", encoding="utf-8") as fh:
        fh.write(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    shown = sum(1 for c in cams if c["still"])
    print(f"[zones] {len(cams)} cameras ({shown} with a picture) -> {os.path.join(out, 'zones.html')} "
          f"(open it in a browser; Save cameras.toml, then: herd.py zones apply --config {args.config} "
          f"--cameras cameras.toml)")


def cmd_apply(args):
    text = open(args.cameras, encoding="utf-8").read()
    try:
        new = tomllib.loads(text).get("cameras") or []
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"{args.cameras}: not TOML ({e})")
    if not new:
        sys.exit(f"{args.cameras}: no [[cameras]] in it")
    cfg_text = open(args.config, encoding="utf-8").read()
    m = re.search(r"^\[\[cameras\]\]", cfg_text, flags=re.M)
    head = cfg_text[:m.start()] if m else cfg_text.rstrip() + "\n\n"
    rest = cfg_text[m.start():] if m else ""
    if rest and re.search(r"^\[(?!\[cameras\]\])", rest, flags=re.M):
        sys.exit(f"{args.config}: other sections come after its [[cameras]] - move the cameras to the end first")
    shutil.copy(args.config, args.config + ".bak")
    with open(args.config, "w", encoding="utf-8") as fh:
        fh.write(head + text.strip() + "\n")
    cams = load_config(args.config)["cameras"]
    print(f"[zones] {args.config}: {len(cams)} cameras ({', '.join(c['id'] for c in cams)}); the old file is "
          f"{args.config}.bak")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera zones</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1da;--accent:#2f6f4f}
@media (prefers-color-scheme: dark){:root{--bg:#171816;--card:#20221f;--fg:#ecebe6;--mute:#9a9a93;--line:#34362f;--accent:#7fc29c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line)}h1{font-size:17px;margin:0 0 6px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0}
button,select,input{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px}
button{cursor:pointer}button.on{background:var(--accent);color:var(--bg);border-color:var(--accent)}button.main{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.help{color:var(--mute);font-size:13px;margin:2px 0}.mute{color:var(--mute)}
main{padding:8px 16px 30px}canvas{display:block;max-width:100%;touch-action:none;cursor:crosshair;border-radius:6px;background:#000}
textarea{width:100%;height:220px;font:12px ui-monospace,monospace;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px}
.legend span{margin-right:12px;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style></head><body>
<header>
<h1>Camera zones</h1>
<p class="help">Pick a camera and a layer, then click round the area: a click adds a corner, a click on the first corner (or
<b>Enter</b>) closes it; <b>Esc</b> drops an unfinished one. Drag a corner to move it; click inside a shape and press
<b>Del</b> to remove it. <b>mask</b>: the floor this camera counts (empty = the whole picture; with several cameras over one
floor every patch in exactly one mask). <b>exclude</b>: greyed before the detector - neighbouring pens, the yard, cows that
are not this pen's. <b>tracking</b> and <b>floor points</b> only for [mct] (a floor point: a mark on the floor, its X and Y
measured in cm from one corner shared by all cameras).</p>
<div class="bar" id="cams"></div>
<div class="bar"><span class="mute">layer:</span>
<button data-l="mask" class="on">mask</button><button data-l="exclude">exclude</button>
<button data-l="tracking_mask">tracking</button><button data-l="floor_points">floor points</button>
<label>url <input id="url" size="40" placeholder="rtsp://user:pass@10.0.0.11:554/stream1"></label>
<label>min_box_pct <input id="pct" size="4" type="number" step="0.1" min="0"></label></div>
<div class="bar legend"><span><i class="dot" style="background:#3ee07a"></i>cow kept</span><span><i class="dot" style="background:#ff4040"></i>in exclude</span>
<span><i class="dot" style="background:#ffa020"></i>too small</span><span><i class="dot" style="background:#ffffff"></i>dashed: another camera's floor</span>
<span class="mute" id="count"></span></div>
</header>
<main><canvas id="cv"></canvas>
<div class="bar"><button class="main" id="save">Save cameras.toml</button><button id="copy">Copy</button><button id="reset">This camera as in the config</button></div>
<textarea id="toml" readonly></textarea></main>
<script>
const DATA = __DATA__;
const KEY = "herd-zones:" + DATA.key;
let state = {};
DATA.cams.forEach(c => state[c.id] = {url: c.url, min_box_pct: c.min_box_pct, mask: polys(c.mask), exclude: polys(c.exclude),
  tracking_mask: polys(c.tracking_mask), floor_points: (c.floor_points || []).map(p => p.slice())});
try { const s = JSON.parse(localStorage.getItem(KEY) || "{}"); for (const k in s) if (state[k]) state[k] = s[k]; } catch (e) {}
function polys(m) { return !m || !m.length ? [] : (Array.isArray(m[0][0]) ? m.map(p => p.map(q => q.slice())) : [m.map(q => q.slice())]); }
const $ = id => document.getElementById(id), cv = $("cv"), ctx = cv.getContext("2d");
let cam = DATA.cams.find(c => c.still) || DATA.cams[0], layer = "mask", cur = null, sel = null, drag = null, img = new Image();
const COL = {mask: "#3ee07a", exclude: "#ff4040", tracking_mask: "#4aa3ff"};
function store() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
$("cams").innerHTML = DATA.cams.map(c => `<button data-c="${c.id}">${c.id}${c.still ? "" : " (no picture)"}</button>`).join("");
$("cams").querySelectorAll("button").forEach(b => b.onclick = () => { cam = DATA.cams.find(c => c.id === b.dataset.c); open(); });
document.querySelectorAll("[data-l]").forEach(b => b.onclick = () => { layer = b.dataset.l; cur = null; sel = null;
  document.querySelectorAll("[data-l]").forEach(x => x.classList.toggle("on", x === b)); draw(); });
$("url").oninput = () => { state[cam.id].url = $("url").value; store(); toml(); };
$("pct").oninput = () => { state[cam.id].min_box_pct = parseFloat($("pct").value) || 0; store(); draw(); toml(); };
function open() {
  $("cams").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.c === cam.id));
  $("url").value = state[cam.id].url || ""; $("pct").value = state[cam.id].min_box_pct || 0;
  cur = null; sel = null;
  const W = Math.min(document.querySelector("main").clientWidth - 32, 1400);
  cv.width = W; cv.height = Math.round(W * cam.h / cam.w);
  if (cam.still) { img = new Image(); img.onload = draw; img.src = cam.still; } else { img = null; draw(); }
  toml();
}
function inPoly(x, y, p) { let ins = false; for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
  const [x1, y1] = p[i], [x2, y2] = p[j]; if ((y1 > y) !== (y2 > y) && x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1) ins = !ins; } return ins; }
function verdict(b) {
  const s = state[cam.id], cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
  if (s.exclude.some(p => inPoly(cx, cy, p))) return "excluded";
  if ((b[2] - b[0]) * (b[3] - b[1]) < (s.min_box_pct || 0) / 100) return "small";
  if (s.mask.length && !s.mask.some(p => inPoly(cx, cy, p))) return "other";
  return "kept";
}
function draw() {
  const W = cv.width, H = cv.height, X = v => v * W, Y = v => v * H, s = state[cam.id];
  ctx.fillStyle = "#222"; ctx.fillRect(0, 0, W, H);
  if (img && img.complete) ctx.drawImage(img, 0, 0, W, H);
  else { ctx.fillStyle = "#aaa"; ctx.font = "16px sans-serif"; ctx.fillText("no picture of this camera: --videos or --image", 20, 30); }
  const path = p => { ctx.beginPath(); p.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.closePath(); };
  s.exclude.forEach(p => { path(p); ctx.fillStyle = "rgba(114,114,114,0.75)"; ctx.fill(); });
  ["mask", "tracking_mask", "exclude"].forEach(l => s[l].forEach((p, i) => {
    path(p); ctx.lineWidth = sel && sel.l === l && sel.i === i ? 4 : 2; ctx.strokeStyle = COL[l];
    ctx.setLineDash(l === "tracking_mask" ? [8, 5] : []); ctx.stroke(); ctx.setLineDash([]);
    if (l === "mask") { ctx.fillStyle = "rgba(62,224,122,0.10)"; ctx.fill(); }
    if (l === layer) p.forEach(([x, y]) => { ctx.fillStyle = COL[l]; ctx.fillRect(X(x) - 4, Y(y) - 4, 8, 8); });
  }));
  const n = {kept: 0, excluded: 0, small: 0, other: 0};
  cam.boxes.forEach(b => {
    const v = verdict(b); n[v]++;
    ctx.lineWidth = 2; ctx.setLineDash(v === "other" ? [6, 4] : []);
    ctx.strokeStyle = {kept: "#3ee07a", excluded: "#ff4040", small: "#ffa020", other: "#ffffff"}[v];
    ctx.strokeRect(X(b[0]), Y(b[1]), X(b[2] - b[0]), Y(b[3] - b[1])); ctx.setLineDash([]);
  });
  $("count").textContent = cam.boxes.length ? `cows found: ${cam.boxes.length} - kept ${n.kept}, excluded ${n.excluded}, too small ${n.small}, another camera's ${n.other}` : "";
  s.floor_points.forEach(([u, v, fx, fy], i) => {
    ctx.fillStyle = sel && sel.l === "floor_points" && sel.i === i ? "#fff" : "#ffe94a";
    ctx.beginPath(); ctx.arc(X(u), Y(v), 6, 0, 7); ctx.fill();
    ctx.font = "13px sans-serif"; ctx.fillText(`${fx}, ${fy}`, X(u) + 9, Y(v) - 6);
  });
  if (cur) {
    ctx.strokeStyle = COL[layer]; ctx.lineWidth = 2; ctx.beginPath();
    cur.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.stroke();
    cur.forEach(([x, y], i) => { ctx.fillStyle = i ? COL[layer] : "#fff"; ctx.fillRect(X(x) - 4, Y(y) - 4, 8, 8); });
  }
}
function pt(e) { const r = cv.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))]; }
const near = (a, b) => { const r = cv.getBoundingClientRect(); return Math.abs(a[0] - b[0]) * r.width < 9 && Math.abs(a[1] - b[1]) * r.height < 9; };
function changed() { store(); draw(); toml(); }
cv.addEventListener("pointerdown", e => {
  const p = pt(e), s = state[cam.id];
  if (layer === "floor_points") {
    const i = s.floor_points.findIndex(f => near(f, p));
    if (i >= 0) { sel = {l: layer, i}; drag = {fp: i}; cv.setPointerCapture(e.pointerId); draw(); return; }
    const xy = prompt("This mark's X and Y on the floor, in cm (e.g. 0 1200):");
    if (!xy) return;
    const m = xy.trim().split(/[\s,;]+/).map(Number);
    if (m.length < 2 || m.some(isNaN)) { alert("two numbers, X and Y in cm"); return; }
    s.floor_points.push([+p[0].toFixed(4), +p[1].toFixed(4), m[0], m[1]]); changed(); return;
  }
  if (cur) {
    if (cur.length >= 3 && near(cur[0], p)) { s[layer].push(cur); cur = null; changed(); return; }
    cur.push(p); draw(); return;
  }
  for (let i = 0; i < s[layer].length; i++) {
    const k = s[layer][i].findIndex(q => near(q, p));
    if (k >= 0) { sel = {l: layer, i}; drag = {i, k}; cv.setPointerCapture(e.pointerId); draw(); return; }
  }
  const inside = s[layer].findIndex(q => inPoly(p[0], p[1], q));
  if (inside >= 0 && !(sel && sel.l === layer && sel.i === inside)) { sel = {l: layer, i: inside}; draw(); return; }
  sel = null; cur = [p]; draw();
});
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  const p = pt(e).map(v => +v.toFixed(4)), s = state[cam.id];
  if (drag.fp !== undefined) { s.floor_points[drag.fp][0] = p[0]; s.floor_points[drag.fp][1] = p[1]; }
  else s[layer][drag.i][drag.k] = p;
  draw();
});
cv.addEventListener("pointerup", () => { if (drag) { drag = null; changed(); } });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const s = state[cam.id];
  if (e.key === "Enter" && cur && cur.length >= 3) { s[layer].push(cur); cur = null; changed(); }
  else if (e.key === "Escape") { cur = null; sel = null; draw(); }
  else if ((e.key === "Delete" || e.key === "Backspace") && sel) { s[sel.l].splice(sel.i, 1); sel = null; changed(); e.preventDefault(); }
});
const r3 = v => Math.round(v * 1000) / 1000;
const fmt = ps => ps.length === 1 ? JSON.stringify(ps[0].map(q => q.map(r3))) : JSON.stringify(ps.map(p => p.map(q => q.map(r3))));
function toml() {
  const out = [];
  DATA.cams.forEach(c => {
    const s = state[c.id];
    out.push("[[cameras]]", `id = "${c.id}"`, `url = "${(s.url || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`);
    ["mask", "exclude", "tracking_mask"].forEach(k => { if (s[k].length) out.push(`${k} = ${fmt(s[k]).replace(/,/g, ", ")}`); });
    if (s.floor_points.length) out.push(`floor_points = ${JSON.stringify(s.floor_points.map(f => [r3(f[0]), r3(f[1]), f[2], f[3]])).replace(/,/g, ", ")}`);
    if (s.min_box_pct) out.push(`min_box_pct = ${s.min_box_pct}`);
    out.push("");
  });
  $("toml").value = out.join("\n");
}
$("save").onclick = () => { const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([$("toml").value], {type: "text/plain"})); a.download = "cameras.toml"; a.click(); };
$("copy").onclick = () => { $("toml").select(); try { navigator.clipboard.writeText($("toml").value); } catch (e) { document.execCommand("copy"); } };
$("reset").onclick = () => { const c = cam; state[c.id] = {url: c.url, min_box_pct: c.min_box_pct, mask: polys(c.mask), exclude: polys(c.exclude),
  tracking_mask: polys(c.tracking_mask), floor_points: (c.floor_points || []).map(p => p.slice())}; open(); store(); };
window.addEventListener("resize", open);
open();
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("page", help="a still of every camera -> zones.html, to draw the zones on")
    g.add_argument("--config", required=True)
    g.add_argument("--videos", default=None, help="recordings named <camera>..<date and time>: a still of each camera")
    g.add_argument("--image", action="append", default=[], help="cam1=picture.jpg (instead of, or besides, --videos)")
    g.add_argument("--cameras", nargs="*", default=None, help="the camera ids (default: the config's), e.g. "
                   "cam1 cam2 cam3 cam4 - recordings are found by them")
    g.add_argument("--at", type=float, default=30.0, help="the still is this many seconds into the recording")
    g.add_argument("--detector", default=None, help="a best/ folder (default: the config's); none: no cows drawn")
    g.add_argument("--out", required=True)
    a = sub.add_parser("apply", help="cameras.toml (the page's Save) -> the config's [[cameras]] blocks")
    a.add_argument("--config", required=True)
    a.add_argument("--cameras", required=True)
    args = p.parse_args(argv)
    if args.cmd == "page" and not (args.videos or args.image):
        p.error("page: --videos DIR or --image cam=picture")
    {"page": cmd_page, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
