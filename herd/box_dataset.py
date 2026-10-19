#!/usr/bin/env python3
"""Cow boxes on the barn's own cameras: a detector training set made from their recordings.

The detector was trained on CBVD-5 (side view) and public top-down sets; the barn's cameras, light
and pens are its own. A few hundred frames of them, boxes corrected by hand, teach it the barn.

  1. frames  recordings -> one frame every --every s per camera (frames that barely changed are
             skipped; at most --max a camera, spread over all the recordings), cut and greyed as the
             pipeline gives them to the detector (frame_width, the camera's `exclude` zone grey), the
             detector's boxes on each as a first guess -> DATA/frames/<cam>/*.jpg, DATA/frames.json
             and the labelling page DATA/boxes.html
  2. label   (a person, in a browser) boxes.html: move, resize, delete, draw boxes; a frame marked
             done (Enter) is exported, one not marked is not. "Save labels.json" writes the result.
  3. export  -> herd_det_barn_<date>.tgz: COCO train/ and valid/ (valid: whole hours of a camera, a
             fifth, never next to a training frame of the same minutes) as extra_det/barn - what
             run_pod.sh's detector step (DET_ARCH=rfdetr) takes in beside CBVD-5 and the public sets.

    python herd/herd.py boxes frames --config /srv/herd/barn.toml --videos /srv/herd/rec --out /srv/herd/boxes
    python herd/herd.py boxes page   --data /srv/herd/boxes --labels labels.json   # the page again, names kept
    python herd/herd.py boxes export --data /srv/herd/boxes --labels labels.json

Recordings are named by camera and local start time (cam1_2026-10-20_06-00-00.mp4, cam1/20261020T060000.mkv
...): see common.recordings. One file: --video cam1=file.mp4 --start 2026-10-20T06:00:00.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402,F401  (puts cowbench on the path)
from common import blank_excluded, load_config, recordings  # noqa: E402


def _tz(cfg):
    return dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def plan_times(recs, every, max_n):
    """[(path, start ts, fps, n frames)] of one camera -> [(path, frame index, ts)]: a frame every `every`
    s, thinned evenly to at most max_n over all its recordings."""
    out = []
    for path, ts, fps, n in recs:
        step = max(1, int(round(every * fps)))
        out += [(path, i, ts + i / fps) for i in range(step // 2, n, step)]
    if max_n and len(out) > max_n:
        keep = np.unique(np.round(np.linspace(0, len(out) - 1, max_n)).astype(int))
        out = [out[i] for i in keep]
    return out


def changed(a, b, min_change):
    """Two small grey thumbnails differ by more than min_change (0-255, mean) - a frame worth labelling."""
    return b is None or float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) >= min_change


def cmd_frames(args):
    import cv2
    from PIL import Image
    cfg = load_config(args.config)
    tz = _tz(cfg)
    cams = {c["id"]: c for c in cfg["cameras"]}
    if args.video:
        if not args.start:
            sys.exit("--video needs --start (when the recording began, local time)")
        t0 = dt.datetime.fromisoformat(args.start)
        t0 = (t0 if t0.tzinfo else t0.replace(tzinfo=tz)).timestamp()
        recs = []
        for v in args.video:
            cam, _, path = v.partition("=")
            if cam not in cams or not os.path.exists(path):
                sys.exit(f"--video {v}: want <camera id>=<file>, a camera of {args.config} ({', '.join(cams)})")
            recs.append((cam, path, t0))
    else:
        recs, bad = recordings(args.videos, list(cams), cfg["farm"]["timezone_offset_hours"])
        if bad:
            print(f"[boxes] {len(bad)} files without a camera id or a time in their name - skipped, e.g. {bad[0]}")
        if not recs:
            sys.exit(f"no recordings under {args.videos} named <camera>..<YYYY-MM-DD_HH-MM-SS>")
    out = os.path.abspath(args.out)
    index_path = os.path.join(out, "frames.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else {"frames": []}
    have = {f["file"] for f in index["frames"]}
    width = cfg["sampling"]["frame_width"]
    by_cam = collections.defaultdict(list)
    for cam, path, ts in recs:
        info = video_info(path)
        if info is None:
            print(f"[boxes] cannot open {path} - skipped")
            continue
        by_cam[cam].append((path, ts) + info)
    for cam, rs in sorted(by_cam.items()):
        c = cams[cam]
        plan = plan_times(rs, args.every, args.max)
        folder = os.path.join(out, "frames", cam)
        os.makedirs(folder, exist_ok=True)
        kept, last, cap, cap_path = 0, None, None, None
        for path, i, ts in plan:
            name = f"{cam}_{dt.datetime.fromtimestamp(ts, tz):%Y%m%d_%H%M%S}.jpg"
            rel = f"frames/{cam}/{name}"
            if rel in have:
                kept += 1
                continue
            if path != cap_path:
                if cap is not None:
                    cap.release()
                cap, cap_path = cv2.VideoCapture(path), path
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                continue
            img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            if w > width:                                        # as Camera.push
                img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            if c.get("blank_excluded", True):                   # as Camera.blank: what the detector sees
                img = blank_excluded(img, c.get("exclude"))
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
            if not changed(small, last, args.min_change):
                continue
            last = small
            Image.fromarray(img).save(os.path.join(out, rel), quality=92)
            index["frames"].append({"file": rel, "cam": cam, "ts": ts, "w": img.shape[1], "h": img.shape[0],
                                    "time": dt.datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M:%S")})
            have.add(rel)
            kept += 1
        if cap is not None:
            cap.release()
        print(f"[boxes] {cam}: {kept} frames from {len(rs)} recordings ({len(plan)} looked at)", flush=True)
    index["frames"].sort(key=lambda f: (f["cam"], f["ts"]))
    if args.detector != "none":
        detect(index, out, args.detector or cfg["detector"]["weights"], cfg)
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    write_page(out, index, {})
    print(f"[boxes] {len(index['frames'])} frames -> {os.path.join(out, 'boxes.html')} (open it in a browser)")


def detect(index, out, weights, cfg):
    """The detector's boxes on every frame not yet looked at: those above its threshold drawn, those down
    to half of it offered (dashed) - a cow it nearly missed is one click."""
    from PIL import Image
    import detector as det_mod
    meta = json.load(open(os.path.join(weights, "det_train_meta.json"), encoding="utf-8"))
    thr = cfg["detector"]["threshold"] or meta["threshold"]
    todo = [f for f in index["frames"] if "boxes" not in f]
    index.update(threshold=thr, detector=os.path.abspath(weights))
    if not todo:
        return
    live = det_mod.Live(weights, threshold=thr / 2, tiles=cfg["detector"].get("tiles"))
    print(f"[boxes] the detector ({live.name}, threshold {thr}) on {len(todo)} frames ({live.device})", flush=True)
    for k in range(0, len(todo), 4):
        chunk = todo[k:k + 4]
        imgs = [Image.open(os.path.join(out, f["file"])).convert("RGB") for f in chunk]
        for f, found in zip(chunk, live.many(imgs)):
            f["boxes"] = [[round(v, 4) for v in d["bbox"]] + [round(float(d["det_score"]), 3)] for d in found]
        print(f"\r  {min(k + 4, len(todo))}/{len(todo)}", end="", flush=True)
    print(flush=True)


def read_labels(path):
    """labels.json (the page's Save) -> {frame file: {"status": "done" | "skip", "boxes": [[x1, y1, x2, y2] 0-1]}}."""
    if not path:
        return {}
    d = json.load(open(path, encoding="utf-8"))
    return d.get("frames", d)


def write_page(out, index, labels):
    data = {"key": out, "threshold": index.get("threshold", 0.5), "frames": index["frames"], "labels": labels}
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    with open(os.path.join(out, "boxes.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


def cmd_page(args):
    out = os.path.abspath(args.data)
    index = json.load(open(os.path.join(out, "frames.json"), encoding="utf-8"))
    labels = read_labels(args.labels)
    write_page(out, index, labels)
    done = sum(1 for v in labels.values() if v.get("status") == "done")
    print(f"[boxes] {len(index['frames'])} frames ({done} done in {args.labels or 'no labels file'}) -> "
          f"{os.path.join(out, 'boxes.html')}")


def split_blocks(frames, val_share, block_s, seed):
    """Done frames -> (train, valid): whole blocks (a camera's hour) go to valid, so no valid frame has a
    training frame of the same minutes beside it; at least one block each when there are two."""
    blocks = collections.defaultdict(list)
    for f in frames:
        blocks[(f["cam"], int(f["ts"] // block_s))].append(f)
    keys = sorted(blocks)
    random.Random(seed).shuffle(keys)
    val, n_val = set(), 0
    for k in keys:
        if n_val >= val_share * len(frames) or len(val) + 1 >= len(keys):
            break
        val.add(k)
        n_val += len(blocks[k])
    tr = [f for k in keys if k not in val for f in blocks[k]]
    va = [f for k in keys if k in val for f in blocks[k]]
    return tr, va


def write_coco(folder, frames, src):
    import shutil
    os.makedirs(folder, exist_ok=True)
    images, anns = [], []
    for i, f in enumerate(frames, 1):
        name = os.path.basename(f["file"])
        shutil.copy(os.path.join(src, f["file"]), os.path.join(folder, name))
        w, h = f["w"], f["h"]
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for b in f["labelled"]:
            x, y, bw, bh = b[0] * w, b[1] * h, (b[2] - b[0]) * w, (b[3] - b[1]) * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1,
                         "bbox": [round(x, 1), round(y, 1), round(bw, 1), round(bh, 1)], "area": bw * bh, "iscrowd": 0})
    with open(os.path.join(folder, "_annotations.coco.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "cow", "supercategory": "none"}]}, fh)
    return len(anns)


def cmd_export(args):
    import shutil
    import tarfile
    import tempfile
    src = os.path.abspath(args.data)
    index = json.load(open(os.path.join(src, "frames.json"), encoding="utf-8"))
    labels = read_labels(args.labels)
    done = []
    for f in index["frames"]:
        lab = labels.get(f["file"])
        if lab and lab.get("status") == "done":
            boxes = [b[:4] for b in lab.get("boxes", []) if b[2] - b[0] > 0.002 and b[3] - b[1] > 0.002]
            done.append(dict(f, labelled=boxes))
    if len(done) < 2:
        sys.exit(f"{len(done)} frames marked done in {args.labels}: label more (Enter on the page marks a frame done)")
    tr, va = split_blocks(done, args.val_share, args.block_min * 60, args.seed)
    if not va:
        sys.exit("all done frames are in one hour of one camera: label frames of other hours too (they are "
                 "the held-out check)")
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "extra_det", "barn")
    n_tr = write_coco(os.path.join(root, "train"), tr, src)
    n_va = write_coco(os.path.join(root, "valid"), va, src)
    shutil.copy(args.labels, os.path.join(root, "labels.json"))
    out = args.out or f"herd_det_barn_{dt.datetime.now():%Y%m%d_%H%M}.tgz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(os.path.join(tmp, "extra_det"), arcname="extra_det")
    shutil.rmtree(tmp)
    empty = sum(1 for f in done if not f["labelled"])
    cams = collections.Counter(f["cam"] for f in done)
    print(f"[boxes] {len(done)} frames ({', '.join(f'{c} {n}' for c, n in sorted(cams.items()))}; {empty} with no "
          f"cow): train {len(tr)} frames / {n_tr} cows, valid {len(va)} frames / {n_va} cows -> {out} "
          f"({os.path.getsize(out) / 2 ** 20:.0f} MB)")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Barn cow boxes</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1da;--accent:#2f6f4f;--done:#2f6f4f;--skip:#a33}
@media (prefers-color-scheme: dark){:root{--bg:#171816;--card:#20221f;--fg:#ecebe6;--mute:#9a9a93;--line:#34362f;--accent:#7fc29c;--done:#7fc29c;--skip:#e88}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line)}h1{font-size:17px;margin:0 0 6px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0}
button,select{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.main{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.mute{color:var(--mute)}.help{color:var(--mute);font-size:13px;margin:2px 0}
main{padding:8px 16px 30px}#wrap{position:relative}canvas{display:block;max-width:100%;touch-action:none;cursor:crosshair;border-radius:6px;background:#000}
#status.done{color:var(--done);font-weight:600}#status.skip{color:var(--skip);font-weight:600}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
</style></head><body>
<header>
<h1>Barn cow boxes</h1>
<p class="help">Draw a box round every cow, or the part of her you can see: drag on empty space. Drag a box to move it,
its corners to resize; <kbd>Del</kbd> removes the selected box. Dashed boxes are cows the detector was unsure of: click to keep,
<kbd>A</kbd> keeps all. Grey is the camera's excluded zone: no boxes there. <kbd>Enter</kbd> marks the frame done and opens
the next; only frames marked done are exported. <kbd>S</kbd> skips a frame (spoilt, or unsure); <kbd>&larr;</kbd>
<kbd>&rarr;</kbd> move without marking; <kbd>Ctrl+Z</kbd> undoes.</p>
<div class="bar">
<button id="prev">&larr;</button><span id="pos"></span><button id="next">&rarr;</button>
<select id="cam"></select><label><input type="checkbox" id="todo"> only not done</label>
<button class="main" id="done">Done &#10003;</button><button id="skip">Skip</button><button id="reset">Detector's boxes</button>
<button class="main" id="save">Save labels.json</button><button id="loadbtn">Load labels.json</button><input type="file" id="load" accept=".json" hidden>
</div>
<div class="bar"><span id="info"></span><span id="status"></span><span class="mute" id="stat"></span></div>
</header>
<main><div id="wrap"><canvas id="cv"></canvas></div></main>
<script>
const DATA = __DATA__;
const KEY = "herd-boxes:" + DATA.key;
let labels = Object.assign({}, DATA.labels);
try { Object.assign(labels, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let order = [], at = 0, img = new Image(), boxes = [], cands = [], sel = -1, undo = [], drag = null, scale = 1;
function store() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
const iou = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i || 1); };
function detBoxes(f) {
  const det = f.boxes || [];
  return [det.filter(b => b[4] >= DATA.threshold).map(b => b.slice(0, 4)), det.filter(b => b[4] < DATA.threshold)];
}
const cams = [...new Set(DATA.frames.map(f => f.cam))];
$("cam").innerHTML = '<option value="">all cameras</option>' + cams.map(c => `<option>${c}</option>`).join("");
function filter(keep) {
  const c = $("cam").value, todo = $("todo").checked;
  order = DATA.frames.map((f, i) => i).filter(i => (!c || DATA.frames[i].cam === c) &&
    (!todo || !labels[DATA.frames[i].file] || !labels[DATA.frames[i].file].status || i === keep));
  at = Math.max(0, order.indexOf(keep));
}
function frame() { return DATA.frames[order[at]]; }
function open() {
  const f = frame();
  if (!f) { ctx.clearRect(0, 0, cv.width, cv.height); $("pos").textContent = "0 / 0"; $("info").textContent = "no frames here"; return; }
  const lab = labels[f.file], [det, low] = detBoxes(f);
  boxes = lab ? lab.boxes.map(b => b.slice()) : det;
  cands = low.filter(c => !boxes.some(b => iou(b, c) > 0.5));
  sel = -1; undo = [];
  img = new Image();
  img.onload = () => { fit(); draw(); };
  img.src = f.file;
  $("pos").textContent = `${at + 1} / ${order.length}`;
  $("info").textContent = `${f.cam}  ${f.time}`;
  status();
}
function fit() {
  const W = $("wrap").clientWidth, H = window.innerHeight * 0.78;
  scale = Math.min(W / img.naturalWidth, H / img.naturalHeight, 1.5);
  cv.width = Math.round(img.naturalWidth * scale); cv.height = Math.round(img.naturalHeight * scale);
}
function status() {
  const f = frame(), s = f && labels[f.file] ? labels[f.file].status || "" : "";
  $("status").textContent = s === "done" ? "done" : s === "skip" ? "skipped" : "not done";
  $("status").className = s;
  const all = Object.values(labels), d = all.filter(l => l.status === "done");
  $("stat").textContent = `${d.length} of ${DATA.frames.length} frames done, ${d.reduce((n, l) => n + l.boxes.length, 0)} cows, ${all.filter(l => l.status === "skip").length} skipped`;
}
function draw() {
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  const X = v => v * cv.width, Y = v => v * cv.height;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]); ctx.strokeStyle = "#f0a020"; ctx.font = "12px sans-serif"; ctx.fillStyle = "#f0a020";
  cands.forEach(c => { ctx.strokeRect(X(c[0]), Y(c[1]), X(c[2] - c[0]), Y(c[3] - c[1])); ctx.fillText(c[4].toFixed(2), X(c[0]) + 3, Y(c[1]) + 13); });
  ctx.setLineDash([]);
  boxes.forEach((b, i) => {
    ctx.strokeStyle = i === sel ? "#ffe94a" : "#3ee07a";
    ctx.strokeRect(X(b[0]), Y(b[1]), X(b[2] - b[0]), Y(b[3] - b[1]));
    if (i === sel) { ctx.fillStyle = "#ffe94a"; corners(b).forEach(([x, y]) => ctx.fillRect(X(x) - 5, Y(y) - 5, 10, 10)); }
  });
}
const corners = b => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
function edited() {
  const f = frame();
  labels[f.file] = {status: labels[f.file] ? labels[f.file].status || "" : "", boxes: boxes.map(b => b.map(v => Math.round(v * 1e4) / 1e4))};
  store(); status(); draw();
}
function push() { undo.push(JSON.stringify([boxes, cands])); if (undo.length > 50) undo.shift(); }
function pt(e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; }
cv.addEventListener("pointerdown", e => {
  if (!frame()) return;
  const [x, y] = pt(e), tol = 8 / cv.getBoundingClientRect().width, toly = 8 / cv.getBoundingClientRect().height;
  cv.setPointerCapture(e.pointerId);
  if (sel >= 0) {
    const k = corners(boxes[sel]).findIndex(([cx, cy]) => Math.abs(cx - x) < tol && Math.abs(cy - y) < toly);
    if (k >= 0) { push(); drag = {mode: "resize", k}; return; }
  }
  const inside = boxes.map((b, i) => [i, b]).filter(([, b]) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
    .sort((p, q) => (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]) - (q[1][2] - q[1][0]) * (q[1][3] - q[1][1]));
  if (inside.length) { push(); sel = inside[0][0]; drag = {mode: "move", x, y, b: boxes[sel].slice()}; draw(); return; }
  const c = cands.findIndex(b => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]);
  if (c >= 0) { push(); boxes.push(cands[c].slice(0, 4)); cands.splice(c, 1); sel = boxes.length - 1; edited(); return; }
  push(); boxes.push([x, y, x, y]); sel = boxes.length - 1; drag = {mode: "new", x, y};
});
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  const [x, y] = pt(e).map(v => Math.min(1, Math.max(0, v))), b = boxes[sel];
  if (drag.mode === "move") {
    const dx = Math.min(1 - drag.b[2], Math.max(-drag.b[0], x - drag.x)), dy = Math.min(1 - drag.b[3], Math.max(-drag.b[1], y - drag.y));
    boxes[sel] = [drag.b[0] + dx, drag.b[1] + dy, drag.b[2] + dx, drag.b[3] + dy];
  } else if (drag.mode === "new") {
    boxes[sel] = [Math.min(drag.x, x), Math.min(drag.y, y), Math.max(drag.x, x), Math.max(drag.y, y)];
  } else {
    const k = drag.k;
    if (k === 0 || k === 3) b[0] = x; else b[2] = x;
    if (k === 0 || k === 1) b[1] = y; else b[3] = y;
    boxes[sel] = [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
    if (b[0] > b[2]) drag.k = [1, 0, 3, 2][drag.k];
    if (b[1] > b[3]) drag.k = [3, 2, 1, 0][drag.k];
  }
  draw();
});
cv.addEventListener("pointerup", () => {
  if (!drag) return;
  const b = boxes[sel];
  if (drag.mode === "new" && ((b[2] - b[0]) * cv.width < 6 || (b[3] - b[1]) * cv.height < 6)) { boxes.splice(sel, 1); sel = -1; undo.pop(); draw(); }
  else edited();
  drag = null;
});
function mark(s) {
  const f = frame(); if (!f) return;
  labels[f.file] = {status: s, boxes: boxes.map(b => b.map(v => Math.round(v * 1e4) / 1e4))};
  store(); status();
  if (at + 1 < order.length) { at++; open(); } else draw();
}
function go(d) { if (!order.length) return; at = Math.min(order.length - 1, Math.max(0, at + d)); open(); }
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
$("done").onclick = () => mark("done"); $("skip").onclick = () => mark("skip");
$("reset").onclick = () => { const f = frame(); if (!f) return; push(); [boxes, cands] = detBoxes(f); sel = -1; edited(); };
["cam", "todo"].forEach(id => $(id).onchange = () => { filter(order[at]); open(); });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if ((e.ctrlKey || e.metaKey) && e.key === "z") { if (undo.length) { [boxes, cands] = JSON.parse(undo.pop()); sel = -1; edited(); } e.preventDefault(); return; }
  if (e.key === "Enter") { mark("done"); e.preventDefault(); }
  else if (e.key === "s" || e.key === "S") mark("skip");
  else if (e.key === "ArrowRight") go(1);
  else if (e.key === "ArrowLeft") go(-1);
  else if ((e.key === "Delete" || e.key === "Backspace") && sel >= 0) { push(); boxes.splice(sel, 1); sel = -1; edited(); e.preventDefault(); }
  else if (e.key === "a" || e.key === "A") { push(); boxes.push(...cands.map(c => c.slice(0, 4))); cands = []; edited(); }
  else if (e.key === "Escape") { sel = -1; draw(); }
});
$("save").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({frames: labels})], {type: "application/json"}));
  a.download = "labels.json"; a.click();
};
$("loadbtn").onclick = () => $("load").click();
$("load").onchange = e => { const f = e.target.files[0]; if (!f) return;
  f.text().then(t => { const d = JSON.parse(t); Object.assign(labels, d.frames || d); store(); open(); }); };
window.addEventListener("resize", () => { if (img.naturalWidth) { fit(); draw(); } });
filter(0); open();
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames", help="recordings -> frames + the detector's boxes + the labelling page")
    f.add_argument("--config", required=True, help="the barn's config: cameras (ids, exclude zones), detector")
    f.add_argument("--videos", default=None, help="a folder of recordings named <camera>..<date and time>")
    f.add_argument("--video", action="append", default=[], help="cam1=file.mp4 (with --start), instead of --videos")
    f.add_argument("--start", default=None, help="--video: when the recording began, local time (ISO)")
    f.add_argument("--out", required=True)
    f.add_argument("--every", type=float, default=30.0, help="a frame every this many seconds")
    f.add_argument("--max", type=int, default=300, help="frames a camera at most, spread over all its recordings")
    f.add_argument("--min-change", type=float, default=3.0, help="a frame this little changed (0-255) is skipped")
    f.add_argument("--detector", default=None, help="a best/ folder (default: the config's); none: no first guess")
    g = sub.add_parser("page", help="the labelling page again (after a code update, or with a labels.json)")
    g.add_argument("--data", required=True)
    g.add_argument("--labels", default=None)
    x = sub.add_parser("export", help="labels.json -> herd_det_barn_<date>.tgz (COCO, for run_pod.sh detector)")
    x.add_argument("--data", required=True)
    x.add_argument("--labels", required=True)
    x.add_argument("--out", default=None)
    x.add_argument("--val-share", type=float, default=0.2, help="frames held out to check the detector")
    x.add_argument("--block-min", type=float, default=60.0, help="held out by whole blocks of a camera's minutes")
    x.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "frames" and not (args.videos or args.video):
        p.error("frames: --videos DIR or --video cam=file")
    {"frames": cmd_frames, "page": cmd_page, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
