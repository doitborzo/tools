#!/usr/bin/env python3
"""More cow boxes for the detector: three public sets fetched on the pod itself.

    zenodo    Pusweli (HAMK), "A Computer Vision-based system for dairy cow behaviour monitoring
              in a barn environment", zenodo.org/records/20426189 - CC BY 4.0. 771 frames of 11
              barn videos (angled view), ~8.7k whole-cow boxes (standing / lying; used as "cow").
    mmcows    Vu et al., "MmCows: A Multimodal Dataset for Dairy Cattle Monitoring" (Purdue NEIS),
              huggingface.co/datasets/neis-lab/mmcows - CC BY-NC-SA 4.0 (non-commercial; what is
              made from it carries the same licence). One day (7/25) of four high cameras over one
              pen, a frame every 15 s, 4480x2800, whole-cow boxes with the cow's ID; the other pens
              are masked black. Every EVERY-th frame per camera is taken (default 8: two minutes).
    opencows  Andrew et al., "OpenCows2020", University of Bristol, doi 10.5523/bris.10m32xl88x2b61zlkkgz3fml17
              - UK Non-Commercial Government Licence v2. Top-down (indoor walkway + drone over a
              field); 3,707 real frames (the synthetic ones after 003707.jpg are left out).
              NB: its boxes are the body WITHOUT the head, and cows cut by the frame's edge are
              often not boxed - not CBVD-5's habit (the whole cow); train with and without it.

Nothing is downloaded whole: the zips are read through HTTP Range requests (the
central directory, then only the members wanted), so MmCows' 20 GB zip costs ~2 GB.
Images are stored with the longer side at most --max-side (1920) px.

Each source becomes <dest>/<name>/{train,valid}/ with _annotations.coco.json (one class:
cow) - what rfdetr_det.py export --extra merges into CBVD-5's train / valid. Valid is
split off by video (zenodo), by hour of the day (mmcows: every 6th hour) and by the
authors' own fold 0 (opencows), so neighbouring frames never land on both sides.

    python cowbench/extra_det.py fetch --dest /workspace/extra_det --sources zenodo,mmcows,opencows
    python cowbench/extra_det.py eval  --data /workspace/extra_det --detector /workspace/herd/detector_rfdetr_large/best
    python cowbench/extra_det.py fetch --dest /workspace/extra_det --splits valid     # a test pod: valid only
    python cowbench/extra_det.py cbvd-videos --dest /workspace/cbvd5   # CBVD-5's val clips only, for the stress test
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import random
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "lora")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SOURCES = {
    "zenodo": {"url": "https://zenodo.org/api/records/20426189/files/CowAnalysis_Dataset.zip/content",
               "licence": "CC BY 4.0", "page": "https://zenodo.org/records/20426189",
               "cite": "N. Pusweli, A Computer Vision-based system for dairy cow behaviour monitoring "
                       "in a barn environment (dataset), HAMK, Zenodo 20426189"},
    "mmcows": {"url": "https://huggingface.co/datasets/neis-lab/mmcows/resolve/main/visual_data.zip",
               "licence": "CC BY-NC-SA 4.0", "page": "https://github.com/neis-lab/mmcows",
               "cite": "H. Vu et al., MmCows: A Multimodal Dataset for Dairy Cattle Monitoring, "
                       "NeurIPS 2024 Datasets and Benchmarks, doi 10.57967/hf/5965"},
    "opencows": {"url": "https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip",
                 "licence": "UK Non-Commercial Government Licence v2.0",
                 "page": "https://doi.org/10.5523/bris.10m32xl88x2b61zlkkgz3fml17",
                 "cite": "W. Andrew et al., Visual identification of individual Holstein-Friesian cattle "
                         "via deep metric learning, Computers and Electronics in Agriculture 185 (2021)"},
}
UA = {"User-Agent": "herd-extra-det/1.0"}


# ------------------------------------------------------------- remote zip

class RemoteZip:
    """A zip on an HTTP server that answers Range requests: list it, read single members."""

    def __init__(self, url, tries=6):
        self.url, self.tries = url, tries
        self._resolve()
        self.zip = zipfile.ZipFile(io.BufferedReader(_RangeFile(self), buffer_size=1 << 20))

    def _resolve(self):
        try:
            req = urllib.request.Request(self.url, method="HEAD", headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                self.size, self.direct = int(r.headers["Content-Length"]), r.url   # after redirects
        except urllib.error.HTTPError:
            # some servers (Kaggle) answer HEAD with 404: one byte by GET, the size from Content-Range
            req = urllib.request.Request(self.url, headers={**UA, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.size, self.direct = int(r.headers["Content-Range"].split("/")[1]), r.url

    def get(self, start, end):
        """Bytes start..end (inclusive). Retries; a signed redirect that ran out is followed afresh."""
        for k in range(self.tries):
            url = self.direct if k == 0 else self.url
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"no Range support (HTTP {r.status})")
                    data = r.read()
                    self.direct = r.url
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} of {end - start + 1}")
                return data
            except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
                if k == self.tries - 1:
                    raise
                time.sleep(_wait(e, k))
                print(f"\n  retry {k + 1} ({e.__class__.__name__}: {e})", flush=True)

    def names(self):
        return [i.filename for i in self.zip.infolist() if not i.is_dir()]

    def read(self, name):
        return read_member(self.get, self.zip.getinfo(name))


class LocalZip:
    """The same two calls on a zip downloaded whole."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        self.size = os.path.getsize(path)
        self._lock = __import__("threading").Lock()

    def names(self):
        return [i.filename for i in self.zip.infolist() if not i.is_dir()]

    def read(self, name):
        with self._lock:                       # one ZipFile, many threads
            return self.zip.read(name)


def download(url, path, tries=8):
    """The whole file to path, resuming a cut transfer where it stopped."""
    if os.path.exists(path):
        return path
    part = path + ".part"
    for k in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if r.status == 206 else "wb") as fh:
                total = int(r.headers.get("Content-Length", 0)) + (have if r.status == 206 else 0)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    print(f"\r  {fh.tell() / 2**20:.0f} / {total / 2**20:.0f} MB", end="", flush=True)
            print(flush=True)
            os.replace(part, path)
            return path
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            if k == tries - 1:
                raise
            print(f"\n  retry {k + 1} ({e.__class__.__name__}: {e})", flush=True)
            time.sleep(_wait(e, k))


def _wait(e, k):
    """Seconds before retry k: doubling, or what a 429 / 503 asks for (Retry-After) when it asks for more."""
    after = getattr(e, "headers", None) and e.headers.get("Retry-After")
    try:
        return max(2 ** k, min(600.0, float(after))) if after else 2 ** k
    except ValueError:
        return 2 ** k


def open_zip(url, cache_dir, name, whole_below=1 << 30):
    """A small zip is downloaded whole (some servers throttle many Range requests), a large one read in place."""
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
                size = int(r.headers["Content-Length"])
            break
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError):
            if k == 5:
                raise
            time.sleep(2 ** k)
    if size <= whole_below:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[extra] {name}: downloading the zip whole ({size / 2**20:.0f} MB)", flush=True)
        return LocalZip(download(url, os.path.join(cache_dir, name + ".zip")))
    return RemoteZip(url)


class _RangeFile(io.RawIOBase):
    def __init__(self, rz):
        self.rz, self.pos = rz, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.rz.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.rz.size:
            return 0
        data = self.rz.get(self.pos, min(self.rz.size, self.pos + len(b)) - 1)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def read_member(get, info):
    """One member's bytes with one request: local header + data (the local extra field is guessed generously)."""
    off, n = info.header_offset, info.compress_size
    blob = get(off, off + 30 + 1024 + 1024 + n - 1) if n else get(off, off + 30 + 2048 - 1)
    if blob[:4] != b"PK\x03\x04":
        raise IOError(f"{info.filename}: no local header")
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    if start + n > len(blob):
        blob += get(off + len(blob), off + start + n - 1)
    raw = blob[start:start + n]
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise IOError(f"{info.filename}: compression {info.compress_type} not supported")
    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise IOError(f"{info.filename}: CRC mismatch")
    return data


# --------------------------------------------------------- label formats

def yolo_boxes(text):
    """YOLO / darknet lines "class cx cy w h" (0-1) -> [[x1, y1, x2, y2], ...] clipped to 0-1."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cx, cy, w, h = map(float, p[1:])
        b = [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]
        if b[2] - b[0] > 1e-3 and b[3] - b[1] > 1e-3:
            out.append([round(v, 5) for v in b])
    return out


def plan_zenodo(names):
    """-> [(split, out_name, image member, label member)]; valid = every 5th of the 11 source videos."""
    imgs = sorted(n for n in names if "/images/" in n and n.lower().endswith((".jpg", ".jpeg", ".png")))
    video = lambda n: re.sub(r"_frame.*", "", os.path.basename(n))
    videos = sorted({video(n) for n in imgs})
    held = {v for i, v in enumerate(videos) if i % 5 == 4}
    out = []
    for n in imgs:
        lab = n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if lab in names:
            out.append(("valid" if video(n) in held else "train", os.path.basename(n), n, lab))
    return out


def plan_mmcows(names, every=8, held_every=6):
    """-> plan; every EVERY-th 15 s frame per camera, valid = every HELD_EVERY-th hour of the day."""
    have = set(names)
    per_cam = {}
    for n in names:
        m = re.match(r".*/images/(\d+)/(cam_\d+)/(\d+)_[\d-]+\.jpg$", n)
        if m:
            lab = n.replace("/images/", "/labels/combined/").rsplit(".", 1)[0] + ".txt"
            if lab in have:
                per_cam.setdefault(m.group(2), []).append((int(m.group(3)), n, lab))
    out = []
    for cam, rows in sorted(per_cam.items()):
        for ts, n, lab in sorted(rows)[::every]:
            split = "valid" if (ts // 3600) % held_every == held_every - 1 else "train"
            out.append((split, f"{cam}_{os.path.basename(n)}", n, lab))
    return out


def plan_opencows(names, folds):
    """-> plan; the real frames only (<= 003707.jpg), valid = the authors' fold 0 test."""
    test = set(folds.get("0", {}).get("test", []))
    out = []
    for n in sorted(names):
        m = re.match(r".*/detection_and_localisation/images/(\d+)\.jpg$", n)
        if m and int(m.group(1)) <= 3707:
            lab = n.replace("/images/", "/labels-darknet/").rsplit(".", 1)[0] + ".txt"
            if lab in names:
                b = os.path.basename(n)
                out.append(("valid" if b in test else "train", b, n, lab))
    return out


# ------------------------------------------------------------------ fetch

def shrink(data, max_side):
    """Image bytes -> JPEG bytes with the longer side at most max_side (as is when already small)."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        if max(w, h) <= max_side and im.format == "JPEG":
            return data, (w, h)
        im = im.convert("RGB")
        s = min(1.0, max_side / max(w, h))
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        return buf.getvalue(), im.size


def write_coco(folder, items):
    """items [(file name, (w, h), boxes 0-1)] -> folder/_annotations.coco.json (one class: cow)."""
    images, anns = [], []
    for i, (name, (w, h), boxes) in enumerate(items, 1):
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for b in boxes:
            x, y, bw, bh = b[0] * w, b[1] * h, (b[2] - b[0]) * w, (b[3] - b[1]) * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [x, y, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
    with open(os.path.join(folder, "_annotations.coco.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "cow", "supercategory": "none"}]}, fh)
    return len(images), len(anns)


def fetch(name, dest, every=8, max_side=1920, workers=8, fresh=False, limit=0, splits=("train", "valid")):
    out = os.path.join(dest, name)
    done = os.path.join(out, "source.json")
    if os.path.exists(done) and not fresh:
        print(f"[extra] {name}: done already ({out})")
        return out
    src = SOURCES[name]
    print(f"[extra] {name}: reading the zip's index ({src['url']})", flush=True)
    rz = open_zip(src["url"], os.path.join(dest, "_zips"), name)
    names = rz.names()
    if name == "zenodo":
        plan = plan_zenodo(names)
    elif name == "mmcows":
        plan = plan_mmcows(names, every)
    else:
        cv = next((n for n in names if n.endswith("10-fold-CV.json")), None)
        plan = plan_opencows(set(names), json.loads(rz.read(cv)) if cv else {})
    if set(splits) != {"train", "valid"}:
        # e.g. only valid, for a test pod: done.json is then written for what was fetched only
        plan = [p for p in plan if p[0] in splits]
        done = os.path.join(out, f"source_{'_'.join(splits)}.json")
        if os.path.exists(done) and not fresh:
            print(f"[extra] {name} ({', '.join(splits)}): done already ({out})")
            return out
    if limit and len(plan) > limit:             # a trial: frames spread over the whole plan
        plan = plan[::len(plan) // limit][:limit]
    for split in ("train", "valid"):
        os.makedirs(os.path.join(out, split), exist_ok=True)
    print(f"[extra] {name}: {len(plan)} frames to fetch ({sum(p[0] == 'valid' for p in plan)} valid), "
          f"zip {rz.size / 2**30:.1f} GB", flush=True)

    def one(p):
        split, fname, img, lab = p
        boxes = yolo_boxes(rz.read(lab).decode("utf-8", "replace"))
        path = os.path.join(out, split, os.path.splitext(fname)[0] + ".jpg")
        if os.path.exists(path) and not fresh:
            from PIL import Image
            with Image.open(path) as im:
                return split, os.path.basename(path), im.size, boxes
        data, size = shrink(rz.read(img), max_side)
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
        return split, os.path.basename(path), size, boxes

    items = {"train": [], "valid": []}
    empty = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, (split, fname, size, boxes) in enumerate(ex.map(one, plan), 1):
            # a frame with no box: no cow, or cows nobody boxed - left out, not taught as background
            if boxes:
                items[split].append((fname, size, boxes))
            else:
                empty += 1
            if k % 25 == 0 or k == len(plan):
                print(f"\r  {k}/{len(plan)} frames", end="", flush=True)
    print(flush=True)
    counts = {}
    for split, it in items.items():
        if split in splits:                    # a split not fetched keeps what an earlier fetch wrote
            counts[split] = dict(zip(("images", "cows"), write_coco(os.path.join(out, split), it)))
    meta = {"name": name, **src, "every": every if name == "mmcows" else None, "max_side": max_side,
            "frames_without_boxes_left_out": empty, **counts}
    with open(done, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[extra] {name}: " + ", ".join(f"{k} {v['images']} frames / {v['cows']} cows" for k, v in counts.items())
          + f" -> {out}  ({src['licence']})")
    return out


def cmd_fetch(args):
    for name in _sources(args.sources):
        fetch(name, args.dest, args.every, args.max_side, args.workers, args.fresh, args.limit,
              tuple(x.strip() for x in args.splits.split(",") if x.strip()))


CBVD_URL = "https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cbvd-5cow-behavior-video-dataset"


def cmd_cbvd_videos(args):
    """Only CBVD-5's val clips (and the annotation files that list them) from its 11.6 GB zip: the
    stress test plays them as cameras; nothing is scored against their labels."""
    dest = args.data
    rz = RemoteZip(CBVD_URL)
    names = rz.names()
    csvs = [n for n in names if re.match(r"(.*/)?annotations/ava_(val|train)_v2\.1\.csv$", n)
            and "mini" not in n]
    os.makedirs(os.path.join(dest, "annotations"), exist_ok=True)
    for n in csvs:
        with open(os.path.join(dest, "annotations", os.path.basename(n)), "wb") as fh:
            fh.write(rz.read(n))
    val = {line.split(",")[0] for line in open(os.path.join(dest, "annotations", "ava_val_v2.1.csv"))
           if line.strip()}
    vids = [n for n in names if re.match(r"(.*/)?videos/videos/[^/]+\.mp4$", n)
            and os.path.splitext(os.path.basename(n))[0] in val]
    out = os.path.join(dest, "videos", "videos")
    os.makedirs(out, exist_ok=True)

    def one(n):
        path = os.path.join(out, os.path.basename(n))
        if not os.path.exists(path):
            with open(path + ".part", "wb") as fh:
                fh.write(rz.read(n))
            os.replace(path + ".part", path)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(one, vids), 1):
            print(f"\r  {k}/{len(vids)} val clips", end="", flush=True)
    print(flush=True)
    print(f"[cbvd-videos] {len(vids)} val clips -> {out} (annotation files in {dest}/annotations)")


def _sources(s):
    names = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in names if x not in SOURCES]
    if bad:
        sys.exit(f"unknown source(s) {bad}; known: {', '.join(SOURCES)}")
    return names


# ------------------------------------------------------------------- eval

def load_coco(folder):
    """folder/_annotations.coco.json -> [{"video_id": file, "timestamp": 0, "boxes": [[x1, y1, x2, y2] 0-1]}]."""
    with open(os.path.join(folder, "_annotations.coco.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    by = {im["id"]: {"video_id": im["file_name"], "timestamp": 0, "boxes": [], "wh": (im["width"], im["height"])}
          for im in d["images"]}
    for a in d["annotations"]:
        f = by[a["image_id"]]
        w, h = f["wh"]
        x, y, bw, bh = a["bbox"]
        f["boxes"].append([x / w, y / h, (x + bw) / w, (y + bh) / h])
    return list(by.values())


def cmd_eval(args):
    """Each detector on each source's valid frames: recall / precision at its own threshold (IoU 0.5 and
    0.3) and the best F1 over thresholds - the boxes as the detector draws them (no CBVD-5 calibration,
    unless --calib 1)."""
    from PIL import Image
    import detector as det_mod
    report = {}
    for best in args.detector:
        live = det_mod.Live(best, threshold=det_mod.KEEP_SCORE, calib=bool(args.calib))
        with open(os.path.join(best, "det_train_meta.json"), encoding="utf-8") as fh:
            own = json.load(fh)["threshold"]
        tag = os.path.basename(os.path.dirname(os.path.abspath(best)))
        report[tag] = {"detector": os.path.abspath(best), "threshold": own}
        for name in _sources(args.sources):
            folder = os.path.join(args.data, name, "valid")
            if not os.path.exists(os.path.join(folder, "_annotations.coco.json")):
                print(f"[extra-eval] {name}: not fetched, skipped")
                continue
            frames = load_coco(folder)
            if args.max_frames and len(frames) > args.max_frames:
                frames = random.Random(0).sample(frames, args.max_frames)
            dets = {}
            for i in range(0, len(frames), 8):
                chunk = frames[i:i + 8]
                imgs = [Image.open(os.path.join(folder, f["video_id"])).convert("RGB") for f in chunk]
                for f, found in zip(chunk, live.many(imgs)):
                    dets[(f["video_id"], 0)] = [c["bbox"] + [c["det_score"]] for c in found]
                print(f"\r  {tag} on {name}: {min(i + 8, len(frames))}/{len(frames)}", end="", flush=True)
            print(flush=True)
            at05, at03 = det_mod.prf(dets, frames, own, 0.5), det_mod.prf(dets, frames, own, 0.3)
            thr, best_m = det_mod.best_threshold(dets, frames, "f1")
            report[tag][name] = {"frames": len(frames), "cows": sum(len(f["boxes"]) for f in frames),
                                 "iou0.5": at05, "iou0.3": at03, "best_f1_threshold": thr, "best_f1": best_m}
    print(f"\n{'detector':34s} {'source':9s} {'frames':>6s}  {'recall':>7s} {'prec':>7s}  "
          f"{'r@0.3':>7s} {'p@0.3':>7s}  best F1 (thr)")
    for tag, r in report.items():
        for name, m in r.items():
            if isinstance(m, dict) and "iou0.5" in m:
                a, b = m["iou0.5"], m["iou0.3"]
                print(f"{tag:34s} {name:9s} {m['frames']:6d}  {a['recall']:7.1%} {a['precision']:7.1%}  "
                      f"{b['recall']:7.1%} {b['precision']:7.1%}  {m['best_f1']['f1']:.3f} ({m['best_f1_threshold']})")
    out = args.out or os.path.join(args.data, "eval_extra.json")
    old = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            old = json.load(fh)
    old.update(report)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(old, fh, indent=2)
    print(f"-> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("fetch", "eval", "cbvd-videos"))
    p.add_argument("--dest", "--data", dest="data", default="/workspace/extra_det")
    p.add_argument("--sources", default="zenodo,mmcows,opencows")
    p.add_argument("--every", type=int, default=8, help="mmcows: every N-th 15 s frame per camera")
    p.add_argument("--max-side", type=int, default=1920)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="fetch: a trial with this many frames per source")
    p.add_argument("--splits", default="train,valid", help="fetch: valid only for a test pod")
    p.add_argument("--detector", action="append", default=[], help="eval: a best/ folder (several allowed)")
    p.add_argument("--calib", type=int, default=0, help="eval: 1 applies the detector's CBVD-5 box calibration")
    p.add_argument("--max-frames", type=int, default=600, help="eval: at most this many valid frames per source")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.stage == "fetch":
        args.dest = args.data
        cmd_fetch(args)
    elif args.stage == "cbvd-videos":
        cmd_cbvd_videos(args)
    else:
        if not args.detector:
            sys.exit("eval needs --detector <best/>")
        cmd_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
