"""Turning a keyframe plus one bounding box into what the model actually sees."""

from __future__ import annotations

import base64
import io
import threading

from PIL import Image, ImageDraw

# Lime green: no cow, stall, feed or barn floor in this dataset is anywhere
# near it, so the outline can never be mistaken for part of the scene.
OUTLINE_RGB = (0, 255, 0)


def _to_pixels(box_xyxy, size):
    w, h = size
    x1, y1, x2, y2 = box_xyxy
    return [x1 * w, y1 * h, x2 * w, y2 * h]


_frame_cache = {}
_cache_lock = threading.Lock()


def load_frames(video_path: str, timestamp: int, count: int, span: float):
    """`count` frames evenly spread over `span` seconds, centred on `timestamp`.

    Only needed for rumination, and only because of what rumination is: chewing
    cud. On a single still frame it is not observable at all - a ruminating cow
    and an idle one are the same picture - so grading a still-image answer
    against that label measures nothing. A short window of frames at least puts
    the jaw movement in front of the model.

    Verified against this dataset: the keyframe labelframes/<id>_<t>.jpg is
    exactly frame t*25 of <id>.mp4 (25 fps, 10 s clips), so the annotation and
    these frames refer to the same moment.

    Caveat that belongs in the report, not in a comment only: the bounding box
    is annotated on the keyframe alone. Over a +-1 s window a moving cow drifts
    out of its own outline, which is the reason the default span is small.
    """
    key = (video_path, timestamp, count, span)
    with _cache_lock:
        hit = _frame_cache.get(key)
    if hit is not None:
        return hit

    import cv2  # optional: only a multi-frame run needs a video decoder

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count == 1:
            offsets = [0.0]
        else:
            step = span / (count - 1)
            offsets = [-span / 2 + i * step for i in range(count)]
        frames = []
        for off in offsets:
            index = int(round((timestamp + off) * fps))
            index = max(0, min(total - 1, index) if total else max(0, index))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"decoded no frames from {video_path} around t={timestamp}s")
    with _cache_lock:
        _frame_cache[key] = frames
    return frames


def render(frame_path, box_xyxy, mode: str = "marked",
           max_width: int = 1280, margin: float = 0.3, min_width: int = 0):
    """Render one example.

    marked - the whole frame with the target cow outlined. Keeps the scene
             context, which is the only thing separating drinking from feeding:
             both are "head down", and what disambiguates them is whether the
             head is over a water trough or the feed barrier.
    crop   - the box plus a margin. Bigger cow, no context. Useful as a
             contrast run to tell "cannot see the cow" apart from
             "cannot read the scene".
    plain  - the whole frame, nothing drawn: for finding the cows (detect.py),
             where no box is given.
    """
    # Either a path to a keyframe on disk or an already-decoded video frame.
    img = (frame_path if isinstance(frame_path, Image.Image)
           else Image.open(frame_path)).convert("RGB")

    if mode == "crop":
        x1, y1, x2, y2 = _to_pixels(box_xyxy, img.size)
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - bw * margin)
        y1 = max(0, y1 - bh * margin)
        x2 = min(img.width, x2 + bw * margin)
        y2 = min(img.height, y2 + bh * margin)
        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
    elif mode == "plain":
        pass
    elif mode == "marked":
        draw = ImageDraw.Draw(img)
        # Scale the stroke with the frame so it survives the downscale below.
        width = max(3, round(img.width / 320))
        draw.rectangle(_to_pixels(box_xyxy, img.size), outline=OUTLINE_RGB, width=width)
    else:
        raise ValueError(f"unknown render mode: {mode!r}")

    if img.width > max_width:
        height = round(img.height * max_width / img.width)
        img = img.resize((max_width, height), Image.LANCZOS)
    elif min_width and img.width < min_width:
        # Upscaling adds no information, and that is not the point. A crop of a
        # distant cow comes out ~190x110 px, which the vision encoder turns into
        # a few dozen patches - too coarse to carry a jaw. Enlarging spends more
        # patches on the same pixels, which is the only way to ask the model to
        # look closely at something small.
        height = round(img.height * min_width / img.width)
        img = img.resize((min_width, height), Image.LANCZOS)
    return img


def to_data_url(img, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
