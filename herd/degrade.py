"""Bad frames on purpose: what the quality head learns to weigh down.

The brief asks the burst model to know a good burst (the cow's coat and gait
in plain view) from a bad one (the cow in mud, behind another cow). CBVD-5
has no such labels, so they are made: stretches of a burst's crops are
spoilt the way a barn spoils them, and the model is told which frames were.

    occlusion   another cow's crop over 35-70 % of the frame, from one side,
                for the whole stretch (a cow walking past, a cow lying in front)
    mud         brown blotches with grain over the coat (the pattern hidden)
    blur        out of focus / motion (Gaussian, sigma 3-6 px at 224)
    dark        night, a dim corner: brightness down to 15-35 %, sensor noise

A stretch is 0.4-2.4 s (10-60 frames at 25 fps); a burst gets 1-3 of them,
about a fifth to three fifths of its frames. Each frame keeps its clean
twin, so training can mix clean and spoilt frames per view (train.py).
"""

from __future__ import annotations

import numpy as np

KINDS = ("none", "occlusion", "mud", "blur", "dark")


def plan(n_frames, rng, frac=(0.2, 0.6), seg=(10, 60), n_seg=(1, 3)):
    """-> [(start, stop, kind)] stretches to spoil, not overlapping."""
    want = rng.uniform(*frac) * n_frames
    out, used = [], np.zeros(n_frames, bool)
    for _ in range(rng.randint(*n_seg)):
        length = int(min(rng.randint(*seg), max(1, want - used.sum()), n_frames))
        for _ in range(10):
            s = rng.randint(0, max(0, n_frames - length))
            if not used[s:s + length].any():
                used[s:s + length] = True
                out.append((s, s + length, rng.choice(KINDS[1:])))
                break
        if used.sum() >= want:
            break
    return sorted(out)


def occlude(img, other, side, cover):
    """Another cow's crop over `cover` of the frame from one side."""
    h, w = img.shape[:2]
    out = img.copy()
    if side in ("left", "right"):
        k = int(round(w * cover))
        sl = (slice(None), slice(0, k)) if side == "left" else (slice(None), slice(w - k, w))
    else:
        k = int(round(h * cover))
        sl = (slice(0, k), slice(None)) if side == "top" else (slice(h - k, h), slice(None))
    out[sl] = other[sl]
    return out


def mud(img, blobs, rng_np):
    """Brown blotches with grain; `blobs` [(cx, cy, rx, ry, alpha)] in 0-1."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    x = img.astype(np.float32)
    brown = np.array([88, 66, 42], np.float32)
    for cx, cy, rx, ry, a in blobs:
        m = (((xx / w - cx) / rx) ** 2 + ((yy / h - cy) / ry) ** 2) <= 1.0
        grain = rng_np.normal(0, 12, (int(m.sum()), 3)).astype(np.float32)
        x[m] = (1 - a) * x[m] + a * (brown + grain)
    return np.clip(x, 0, 255).astype(np.uint8)


def blur(img, sigma):
    from PIL import Image, ImageFilter
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(sigma)))


def dark(img, level, rng_np):
    x = img.astype(np.float32) * level + rng_np.normal(0, 6, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def spoil(crops, stretches, others, rng):
    """crops (T, H, W, 3) uint8 of one cow's burst; others: crops (T, H, W, 3) of
    other cows of the same clip (for occlusion; may be empty). -> (frame
    indices, kind per index, spoilt crops). One setting per stretch: the same
    cow stays in front, the same mud stays on."""
    rng_np = np.random.default_rng(rng.randrange(2 ** 31))
    idx, kinds, out = [], [], []
    for s, e, kind in stretches:
        if kind == "occlusion" and not len(others):
            kind = "mud"                                  # nothing to put in front: mud instead
        if kind == "occlusion":
            other = others[rng.randrange(len(others))]
            side = rng.choice(("left", "right", "top", "bottom"))
            cover = rng.uniform(0.35, 0.7)
        elif kind == "mud":
            blobs = [(rng.uniform(0.2, 0.8), rng.uniform(0.3, 0.8), rng.uniform(0.12, 0.3),
                      rng.uniform(0.1, 0.25), rng.uniform(0.6, 0.9)) for _ in range(rng.randint(2, 5))]
        elif kind == "blur":
            sigma = rng.uniform(3, 6)
        else:
            level = rng.uniform(0.15, 0.35)
        for i in range(s, e):
            img = crops[i]
            if kind == "occlusion":
                img = occlude(img, other[i % len(other)], side, cover)
            elif kind == "mud":
                img = mud(img, blobs, rng_np)
            elif kind == "blur":
                img = blur(img, sigma)
            else:
                img = dark(img, level, rng_np)
            idx.append(i)
            kinds.append(KINDS.index(kind))
            out.append(img)
    if not out:
        return np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros((0,) + crops.shape[1:], np.uint8)
    return np.array(idx, np.int16), np.array(kinds, np.int8), np.stack(out)
