"""Rhythm of motion in a burst: what rumination looks like.

A ruminating cow chews her cud about once a second (40-70 chews a minute),
in long regular runs; a resting cow barely moves, a feeding one chews faster
and moves her head, a walking one moves everywhere. One still frame cannot
tell these apart and a frame vector (DINOv2) mostly ignores a few pixels of
jaw; the spectrum of the pixels over the 7 s burst can.

From the burst's own crops (the 175 the encoder gets - nothing new to cut):
grey, 56 x 56, the slow drift removed per pixel, then the power spectrum per
pixel over time. Per cell of a 4 x 4 grid (the head may be anywhere in the
crop) and per band around the chewing rate:
    log mean power                   how much it moves at that rate
    90th percentile of band share    how much of a pixel's motion is that rate
                                     (rhythm, not just movement; contrast-free)
plus the max over cells of the band shares, the overall motion and how peaked
the spectrum of the most rhythmic pixels is. Same code in training
(cbvd_bursts.py motion) and in the barn (pipeline.py).
"""

from __future__ import annotations

import numpy as np

BANDS = ((0.4, 0.7), (0.7, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 2.5), (2.5, 5.0))   # Hz
RANGE = (0.3, 6.0)          # the motion the shares are taken of
SIDE, GRID = 56, 4
MIN_SECONDS = 3.0
DIM = GRID * GRID * 2 * len(BANDS) + len(BANDS) + 3


def grey_small(crops):
    """(T, H, W, 3) uint8 -> (T, SIDE, SIDE) float32, by block means (H, W multiples of SIDE)."""
    t, h, w, _ = crops.shape
    fy, fx = h // SIDE, w // SIDE
    s = crops[:, :fy * SIDE, :fx * SIDE].reshape(t, SIDE, fy, SIDE, fx, 3).sum((2, 4, 5), dtype=np.uint32)
    return s.astype(np.float32) / (fy * fx * 3)


def motion_features(crops, times=None, valid=None, fps=25.0):
    """crops (T, H, W, 3) uint8 of one cow, times (T,) s, valid (T,) bool ->
    (DIM,) float32 and whether it was computed (False: under MIN_SECONDS seen)."""
    crops = np.asarray(crops)
    t_all = np.arange(len(crops)) / fps if times is None else np.asarray(times, np.float64)
    keep = np.ones(len(crops), bool) if valid is None else np.asarray(valid, bool)
    if keep.sum() < 2:
        return np.zeros(DIM, np.float32), False
    idx = np.flatnonzero(keep)
    idx = np.arange(idx[0], idx[-1] + 1)               # the seen stretch; a hidden frame or two inside stays
    span = t_all[idx[-1]] - t_all[idx[0]]
    if span < MIN_SECONDS:
        return np.zeros(DIM, np.float32), False
    rate = (len(idx) - 1) / span                        # the camera's real frame rate
    x = grey_small(crops[idx])                          # (T, S, S)
    n = len(x)
    tt = np.linspace(-1, 1, n, dtype=np.float32)[:, None, None]
    x = x - x.mean(0)
    x = x - tt * (x * tt).sum(0) / (tt * tt).sum()      # drift: light, a slow turn of the box
    x = x * np.hanning(n).astype(np.float32)[:, None, None]
    power = np.abs(np.fft.rfft(x, axis=0)) ** 2         # (F, S, S)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    inr = (freqs >= RANGE[0]) & (freqs < min(RANGE[1], rate / 2))
    total = power[inr].sum(0) + 1e-6                    # (S, S)
    band_power = np.stack([power[(freqs >= lo) & (freqs < hi)].sum(0) for lo, hi in BANDS])   # (B, S, S)
    share = band_power / total
    c = SIDE // GRID
    cells = lambda a: a.reshape(a.shape[0], GRID, c, GRID, c).transpose(0, 1, 3, 2, 4).reshape(a.shape[0], GRID * GRID, c * c)
    log_pow = np.log1p(cells(band_power).mean(-1) / n)                 # (B, G*G)
    k = int(round(0.9 * (c * c - 1)))
    share_p90 = np.partition(cells(share), k, axis=-1)[..., k]          # (B, G*G), 90th percentile
    # how peaked the spectrum of the most rhythmic pixels is (a chewing rate stands out)
    best = np.argsort(share[1:4].sum(0).ravel())[-max(1, SIDE * SIDE // 20):]
    spec = power[inr].reshape(int(inr.sum()), -1)[:, best].mean(1)
    peak = float(spec.max() / (np.median(spec) + 1e-6)) if len(spec) else 0.0
    out = np.concatenate([log_pow.T.ravel(), share_p90.T.ravel(), share_p90.max(1),
                          [np.log1p(total.mean() / n), np.log1p(peak), span / 7.0]]).astype(np.float32)
    return out, True
