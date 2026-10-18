"""Scene tokens: a camera's whole frame as a small grid of DINOv2 patch descriptions.

DINOv2-S (the same frozen encoder as the crops) reads the frame at 672 x 420 (48 x 30
patches of 14 px), and the patches are averaged into a 12 x 12 grid of 384 numbers. The
association transformer gets 8 x 8 of them per camera (pooled from the 12 x 12, or from
the part of it a virtual camera sees), refreshed every minute: the barn hardly changes.
"""

from __future__ import annotations

import numpy as np

SCENE_SIZE = (672, 420)
GRID = 12
OUT = 8
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def scene_grid(enc, imgs, size=SCENE_SIZE, grid=GRID):
    """PIL images -> (n, grid * grid, hidden) float16, from enc.backbone (model.FrameEncoder)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if not imgs:
        return np.zeros((0, grid * grid, enc.hidden), np.float16)
    dev = next(enc.parameters()).device
    x = np.stack([(np.asarray(im.convert("RGB").resize(size, Image.BILINEAR), np.float32) / 255.0 - MEAN) / STD
                  for im in imgs]).transpose(0, 3, 1, 2)
    ps = enc.backbone.config.patch_size
    gh, gw = size[1] // ps, size[0] // ps
    out = []
    with torch.no_grad():
        for s in range(0, len(x), 8):
            xb = torch.from_numpy(x[s:s + 8]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                tok = enc.backbone(pixel_values=xb).last_hidden_state
            p = tok[:, -gh * gw:].float().reshape(len(xb), gh, gw, -1).permute(0, 3, 1, 2)
            g = F.adaptive_avg_pool2d(p, grid).flatten(2).transpose(1, 2)
            out.append(g.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def sub_grid(g, region=(0.0, 0.0, 1.0, 1.0), out=OUT, grid=GRID):
    """(grid * grid, C) of a whole frame -> (out * out, C) of the part `region` (x0, y0, x1, y1, 0-1)."""
    import torch
    import torch.nn.functional as F
    C = g.shape[-1]
    t = torch.from_numpy(np.asarray(g, np.float32)).reshape(1, grid, grid, C).permute(0, 3, 1, 2)
    up = F.interpolate(t, size=(grid * 4, grid * 4), mode="bilinear", align_corners=False)
    x0, y0, x1, y1 = region
    n = grid * 4
    c0, c1 = int(np.floor(x0 * n)), max(int(np.floor(x0 * n)) + 1, int(np.ceil(x1 * n)))
    r0, r1 = int(np.floor(y0 * n)), max(int(np.floor(y0 * n)) + 1, int(np.ceil(y1 * n)))
    part = up[:, :, r0:r1, c0:c1]
    o = F.adaptive_avg_pool2d(part, out).flatten(2).transpose(1, 2)[0]
    return o.numpy()


def cell_centres(out=OUT, region=(0.0, 0.0, 1.0, 1.0)):
    """The centres of an out x out grid over `region`, in the source frame's 0-1 coordinates."""
    x0, y0, x1, y1 = region
    c = (np.arange(out) + 0.5) / out
    xs, ys = x0 + c * (x1 - x0), y0 + c * (y1 - y0)
    return np.array([(x, y) for y in ys for x in xs], np.float32)
