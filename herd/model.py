"""The two transformers and their heads.

    crop (224x224)  --DINOv2-S, frozen-->  frame vector: CLS + a grid x grid
                                           pool of the patch tokens
    1 fps path      frame vector --FrameHeads-->        posture, activity
    burst path      175 frame vectors + their times
                    --TemporalModel (2-4 layers)-->     per-frame outputs
                       quality per frame  -> weights
                       ID per frame       -> fingerprint = weighted average
                       pooled             -> posture, activity, rumination, lameness
                    + motion rhythm (motion.py), if the model has it -> rumination, activity
                       pooled             -> burst quality (0-1: will this burst
                                             identify the cow?), if the model has it

The fingerprint is the brief's "weighted average of all the fingerprints from
the burst", the weights the quality head's: a frame where the cow is hidden
or blurred counts for little. With train.py --quality the quality head is
also taught directly (frames spoilt on purpose by degrade.py must score low)
and a burst-level head learns whether the burst as a whole will identify the
cow - the brief's "good burst / bad burst". The grid pool keeps some of where things are
in the crop (head vs body), which a jaw movement needs and a CLS token alone
would average away.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ID_DIM = 512


def box_pos(box):
    """[x1, y1, x2, y2] (0-1) -> [centre x, centre y, width, height]."""
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameEncoder(nn.Module):
    """DINOv2 (Apache-2.0) as a frozen feature extractor: uint8 crops -> vectors."""

    def __init__(self, name="facebook/dinov2-small", grid=2):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(name)
        self.backbone.requires_grad_(False)
        self.grid = grid
        self.hidden = self.backbone.config.hidden_size
        self.out_dim = self.hidden * (1 + grid * grid)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, crops_uint8):
        """(B, H, W, 3) uint8 -> (B, out_dim) float."""
        x = crops_uint8.permute(0, 3, 1, 2).float().div(255.0)
        x = (x - self.mean) / self.std
        tokens = self.backbone(pixel_values=x).last_hidden_state      # (B, 1+N(+reg), C)
        side = x.shape[-1] // self.backbone.config.patch_size
        patches = tokens[:, -side * side:]                              # registers, if any, sit in between
        grid = patches.reshape(x.shape[0], side, side, -1).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid, self.grid).flatten(2).transpose(1, 2)   # (B, g*g, C)
        return torch.cat([tokens[:, 0:1], pooled], 1).flatten(1)


def time_embedding(t, dim):
    """Sinusoidal embedding of real time in seconds (not of the frame index:
    speed-jittered and frame-dropped bursts keep their true spacing)."""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half) * 2 * math.pi
    ang = t.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class FrameHeads(nn.Module):
    """Posture and activity from one frame vector: the once-a-second path,
    which has no burst to look at."""

    def __init__(self, in_dim, hidden=256, n_posture=2, n_activity=3, use_pos=False):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1))
        # Where the cow is in the frame (box centre and size). Cameras do not
        # move, so "at the feed barrier" is a place in the picture - context a
        # tight crop does not have and a wide crop pays for with the neighbours.
        self.pos = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, hidden)) if use_pos else None
        self.posture = nn.Linear(hidden, n_posture)
        self.activity = nn.Linear(hidden, n_activity)

    def forward(self, x, pos=None):
        h = self.body(x)
        if self.pos is not None and pos is not None:
            h = h + self.pos(pos)
        return {"posture": self.posture(h), "activity": self.activity(h)}


class TemporalModel(nn.Module):
    def __init__(self, in_dim, d=256, layers=3, heads=4, dropout=0.1, n_posture=2, n_activity=3, use_pos=False,
                 motion_dim=0, burst_quality=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.pos = nn.Linear(4, d) if use_pos else None
        self.time = nn.Linear(d, d)
        self.d = d
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.quality = nn.Linear(d, 1)
        self.id = nn.Linear(d, ID_DIM)
        self.posture = nn.Linear(d, n_posture)
        self.activity = nn.Linear(d, n_activity)
        self.lameness = nn.Linear(d, 1)
        # Good burst or bad: will this burst's fingerprint find the right cow?
        self.burst_quality = nn.Linear(d, 1) if burst_quality else None
        # The rhythm of the pixels over the burst (motion.py): chewing at ~1 Hz
        # is what rumination is, and what frame vectors average away. It goes
        # to rumination and activity (feeding chews faster, with head moves).
        self.motion_dim = motion_dim
        self.motion = (nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, d), nn.GELU(),
                                     nn.Dropout(0.2), nn.Linear(d, d)) if motion_dim else None)
        self.rumination = (nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
                           if motion_dim else nn.Linear(d, 1))
        if motion_dim:
            self.activity_motion = nn.Linear(d, n_activity, bias=False)

    def forward(self, x, t, valid, pos=None, motion=None):
        """x (B, T, in_dim) frame vectors, t (B, T) seconds, valid (B, T) bool,
        pos (B, 4) or (B, T, 4) box centre and size, used if the model has it;
        motion (B, motion_dim) the burst's rhythm features, if it has that."""
        h = self.proj(x) + self.time(time_embedding(t, self.d))
        if self.pos is not None and pos is not None:
            h = h + self.pos(pos if pos.dim() == 3 else pos.unsqueeze(1))
        h = self.norm(self.encoder(h, src_key_padding_mask=~valid))
        q = self.quality(h).squeeze(-1).masked_fill(~valid, -1e4)
        w = q.softmax(-1)                                          # quality weights over frames
        ids = F.normalize(self.id(h), dim=-1)
        fingerprint = F.normalize((w.unsqueeze(-1) * ids).sum(1), dim=-1)
        pooled = (w.unsqueeze(-1) * h).sum(1)
        activity = self.activity(pooled)
        if self.motion is not None:
            if motion is None:                                   # not computed (cow seen too briefly)
                motion = pooled.new_zeros(len(pooled), self.motion_dim)
            m = self.motion(motion)
            rumination = self.rumination(torch.cat([pooled, m], -1))
            activity = activity + self.activity_motion(m)
        else:
            rumination = self.rumination(pooled)
        return {"fingerprint": fingerprint, "quality": q, "weights": w, "frame_ids": ids,
                "burst_quality": None if self.burst_quality is None else self.burst_quality(pooled).squeeze(-1),
                "posture": self.posture(pooled), "activity": activity,
                "rumination": rumination.squeeze(-1),
                "lameness": self.lameness(pooled).squeeze(-1)}


class HerdModel(nn.Module):
    def __init__(self, in_dim, use_pos=False, **kw):
        super().__init__()
        self.in_dim = in_dim
        self.kw = dict(kw, use_pos=use_pos)
        self.frame = FrameHeads(in_dim, use_pos=use_pos)
        self.temporal = TemporalModel(in_dim, use_pos=use_pos, **kw)

    @property
    def motion_dim(self):
        return self.kw.get("motion_dim", 0)

    @property
    def has_burst_quality(self):
        return bool(self.kw.get("burst_quality", False))

    def save(self, path, extra=None):
        torch.save({"in_dim": self.in_dim, "kw": self.kw, "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(ck["in_dim"], **ck["kw"])
        m.load_state_dict(ck["state"])
        return m, ck


# ------------------------------------------------------------------ losses

def supcon(fingerprints, labels, temperature=0.1):
    """Supervised contrastive loss: bursts of the same cow together, all others
    apart. Rows whose label is -1 are skipped; a label with one row only has no
    positive and contributes nothing."""
    keep = labels >= 0
    z, y = fingerprints[keep], labels[keep]
    if len(y) < 2:
        return fingerprints.sum() * 0
    sim = z @ z.t() / temperature
    eye = torch.eye(len(y), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    has = pos.any(1)
    if not has.any():
        return fingerprints.sum() * 0
    logp = sim.log_softmax(1)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def masked_ce(logits, target):
    """Cross-entropy over the rows that have a label (target >= 0); the head of a
    missing label is switched off for that example."""
    keep = target >= 0
    return F.cross_entropy(logits[keep], target[keep]) if keep.any() else logits.sum() * 0


def masked_bce(logits, target):
    keep = target >= 0
    return (F.binary_cross_entropy_with_logits(logits[keep], target[keep].float())
            if keep.any() else logits.sum() * 0)
