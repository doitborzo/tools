#!/usr/bin/env python3
"""Version 14 (F): OpenCowID's identification model in place of the ID transformer.

OpenCowID (Prabhune and Kim, WACV 2026; github.com/neis-lab/OpenCowID, MIT): a ResNet-50 with
ImageNet weights and a 128-number projection, unit length, trained by metric learning around
virtual class centroids - first on synthetic Holstein coats (random blobs, no real cow: "zero-
shot"), here then on the real cattle version 14 (D) trained on (WildlifeReID-10k). One image ->
one embedding; a sequence's fingerprint is the mean of its images' embeddings, unit length.
Everything around it is version 14 (D)'s: the detector, the association transformer, MmCows'
test window, the scores.

OpenCowID's own, written again here (porespy not needed):
  coats   blobs of 107 x 224 (porosity 0.5, blobiness 0.25) -> cream and dark grey with noise,
          blurred; views: shift 15 %, any rotation, perspective 0.2, blur; black padding to 224
  net     torchvision resnet50 (1000 outputs) -> ReLU -> Linear(1000, 128) -> unit length;
          every image normalised with the synthetic coats' mean and spread, as their code does
  loss    batch convergence (one cow's embeddings pulled together)
          + 0.005 x distance to the cow's virtual centroid
          + 0.005 x sum of 1 / distance to the other cows' centroids
          centroids: each class's mean embedding at the stage's start, pushed apart on the
          unit sphere for 200 steps (lr 0.1)
  train   Adam lr 1e-5, batch 128, colour jitter / shift 20 % / perspective 0.4 / blur; the
          epoch kept by a real validation set (held-out cows)
  test    1-NN against a gallery of real images
Not theirs: batches of P cows x K views (--K 1 is their plain shuffle; each batch from one
source set, as v14 D), validation and test as version 14 (D)'s - sequences, album by k-means
per cow, top-1 / mAP, the "not sure" model and the enrolment threshold on held-out
WildlifeReID-10k cows, MmCows' 16 cows (none trained on) as the test; the per-image 1-NN
top-1 (their own measure) next to it.

    python herd/v14/opencowid.py coats --out /workspace/v14/coats.png        (a look at the coats)
    python herd/v14/opencowid.py train --images /workspace/v14/iddi --out /workspace/v14/idnet_ocid
    python herd/v14/opencowid.py eval  --images /workspace/v14/iddi --out /workspace/v14/idnet_ocid

train writes <out>/idnet.pt (idnet.load reads it; system.py / system_e.py use it as the ID model
with its own encoder), <out>/idnet_synth.pt + eval_id_synth.json (the synthetic stage alone:
zero-shot), <out>/eval_id.json (idnet.py's scores + test_single_image), <out>/history.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np

import v14common as C  # noqa: F401  (paths)
from v14common import read_jsonl, write_json

EMB_DIM = 128
MEAN = (0.2032, 0.1978, 0.2084)          # OpenCowID's: the synthetic coats' mean and spread, used for every image
STD = (0.3502, 0.3455, 0.3491)
COAT_SHAPE = (107, 224)
STAGE_NAMES = {"synth": "synthetic coats (OpenCowID's, zero-shot)", "wr10k": "WildlifeReID-10k's cattle (v14 D's)"}


# ------------------------------------------------------------------ synthetic coats

def blobs(rng, shape=COAT_SHAPE, porosity=0.5, blobiness=0.25):
    """porespy.generators.blobs as OpenCowID calls it: smoothed noise cut at `porosity` -> bool (H, W), True = white."""
    from scipy.ndimage import gaussian_filter
    from scipy.special import erfc
    sigma = float(np.mean(shape)) / (40 * blobiness)
    im = gaussian_filter(rng.random(shape), sigma)
    im = (im - im.mean()) / (im.std() + 1e-12)
    im = 0.5 * erfc(-im / np.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min() + 1e-12)
    return im < porosity


def coat(rng, shape=COAT_SHAPE, porosity=0.5, blobiness=0.25):
    """One synthetic Holstein coat (OpenCowID's make_cow_coat_realistic) -> uint8 (H, W, 3)."""
    from PIL import Image, ImageFilter
    white = blobs(rng, shape, porosity, blobiness)
    a = np.zeros(tuple(shape) + (3,), np.int32)
    a[white] = np.array([255, 255, 250]) + rng.integers(-10, 10, (int(white.sum()), 3))
    a[~white] = 40 + rng.integers(-20, 20, (int((~white).sum()), 3))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
    return np.asarray(img)


def pad_square(img, size=224):
    """OpenCowID's Custom_resize_transform: longest side to `size`, centred on black."""
    from PIL import Image
    w, h = img.size
    r = size / max(w, h)
    img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    out = Image.new("RGB", (size, size))
    out.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return out


def coat_views():
    """The views OpenCowID saves of every coat (generate_data.py): shift, any rotation, perspective, blur."""
    from torchvision import transforms as T
    return T.Compose([T.RandomAffine(degrees=0, translate=(0.15, 0.15)),
                      T.RandomRotation(degrees=360, expand=True),
                      T.RandomPerspective(distortion_scale=0.2, p=0.6),
                      T.RandomApply([T.GaussianBlur(kernel_size=(7, 7), sigma=(0.01, 2))], p=0.3)])


def train_views():
    """OpenCowID's training transform (utils.utils_load_data)."""
    from torchvision import transforms as T
    return T.Compose([T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.8, hue=0.1)], p=0.5),
                      T.RandomApply([T.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.7),
                      T.RandomPerspective(distortion_scale=0.4, p=0.6),
                      T.RandomApply([T.GaussianBlur(kernel_size=(7, 7), sigma=(0.01, 2))], p=0.7)])


def sheet(images, path, cols=10, tile=112):
    """A few coats or crops on one picture, to look at."""
    from PIL import Image
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * tile, rows * tile), (25, 25, 25))
    for i, a in enumerate(images):
        out.paste(pad_square(Image.fromarray(np.asarray(a)), tile), ((i % cols) * tile, (i // cols) * tile))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.save(path)


# ------------------------------------------------------------------ the net

def build(backbone="resnet50", pretrained=True, emb=EMB_DIM):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    class CoatNet(nn.Module):
        """OpenCowID's Orig_resnet_with_projection: uint8 crops (B, H, W, 3) -> unit embeddings (B, 128)."""

        def __init__(self):
            super().__init__()
            self.kw = dict(backbone=backbone, pretrained=bool(pretrained), emb=emb)
            weights = None
            if pretrained:
                weights = getattr(torchvision.models, {"resnet50": "ResNet50_Weights", "resnet18": "ResNet18_Weights",
                                                       "resnet34": "ResNet34_Weights"}[backbone]).IMAGENET1K_V1
            self.encoder = getattr(torchvision.models, backbone)(weights=weights)       # 1000 outputs, as theirs
            self.projection_head = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(1000, emb))
            self.out_dim = emb
            self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

        def forward(self, crops_uint8):
            if not self.training:                      # in use (cbvd_bursts.encode): no gradient kept
                with torch.no_grad():
                    return self.embed(crops_uint8)
            return self.embed(crops_uint8)

        def embed(self, crops_uint8):
            x = crops_uint8.permute(0, 3, 1, 2).float().div(255.0)
            x = (x - self.mean) / self.std
            return F.normalize(self.projection_head(self.encoder(x)), dim=1)

    return CoatNet()


def from_checkpoint(ck):
    net = build(**{**ck["kw"], "pretrained": False})      # the saved weights, no ImageNet download
    net.kw = dict(ck["kw"])
    net.load_state_dict(ck["state"])
    return net.eval()


def seq_mean():
    """idnet's interface on OpenCowID embeddings: (x (B, T, 128), t, valid) -> the mean embedding, unit length;
    'quality' of a frame = how much it agrees with that mean (cosine)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SeqMean(nn.Module):
        def forward(self, x, t, valid):
            w = valid.float() / valid.float().sum(1, keepdim=True).clamp(min=1)
            fp = F.normalize((w.unsqueeze(-1) * x).sum(1), dim=-1)
            q = (F.normalize(x, dim=-1) * fp.unsqueeze(1)).sum(-1).masked_fill(~valid, -1e4)
            return {"fingerprint": fp, "quality": q, "weights": w}

    return SeqMean()


# ------------------------------------------------------------------ images

class ImageStore:
    """data_images.py's folder: crops.jsonl rows {name, image, day, cow, seq, t, cam, dataset} grouped into
    sequences (as idnet.Store, without the vectors)."""

    def __init__(self, folder):
        self.folder = folder
        self.rows = read_jsonl(os.path.join(folder, "crops.jsonl"))
        p = os.path.join(folder, "meta.json")
        self.meta = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        seqs = collections.defaultdict(list)
        for i, r in enumerate(self.rows):
            seqs[(r["day"], r["cow"], r["seq"])].append(i)
        self.seqs = {}
        for k, idx in seqs.items():
            idx.sort(key=lambda i: (self.rows[i]["t"], self.rows[i]["name"]))
            self.seqs[k] = idx

    def days(self):
        return sorted({k[0] for k in self.seqs})

    def path(self, i):
        p = self.rows[i]["image"]
        return p if os.path.isabs(p) else os.path.join(self.folder, p)

    def image(self, i, size):
        from PIL import Image
        img = Image.open(self.path(i)).convert("RGB")
        return img if img.size == (size, size) else pad_square(img, size)


class FeatStore:
    """idnet.Store's two calls (seqs, whole) on embeddings held in memory - for idnet.score_keys."""

    def __init__(self, store, feats):
        self.rows, self.seqs, self.feats = store.rows, store.seqs, feats

    def whole(self, keys, max_len=32):
        out = []
        for k in keys:
            idx = self.seqs[k]
            if len(idx) > max_len:
                idx = [idx[i] for i in np.linspace(0, len(idx) - 1, max_len).round().astype(int)]
            t = np.array([self.rows[i]["t"] for i in idx], np.float32)
            out.append((np.asarray(self.feats[idx], np.float32), t - t[0]))
        return out


def make_datasets():
    import torch.utils.data as D

    class CoatViews(D.Dataset):
        """n_ids synthetic coats, `views` items each; every item a fresh view (OpenCowID's coat views, then its
        training views), padded black to `size`."""

        def __init__(self, n_ids, views, size, seed=0, aug=True):
            self.n, self.views, self.size, self.seed, self.aug = n_ids, views, size, seed, aug
            self.cache = {}
            self.gen, self.tr = coat_views(), train_views()

        def __len__(self):
            return self.n * self.views

        def label(self, k):
            return k // self.views

        def base(self, cid):
            if cid not in self.cache:
                self.cache[cid] = coat(np.random.default_rng((self.seed, cid)))
            return self.cache[cid]

        def __getitem__(self, k):
            from PIL import Image
            img = Image.fromarray(self.base(self.label(k)))
            if self.aug:
                img = self.tr(self.gen(img))
            return np.array(pad_square(img, self.size)), self.label(k)

    class RealViews(D.Dataset):
        """Rows of an ImageStore (their row indices), label = a class index per row; training views or plain."""

        def __init__(self, store, idx, labels, size, aug):
            self.store, self.idx, self.labels, self.size, self.aug = store, list(idx), list(labels), size, aug
            self.tr = train_views()

        def __len__(self):
            return len(self.idx)

        def label(self, k):
            return self.labels[k]

        def __getitem__(self, k):
            img = self.store.image(self.idx[k], self.size)
            if self.aug:
                img = self.tr(img)
                if img.size != (self.size, self.size):
                    img = pad_square(img, self.size)
            return np.array(img), self.labels[k]

    return CoatViews, RealViews


def pk_batches(labels, pools, P, K, steps, rng):
    """`steps` batches of P classes x K items: the classes of one pool at a time (a cow in two source sets
    under two names never meets herself as another cow), items with replacement when a class has fewer."""
    items = collections.defaultdict(list)
    for i, c in enumerate(labels):
        items[c].append(i)
    pools = [[c for c in p if c in items] for p in pools]
    pools = [p for p in pools if p]
    for _ in range(steps):
        pool = rng.choices(pools, weights=[len(p) for p in pools])[0]
        batch = []
        for c in rng.sample(pool, min(P, len(pool))):
            its = items[c]
            batch += rng.sample(its, K) if len(its) >= K else rng.choices(its, k=K)
        yield batch


def embed_dataset(net, ds, dev, batch=256, workers=4):
    """Every item of a dataset (no views) -> (n, 128) float32, in order."""
    import torch
    import torch.utils.data as D
    out = []
    loader = D.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    with torch.no_grad():
        for x, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(net(x.to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)


def embed_rows(net, store, idx, size, dev, batch=256, workers=4):
    _, RealViews = make_datasets()
    return embed_dataset(net, RealViews(store, idx, [0] * len(idx), size, aug=False), dev, batch, workers)


# ------------------------------------------------------------------ OpenCowID's losses and centroids

def virtual_centroids(means, steps=200, lr=0.1):
    """get_optimal_virtual_centroids: the class means pushed apart on the unit sphere (the sum of their
    pairwise distances climbed by plain gradient steps, each centroid put back to unit length)."""
    import torch
    c = torch.as_tensor(np.asarray(means, np.float32)).clone()
    dev = c.device
    eye = torch.eye(len(c), device=dev)
    c.requires_grad_(True)
    for _ in range(steps):
        d2 = (c.pow(2).sum(1)[:, None] + c.pow(2).sum(1)[None, :] - 2 * c @ c.T).clamp(min=0)
        loss = -((d2 + eye).sqrt() * (1 - eye)).sum()
        loss.backward()
        with torch.no_grad():
            c -= lr * c.grad
            c /= c.norm(dim=1, keepdim=True).clamp(min=1e-12)
        c.grad = None
    return c.detach()


def losses(e, y, cen):
    """OpenCowID's three terms on a batch: (batch convergence, distance to own centroid, repulsion from the others)."""
    import torch
    dist = ((e.unsqueeze(1) - e.unsqueeze(0)) ** 2).mean(-1)                 # their mean over the 128 numbers
    same = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    same = same - torch.diag(torch.diag(same))
    conv = ((dist * same).sum(-1) / (same.sum(-1) + 1e-3)).mean()
    own = (e - cen[y]).norm(dim=-1).mean()
    inv = 1.0 / torch.cdist(e, cen).clamp(min=1e-6)
    others = 1.0 - torch.nn.functional.one_hot(y, len(cen)).to(inv.dtype)
    rep = (inv * others).sum() / max(1, len(e) - 1) / len(e)
    return conv, own, rep


# ------------------------------------------------------------------ training

def class_means(net, ds, labels, n_classes, dev, per_class, workers, seed=0):
    """Each class's mean embedding (the centroids' start), from up to `per_class` plain items per class."""
    import torch.utils.data as D
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for i, c in enumerate(labels):
        by[c].append(i)
    pick = [i for c in sorted(by) for i in (rng.sample(by[c], per_class) if len(by[c]) > per_class else by[c])]
    aug, ds.aug = ds.aug, False
    try:
        E = embed_dataset(net, D.Subset(ds, pick), dev, workers=workers)
    finally:
        ds.aug = aug
    sums, counts = np.zeros((n_classes, E.shape[1]), np.float64), np.zeros(n_classes)
    for i, e in zip(pick, E):
        sums[labels[i]] += e
        counts[labels[i]] += 1
    return sums / np.maximum(counts, 1)[:, None]


def val_score(net, store, plan, size, dev, cache, workers):
    """Version 14 (D)'s validation: album = valA sequences (k-means per cow), queries = valB sequences."""
    import idnet as I
    if "idx" not in cache:
        cache["idx"] = sorted({i for k in plan["val_album"] + plan["val_query"] for i in store.seqs[k]})
    E = embed_rows(net, store, cache["idx"], size, dev, workers=workers)
    feats = np.zeros((len(store.rows), E.shape[1]), np.float32)
    feats[cache["idx"]] = E
    res, _, _, _, _ = I.score_keys(seq_mean(), FeatStore(store, feats), dev, plan["val_album"], plan["val_query"])
    return res


def checkpoint(net, epoch, stage, plan, size, store):
    return {"arch": "opencowid", "kw": net.kw, "state": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "epoch": epoch, "stage": stage, "split": plan["about"],
            "feats": {"encoder": "opencowid", "crop": size, "margin": 0.1, "dim": net.out_dim, "mean": MEAN, "std": STD,
                      "source": store.meta.get("source"), "images": store.meta}}


def train_stage(net, stage, ds, labels, pools, n_classes, args, dev, store, plan, hist, save_to):
    """OpenCowID's fine_tune on one data stage; the epoch with the best validation (top-1 + mAP, as idnet.py)
    is saved to save_to and left in `net`. Returns the best score."""
    import torch
    rng = random.Random(args.seed)
    epochs = args.synth_epochs if stage == "synth" else args.epochs
    print(f"[ocid] stage {stage}: {STAGE_NAMES.get(stage, stage)} - {len(ds)} items of {n_classes} classes, "
          f"{epochs} epochs at most, batches of {args.P} x {args.K}", flush=True)
    t0 = time.time()
    means = class_means(net, ds, labels, n_classes, dev, args.centroid_views, args.workers, args.seed)
    cen = virtual_centroids(means, args.centroid_steps, args.centroid_lr).to(dev)
    print(f"[ocid] virtual centroids: {len(cen)}, nearest pair {float(torch.cdist(cen, cen).fill_diagonal_(9).min()):.3f} "
          f"apart ({(time.time() - t0) / 60:.1f} min)", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = max(1, len(ds) // (args.P * args.K))
    best, since, cache = -1.0, 0, {}
    for ep in range(epochs):
        net.train()
        batches = list(pk_batches(labels, pools, args.P, args.K, steps, rng))
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=args.workers)
        sums = np.zeros(4)
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                e = net(x).float()
            conv, own, rep = losses(e, y, cen)
            loss = conv + args.w_centroid * own + args.w_repulsion * rep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums += [loss.item(), conv.item(), own.item(), rep.item()]
        net.eval()
        res = val_score(net, store, plan, args.size, dev, cache, args.workers)
        score = (res["top1"] or 0) + (res["mAP"] or 0)
        m = sums / max(1, len(batches))
        rec = {"stage": stage, "epoch": ep + 1, "loss": round(float(m[0]), 5), "batch_conv": round(float(m[1]), 5),
               "centroid": round(float(m[2]), 4), "repulsion": round(float(m[3]), 4), "val_top1": res["top1"],
               "val_mAP": res["mAP"], "minutes": round((time.time() - t0) / 60, 1)}
        hist.append(rec)
        print(f"[ocid] {stage} epoch {ep + 1}: loss {m[0]:.4f} (conv {m[1]:.5f}, centroid {m[2]:.3f}, repulsion "
              f"{m[3]:.3f})  val top-1 {res['top1']:.1%}  mAP {res['mAP']:.3f}  {rec['minutes']} min", flush=True)
        if score > best:
            best, since = score, 0
            torch.save(checkpoint(net, ep + 1, stage, plan, args.size, store), save_to)
        else:
            since += 1
            if since >= args.patience:
                print(f"[ocid] no better val score for {args.patience} epochs - stopping the stage", flush=True)
                break
    ck = torch.load(save_to, map_location=dev, weights_only=False)
    net.load_state_dict(ck["state"])
    net.eval()
    return best


def cmd_train(args):
    import torch
    import idnet as I
    from common import device as dev_of
    dev = dev_of()
    torch.manual_seed(args.seed)
    store = ImageStore(args.images)
    plan = I.split_plan(store, argparse.Namespace(split="groups"))
    net = build(args.backbone, bool(args.pretrained)).to(dev)
    print(f"[ocid] {args.backbone}{' (ImageNet weights)' if args.pretrained else ' (random start)'} + projection to "
          f"{net.out_dim}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters", flush=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    CoatViews, RealViews = make_datasets()
    hist = []
    final = os.path.join(args.out, "idnet.pt")
    for stage in stages:
        if stage == "synth":
            ds = CoatViews(args.synth_ids, args.synth_views, args.size, args.seed)
            labels = [ds.label(k) for k in range(len(ds))]
            pools = [list(range(args.synth_ids))]
            train_stage(net, stage, ds, labels, pools, args.synth_ids, args, dev, store, plan, hist, final)
            if len(stages) > 1:
                # the zero-shot model, kept and scored on its own
                os.replace(final, os.path.join(args.out, "idnet_synth.pt"))
                write_json(os.path.join(args.out, "history.json"), hist)
                cmd_eval(argparse.Namespace(**{**vars(args), "model_name": "idnet_synth.pt", "eval_name": "eval_id_synth.json",
                                               "only_cows": None, "keep_model": False}))
        elif stage == "wr10k":
            keys = plan["train"]
            cows = sorted({k[1] for k in keys}, key=str)
            c_of = {c: i for i, c in enumerate(cows)}
            idx = [i for k in keys for i in store.seqs[k]]
            labels = [c_of[store.rows[i]["cow"]] for i in idx]
            by_set = collections.defaultdict(set)
            for i in idx:
                by_set[store.rows[i].get("dataset")].add(c_of[store.rows[i]["cow"]])
            pools = [sorted(v) for v in by_set.values()] if args.pure_batches else [list(range(len(cows)))]
            if len(pools) > 1:
                print(f"[ocid] batches from one set at a time: {({k: len(v) for k, v in by_set.items()})}", flush=True)
            ds = RealViews(store, idx, labels, args.size, aug=True)
            train_stage(net, stage, ds, labels, pools, len(cows), args, dev, store, plan, hist, final)
        else:
            sys.exit(f"unknown stage {stage!r}: synth, wr10k")
    write_json(os.path.join(args.out, "history.json"), hist)
    cmd_eval(argparse.Namespace(**{**vars(args), "model_name": "idnet.pt", "eval_name": "eval_id.json",
                                   "only_cows": None, "keep_model": False}))


# ------------------------------------------------------------------ eval

def single_image_top1(feats, store, plan, only=None):
    """OpenCowID's own measure: every query image against every gallery image, nearest one wins (1-NN)."""
    import metrics
    out = {}
    for name, (a, q) in (("val", ("val_album", "val_query")), ("test", ("test_album", "test_query"))):
        ga = [i for k in plan[a] for i in store.seqs[k]]
        qu = [i for k in plan[q] for i in store.seqs[k]]
        known = {store.rows[i]["cow"] for i in ga}
        qu = [i for i in qu if store.rows[i]["cow"] in known]
        if not ga or not qu:
            out[name] = None
            continue
        r = metrics.retrieval(feats[qu], [store.rows[i]["cow"] for i in qu], feats[ga], [store.rows[i]["cow"] for i in ga])
        out[name] = {"top1": r["top1"], "top5": r["top5"], "queries": r["queries"], "gallery_images": len(ga)}
    return out


def cmd_eval(args):
    """Embeds every image of the store with the saved model -> <out>/feats (an idnet-style feature folder), then
    idnet.py's scores on it (album / queries per cow, the 'not sure' model and the enrolment threshold on the
    val cows) + the per-image 1-NN top-1."""
    import torch
    import idnet as I
    from common import device as dev_of
    dev = dev_of()
    model_name = getattr(args, "model_name", None) or "idnet.pt"
    ck = torch.load(os.path.join(args.out, model_name), map_location="cpu", weights_only=False)
    net = from_checkpoint(ck).to(dev)
    store = ImageStore(args.images)
    size = int(ck["feats"].get("crop") or args.size)
    t0 = time.time()
    feats = embed_rows(net, store, list(range(len(store.rows))), size, dev, workers=args.workers)
    print(f"[ocid] {len(feats)} images embedded ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fdir = os.path.join(args.out, "feats" if model_name == "idnet.pt" else "feats_" + model_name.replace(".pt", ""))
    os.makedirs(fdir, exist_ok=True)
    np.save(os.path.join(fdir, "feats.npy"), feats.astype(np.float16))
    with open(os.path.join(fdir, "crops.jsonl"), "w", encoding="utf-8") as fh:
        for r in store.rows:
            fh.write(json.dumps(r) + "\n")
    write_json(os.path.join(fdir, "meta.json"), {**ck["feats"], "model": model_name})
    # idnet.py's eval reads <out>/idnet.pt: point it at this model
    swap = model_name != "idnet.pt"
    if swap:
        tmp_out = os.path.join(args.out, "_eval_" + model_name.replace(".pt", ""))
        os.makedirs(tmp_out, exist_ok=True)
        torch.save(ck, os.path.join(tmp_out, "idnet.pt"))
    else:
        tmp_out = args.out
    eval_name = getattr(args, "eval_name", None) or "eval_id.json"
    I.cmd_eval(argparse.Namespace(feats=fdir, out=tmp_out, max_error=args.max_error, split="groups", val_cows=3,
                                  test_cows=5, split_seed=0, only_cows=getattr(args, "only_cows", None),
                                  eval_name=eval_name, keep_model=getattr(args, "keep_model", False)))
    plan = I.split_plan(store, argparse.Namespace(split="groups"))
    only = getattr(args, "only_cows", None)
    if isinstance(only, str) and only:
        keep = {c.strip() for c in only.split(",")}
        for k in ("test_album", "test_query"):
            plan[k] = [q for q in plan[k] if str(q[1]) in keep]
    ev_path = os.path.join(tmp_out, eval_name)
    ev = json.load(open(ev_path, encoding="utf-8"))
    ev["test_single_image"] = single_image_top1(feats, store, plan)
    ev["what_single_image"] = "OpenCowID's own measure: each query image alone against every gallery image (1-NN)"
    ev["stage"] = ck.get("stage")
    ev["training"] = {k: getattr(args, k) for k in ("stages", "synth_ids", "synth_views", "synth_epochs", "epochs", "P", "K",
                                                     "lr", "w_centroid", "w_repulsion", "size") if hasattr(args, k)}
    write_json(ev_path, ev)
    if swap:
        os.replace(ev_path, os.path.join(args.out, eval_name))
        if not getattr(args, "keep_model", False):
            os.replace(os.path.join(tmp_out, "idnet.pt"), os.path.join(args.out, model_name))
        for f in os.listdir(tmp_out):
            os.remove(os.path.join(tmp_out, f))
        os.rmdir(tmp_out)
    si = ev["test_single_image"].get("test")
    if si:
        print(f"[ocid] per image, 1-NN: test top-1 {si['top1']:.1%} ({si['queries']} query images, "
              f"{si['gallery_images']} in the gallery)", flush=True)


def cmd_coats(args):
    rng = np.random.default_rng(args.seed)
    from PIL import Image
    gen = coat_views()
    imgs = []
    for i in range(args.n):
        base = coat(rng)
        imgs.append(base)
        imgs += [np.asarray(gen(Image.fromarray(base))) for _ in range(args.views)]
    sheet(imgs, args.out, cols=1 + args.views)
    print(f"[ocid] {args.n} coats, {args.views} views each -> {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("coats")
    s.add_argument("--out", default="coats.png")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--views", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    for name in ("train", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--images", required=True, help="data_images.py combine's folder")
        s.add_argument("--out", required=True)
        s.add_argument("--max-error", type=float, default=0.01)
        s.add_argument("--size", type=int, default=224, help="the images' side (crops are padded to a square)")
        s.add_argument("--workers", type=int, default=8)
        if name == "eval":
            s.add_argument("--model-name", default="idnet.pt", help="which model in --out (idnet_synth.pt: zero-shot)")
            s.add_argument("--only-cows", default=None, help="comma list: score the test on these cows only")
            s.add_argument("--eval-name", default=None, help="file name of the scores (default eval_id.json)")
            s.add_argument("--keep-model", action="store_true", help="do not write the refitted 'not sure' model back")
        if name == "train":
            s.add_argument("--stages", default="synth,wr10k", help="synth: OpenCowID's synthetic coats; wr10k: v14 D's cattle")
            s.add_argument("--backbone", default="resnet50", choices=("resnet50", "resnet34", "resnet18"))
            s.add_argument("--pretrained", type=int, default=1, help="1: ImageNet weights (OpenCowID's start)")
            s.add_argument("--synth-ids", type=int, default=2000, help="synthetic cows")
            s.add_argument("--synth-views", type=int, default=50, help="views of each synthetic cow per epoch")
            s.add_argument("--synth-epochs", type=int, default=10)
            s.add_argument("--epochs", type=int, default=30, help="real stage epochs at most")
            s.add_argument("--patience", type=int, default=6)
            s.add_argument("--P", type=int, default=32, help="classes per batch")
            s.add_argument("--K", type=int, default=4, help="items per class in a batch (P x K = 128, OpenCowID's batch)")
            s.add_argument("--lr", type=float, default=1e-5, help="OpenCowID's Adam learning rate")
            s.add_argument("--w-centroid", type=float, default=0.005)
            s.add_argument("--w-repulsion", type=float, default=0.005)
            s.add_argument("--centroid-views", type=int, default=8, help="items per class for the centroids' start")
            s.add_argument("--centroid-steps", type=int, default=200)
            s.add_argument("--centroid-lr", type=float, default=0.1)
            s.add_argument("--pure-batches", type=int, default=1, help="1: each real batch from one source set")
            s.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    return {"coats": cmd_coats, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
