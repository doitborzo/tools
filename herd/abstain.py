"""The "enough data?" head: when to say NaN instead of a cow.

Trained after the main model, on bursts it never trained on (the dev clips):

  1. re-identify every dev burst (its last seconds against all bursts' first
     seconds) and note whether the best match was right;
  2. fit a small logistic model: p(right) from how close the best match was,
     how far ahead of the second-best cow, how good the burst's frames were
     (quality head) and how big the cow was in frame;
  3. pick the cut-off: the lowest p at which the answers kept are wrong at
     most --max-error of the time. Below it the system answers NaN.

    python abstain.py --features /workspace/herd/features --run /workspace/herd/run1

Writes <run>/abstain.json (the model and cut-off; the gallery loads it) and
reports coverage - how often it answers at all - at several error budgets,
on dev and on val. A solid-coloured cow with no gait in view should end up
below the cut-off: low margin, low quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from common import device, read_jsonl, write_json

FEATURES = ("sim", "margin", "quality_max", "quality_mean", "log_area")


def feature_matrix(rows, features=FEATURES):
    col = {"log_area": lambda r: np.log(max(r["area"], 1e-4))}
    return np.array([[col[f](r) if f in col else r[f] for f in features] for r in rows], dtype=np.float64)


def fit_logistic(X, y, l2=1e-2, steps=3000, lr=0.1):
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / std
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return {"mean": mean.tolist(), "std": std.tolist(), "w": w.tolist(), "b": float(b)}


def predict(m, X):
    Z = (X - np.array(m["mean"])) / np.array(m["std"])
    return 1 / (1 + np.exp(-(Z @ np.array(m["w"]) + m["b"])))


def cutoff(p, y, max_error):
    """Lowest threshold whose kept answers are wrong <= max_error; 1.01 = never answer."""
    order = np.argsort(-p)
    wrong = np.cumsum(1 - y[order])
    kept = np.arange(1, len(y) + 1)
    ok = np.where(wrong / kept <= max_error)[0]
    if not len(ok):
        return 1.01
    return float(p[order][ok[-1]])


def coverage_table(p, y, thresholds):
    out = []
    for name, t in thresholds:
        keep = p >= t
        out.append({"budget": name, "threshold": round(t, 4), "answered": float(keep.mean()) if len(p) else 0.0,
                    "error_when_answered": float((1 - y[keep]).mean()) if keep.any() else None})
    return out


def cmd(args):
    import train as train_mod
    from model import HerdModel
    model, ck = HerdModel.load(os.path.join(args.run, "model.pt"), map_location=device())
    model.to(device())
    dev = train_mod.Split(os.path.join(args.features, "train"), set(ck["dev_clips"]))
    rows, _ = train_mod.reid(model, dev, ck["crop_s"])
    # bad bursts too (degrade.py), when made: the barn has them, and only from them
    # can the NaN model learn that low frame quality means "do not trust"
    if dev.deg_idx is not None:
        rows += train_mod.reid(model, dev, ck["crop_s"], degrade=True)[0]
    rows = [r for r in rows if r["scope"] == args.scope]
    features = FEATURES
    X, y = feature_matrix(rows, features), np.array([r["correct"] for r in rows], float)
    m = fit_logistic(X, y)
    p = predict(m, X)
    budgets = sorted(set([args.max_error, 0.01, 0.02, 0.05]))
    th = {e: cutoff(p, y, e) for e in budgets}
    same = np.array([r["sim"] for r in rows if r["correct"]])
    out = {"features": list(features), **m, "max_error": args.max_error, "threshold": th[args.max_error],
           "scope": args.scope, "dev_queries": len(rows), "dev_top1": float(y.mean()) if len(y) else None,
           # a new cow's bursts look like each other at least this much: the gallery's
           # starting point for grouping unknown cows (identity.new_cow_similarity)
           "same_cow_sim_p05": float(np.percentile(same, 5)) if len(same) else None,
           "dev": coverage_table(p, y, [(f"{e:.0%}", th[e]) for e in budgets])}
    val_dir = os.path.join(args.features, "val")
    if os.path.exists(os.path.join(val_dir, "index.jsonl")):
        val = train_mod.Split(val_dir)
        sets = [("val", train_mod.reid(model, val, ck["crop_s"])[0])]
        if val.deg_idx is not None:
            sets.append(("val_spoilt", train_mod.reid(model, val, ck["crop_s"], degrade=True)[0]))
    else:
        sets = [("val", read_jsonl(os.path.join(args.run, "reid_val.jsonl")))]
    for name, vr in sets:
        vr = [r for r in vr if r["scope"] == args.scope]
        if vr:
            pv = predict(m, feature_matrix(vr, features))
            yv = np.array([r["correct"] for r in vr], float)
            out[name] = coverage_table(pv, yv, [(f"{e:.0%}", th[e]) for e in budgets])
            out[f"{name}_top1"] = float(yv.mean())
    write_json(os.path.join(args.run, "abstain.json"), out)
    print(f"[abstain] dev: {len(rows)} queries, top-1 {out['dev_top1']:.1%}")
    print(f"[abstain] features: {', '.join(features)}")
    for part in ("dev", "val", "val_spoilt"):
        for r in out.get(part, []):
            e = r["error_when_answered"]
            print(f"  {part}: budget {r['budget']:>4}  threshold {r['threshold']:.3f}  answers "
                  f"{r['answered']:.1%}  wrong when answering {'-' if e is None else f'{e:.1%}'}")
    print(f"-> {os.path.join(args.run, 'abstain.json')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="/workspace/herd/features")
    p.add_argument("--run", default="/workspace/herd/run1")
    p.add_argument("--max-error", type=float, default=0.01)
    p.add_argument("--scope", choices=("all", "clip"), default="all",
                   help="all: every dev cow is a candidate (closer to a barn of many cows)")
    cmd(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
