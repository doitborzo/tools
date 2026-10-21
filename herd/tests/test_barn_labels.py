"""The barn's own cows: the labelling page, merges.csv, the export to the identity model's training set."""
import _paths  # noqa: F401
import json
import os
import tarfile
import tempfile

import numpy as np


def _cache(folder, dim, rng):
    """A training cache of 8 cows, 3 tracks each over 2 days (the pipeline writes it so)."""
    from barn_dataset import TrainingCache
    cfg = {"training_cache": {"folder": folder, "every_n": 1, "max_gb": 1}, "store": {"path": os.path.join(folder, "x.sqlite")},
           "farm": {"timezone_offset_hours": 0}}
    meta = {"encoder": "x", "grid": 2, "crop": 224, "margin": 0.1, "dim": dim}
    cache = TrainingCache(cfg, meta)
    vec = {c: rng.normal(size=dim) * 1.5 for c in range(8)}
    tracks = {}
    for c in range(8):
        for j, day in enumerate((0, 0, 1)):
            track = f"cam{j % 2 + 1}-{c}-{j}"
            tracks[track] = f"UA{c:02d}"
            for b in range(3):
                ts = 1790000000 + day * 86400 + c * 600 + j * 100 + b * 20
                feats = vec[c] + rng.normal(size=(50, dim)) * 0.3
                valid = np.ones(50, bool)
                valid[::5] = False                                  # frames the cow was not found in
                cache.add(f"cam{j % 2 + 1}", track, ts, feats, np.arange(50) / 7.0, valid, None, [0.5, 0.5, 0.2, 0.3],
                          np.full((64, 64, 3), 100 + c * 10, np.uint8),
                          {"state": "confirmed", "cow": f"new-{c}", "sim": 0.9, "p": 0.9, "area": 0.05})
    return meta, tracks


def test_sheet_and_export_feed_the_identity_model():
    import torch
    import barn_dataset
    import model as mdl
    import reid
    rng = np.random.default_rng(0)
    dim = 16
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "training_cache")
        meta, tracks = _cache(cache, dim, rng)
        merges = os.path.join(d, "merges.csv")
        bad = "cam1-7-0"                                            # two cows swapped in it: left out
        with open(merges, "w") as fh:
            fh.write("identity,track\n" + "".join(f"{'x' if t == bad else n},{t}\n" for t, n in tracks.items()))
        # the page: every track with its thumbnails and the names saved earlier
        page = os.path.join(d, "tracks.html")
        names = os.path.join(d, "names.json")
        json.dump({"new-3": "UA03"}, open(names, "w"))
        barn_dataset.main(["sheet", "--cache", cache, "--out", page, "--merges", merges, "--names", names])
        html = open(page, encoding="utf-8").read()
        data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
        assert len(data["tracks"]) == 24 and data["labels"][bad] == "x"
        t3 = next(t for t in data["tracks"] if t["track"] == "cam1-3-0")
        assert t3["n"] == 3 and t3["guess"] == "new-3" and t3["guess_name"] == "UA03" and len(t3["thumbs"]) == 3
        assert all(os.path.exists(os.path.join(d, p)) for p in t3["thumbs"])          # relative to the page
        assert "Save merges.csv" in html and "localStorage" in html
        # build (the herd model's format) leaves the x track out too
        assert barn_dataset.identities([{"track": bad}], {bad: "x"})[bad] is None
        # export -> reid's feat/barn
        out = os.path.join(d, "herd_barn_feat_test.tgz")
        barn_dataset.main(["export", "--cache", cache, "--merges", merges, "--out", out, "--frames", "20"])
        work = os.path.join(d, "work")
        with tarfile.open(out) as tar:
            tar.extractall(work)
        R = os.path.join(work, "reid")
        rows = [json.loads(line) for line in open(os.path.join(R, "feat", "barn", "bursts.jsonl"))]
        assert bad not in {b["track"] for b in rows} and len(rows) == 8 * 9 - 3
        assert all(len(b["frames"]) == 20 for b in rows)                             # 40 valid frames -> 20
        split = {}
        for b in rows:
            assert split.setdefault(b["cow"], b["split"]) == b["split"]             # a cow wholly one side
        val = {c for c, s in split.items() if s == "val"}
        assert len(val) == 1 and len(split) == 8
        feats = np.load(os.path.join(R, "feat", "barn", "crops.npz"))["feats"]
        assert feats.shape == (sum(len(b["frames"]) for b in rows), dim)
        tr = reid.Bursts(R, ["barn"], "train", meta)
        va = reid.Bursts(R, ["barn"], "val", meta)
        assert len(va.bursts) == 9 and tr.whole(0).shape == (reid.SLOTS, dim)
        assert not reid.Bursts(R, ["barn"], "train", dict(meta, encoder="other")).bursts   # other vectors: left out
        # the identity model trains on it
        torch.manual_seed(0)
        m = mdl.HerdModel(dim, d=32, layers=1, heads=2)
        mp = os.path.join(d, "model.pt")
        m.save(mp, {"features": meta, "crop_s": 3.5, "dev_clips": []})
        reid.main(["fetch", "--data", R, "--sources", "barn"])                       # nothing to fetch
        reid.main(["extract", "--data", R, "--sources", "barn", "--model", mp])      # nor to extract
        reid.main(["train", "--data", R, "--sources", "barn", "--model", mp, "--epochs", "2", "--P", "16"])
        ck = torch.load(os.path.join(R, "id_model", "identity.pt"), weights_only=False)
        assert ck["sources"] == ["barn"]
