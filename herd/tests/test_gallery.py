"""Gallery on a synthetic herd: enrol 60 cows from nothing, recognise them,
refuse an ambiguous burst, change-over on Tuesday: 10 leave, 10 arrive."""
import datetime as dt
import tempfile

import numpy as np

import _paths  # noqa: F401
from common import load_config
from gallery import Gallery


def test_herd_life_cycle():
    folder = tempfile.mkdtemp()
    cfg = load_config(None)
    cfg["identity"]["new_cow_similarity"] = 0.6
    rng = np.random.default_rng(1)
    D = 512
    cows = rng.normal(size=(60, D)); cows /= np.linalg.norm(cows, axis=1, keepdims=True)
    def burst(c, noise=1.0):
        v = cows[c] + noise * rng.normal(size=D) / np.sqrt(D)
        return v / np.linalg.norm(v)
    monday = dt.datetime(2026, 10, 5, 12, tzinfo=dt.timezone.utc).timestamp()
    g = Gallery(folder, cfg)
    assert g.migration_due(monday) is False, "first start raises a change-over"
    for c in range(60):
        for k in range(25):
            f = burst(c); d = g.match(f); g.observe(d, f, monday + k * 60, f"cam1-track{c}")
    res = g.nightly(monday + 12 * 3600)
    assert len(res["enrolled"]) == 60, res
    names = {cid: int(np.argmax(cows @ g.cows[cid]["prototypes"][0])) for cid in g.cows}
    assert len(set(names.values())) == 60, "a cow enrolled twice or two cows merged"
    ok = sum(1 for c in range(60) for _ in range(5) if (lambda d: d["state"] == "confirmed" and names[d["cow"]] == c)(g.match(burst(c))))
    assert ok >= 297, ok
    f = cows[0] + cows[1]; f /= np.linalg.norm(f)
    assert g.match(f)["state"] != "confirmed", "an ambiguous burst was confirmed"
    tue = monday + 86400 + 11 * 3600
    assert g.migration_due(tue) is True and g.migration_due(tue + 3600) is False
    new = rng.normal(size=(10, D)); new /= np.linalg.norm(new, axis=1, keepdims=True)
    cows = np.vstack([cows, new])
    for c in list(range(50)) + list(range(60, 70)):
        for k in range(25):
            f = burst(c); d = g.match(f); g.observe(d, f, tue + 3600 + k * 60, f"cam2-track{c}-b")
    res = g.nightly(tue + 3 * 86400)
    left = {names[c] for c in res["retired"]}
    assert left == set(range(50, 60)), left
    assert len(res["enrolled"]) == 10, res
    assert len(Gallery(folder, cfg).cows) == 60

