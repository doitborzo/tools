"""Who-was-where-when in identification: cows seen at the same moment are different cows."""
import _paths  # noqa: F401
import copy
import os
import tempfile

import numpy as np

from common import DEFAULTS


def _cfg(**identity):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["identity"].update(identity)
    return cfg


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def _gallery(d, cows, **identity):
    import gallery as g
    gal = g.Gallery(d, _cfg(**identity))
    for cid, v in cows.items():
        gal.cows[cid] = {"prototypes": _unit(v)[None], "created": 0.0, "last_seen": 0.0}
    return gal


def _near(v, rng, noise=0.05):
    return _unit(np.asarray(v, np.float32) + rng.normal(0, noise, len(v)))


def test_match_batch_gives_no_cow_twice():
    rng = np.random.default_rng(0)
    a, b = np.eye(16)[0], np.eye(16)[0] * 0.8 + np.eye(16)[1] * 0.6
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {"A": a, "B": b})
        items = [{"fingerprint": _near(a, rng, 0.01)}, {"fingerprint": _near(a, rng, 0.2)}]
        ds = gal.match_batch(items)
        cows = [x["cow"] for x in ds if x["cow"]]
        assert len(cows) == len(set(cows))                         # never one cow for two bursts at once
        assert ds[0]["cow"] == "A"
        assert gal.match(items[1]["fingerprint"], exclude={"A"})["cow"] in (None, "B")


def test_claims_hold_a_cow_against_a_track_seen_at_the_same_time():
    import pipeline
    from store import Store
    a, b = np.eye(16)[0], np.eye(16)[2]
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        cfg = _cfg()
        shared = pipeline.SharedGallery(_gallery(d, {"A": a, "B": b}), store, cfg, 1000.0)
        out = lambda v: {"fingerprint": v, "quality_max": 0.5, "quality_mean": 0.5}
        d1 = shared.decide(out(_unit(a)), 0.05, 1000.0, "cam1-1")
        assert d1["state"] == "confirmed" and d1["cow"] == "A"
        shared.together(["cam1-1", "cam1-2"], 1001.0)               # both tracks in one frame: two cows
        # a less sure burst of the other track: A is held, so not A
        d2 = shared.decide(out(_unit(a * 0.9 + b * 0.25)), 0.05, 1010.0, "cam1-2")
        assert d2["cow"] != "A"
        # a track never seen with cam1-1 may still be A (it may be the same cow, later, elsewhere)
        d3 = shared.decide(out(_unit(a)), 0.05, 1020.0, "cam2-7")
        assert d3["cow"] == "A"
        assert store.query("SELECT COUNT(*) FROM events WHERE kind='claim-kept'")[0][0] >= 1


def test_enroll_never_merges_cows_seen_together():
    rng = np.random.default_rng(1)
    x = _unit(rng.normal(size=32))
    y = _unit(x + 0.55 * _unit(rng.normal(size=32)))               # a look-alike, cos ~0.85
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {}, new_cow_min_bursts=4)
        for k in range(6):
            gal.pool.append({"ts": k, "track": "t-x1", "fp": _near(x, rng, 0.02)})
            gal.pool.append({"ts": 100 + k, "track": "t-x2", "fp": _near(x, rng, 0.02)})
            gal.pool.append({"ts": k, "track": "t-y", "fp": _near(y, rng, 0.02)})
        made = gal.enroll(200.0, cannot_link={("t-x1", "t-y")})
        assert len(made) == 2                                      # x's two tracks one cow, y another
        sizes = sorted(len(gal.history[c]) for c in made)
        assert sizes == [6, 12]


def test_enroll_holds_back_when_the_herd_is_full():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {"A": np.eye(8)[0]}, new_cow_min_bursts=3, max_cows=1)
        for k in range(5):
            gal.pool.append({"ts": k, "track": "t1", "fp": _near(np.eye(8)[3], rng, 0.02)})
        assert gal.enroll(10.0) == [] and gal.held_back == 1 and len(gal.pool) == 5


def test_report_follows_cow_aliases():
    import report
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        for k, cow in enumerate(["new-1", "new-1", "A"]):
            store.burst(("cam1", "trk", 100.0 + k, "confirmed", cow, 0.9, 0.2, 0.9, 0.1, "lying", "none",
                         None, 0.5, 100))
        assert report.track_map(store) == {"trk": "new-1"}
        store.alias_cow("new-1", "A", 200.0)
        assert report.track_map(store) == {"trk": "A"}
        store.unalias_cow("new-1")
        assert report.track_map(store) == {"trk": "new-1"}
