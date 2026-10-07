"""Spoilt frames, the quality targets, identities from barn tracks."""
import random

import numpy as np

import _paths  # noqa: F401
import degrade
from barn_dataset import identities, resample
from train import auc


def test_spoil_plans_and_kinds():
    rng = random.Random(0)
    crops = np.random.default_rng(0).integers(0, 255, (175, 64, 64, 3), dtype=np.uint8)
    other = np.zeros_like(crops)
    for _ in range(20):
        stretches = degrade.plan(175, rng)
        used = np.zeros(175, int)
        for s, e, k in stretches:
            used[s:e] += 1
            assert k in degrade.KINDS[1:]
        assert used.max() <= 1                                  # stretches do not overlap
        idx, kinds, spoilt = degrade.spoil(crops, stretches, [other], rng)
        assert len(idx) == len(kinds) == len(spoilt) == used.sum()
        assert all(not np.array_equal(crops[i], sp) for i, sp in zip(idx, spoilt))   # every one changed


def test_auc():
    assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auc([0.5, 0.5], [1, 1]) is None


def test_barn_identities_and_resample():
    rows = [{"track": "a", "state": "confirmed", "cow": "c1"}, {"track": "a", "state": "confirmed", "cow": "c1"},
            {"track": "b", "state": "confirmed", "cow": "c1"}, {"track": "c", "state": "unknown", "cow": None}]
    plain = identities(rows)
    assert len({plain["a"], plain["b"], plain["c"]}) == 3                  # a track is a cow
    gal = identities(rows, gallery_labels=True)
    assert gal["a"] == gal["b"] != gal["c"]                                 # the gallery joins a and b
    hand = identities(rows, merges={"c": "tag7", "a": "tag7"})
    assert hand["a"] == hand["c"] == "label:tag7"
    idx = resample(np.arange(140) / 20.0)                                   # a 20 fps camera
    assert len(idx) == 175 and idx[0] == 0 and idx[-1] <= 139
