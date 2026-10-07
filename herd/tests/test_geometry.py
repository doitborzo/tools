"""Masks, box interpolation, the 1 fps tracker."""
import _paths  # noqa: F401
from common import in_mask, interpolate_box
from pipeline import Tracker


def test_mask_ownership():
    left = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]
    right = [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]
    box = [0.40, 0.2, 0.58, 0.4]              # centre x 0.49: the left camera's cow, not the right's
    assert in_mask(box, left) and not in_mask(box, right)
    assert in_mask(box, [])                   # no mask: the whole frame
    assert in_mask(box, [right, left])        # a list of polygons


def test_interpolate():
    keys = {2.0: [0, 0, 1, 1], 4.0: [1, 1, 2, 2]}
    assert interpolate_box(keys, 3.0) == [0.5, 0.5, 1.5, 1.5]
    assert interpolate_box(keys, 0.0) == [0, 0, 1, 1] and interpolate_box(keys, 9.0) == [1, 1, 2, 2]


def test_tracker_keeps_ids():
    t = Tracker("cam1", 0.3, 5)
    a = t.update(0, [[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.8, 0.8]])
    b = t.update(1, [[0.61, 0.6, 0.81, 0.8], [0.11, 0.1, 0.31, 0.3]])
    assert {tid for tid, _ in a} == {tid for tid, _ in b}
    assert t.track_at(0.5, [0.1, 0.1, 0.3, 0.3]) == a[0][0]
    t.update(10, [])                          # gone longer than max_age
    assert not t.tracks
