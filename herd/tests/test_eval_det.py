"""Detector quality and the error with detector misses counted."""
import _paths  # noqa: F401
from eval_det import detector_quality, errors


def test_detector_quality_and_errors():
    cow = lambda x: {"bbox": [x, 0.3, x + 0.15, 0.7]}
    frames = {("1", 2): [cow(0.1), cow(0.3), cow(0.5)], ("1", 3): [cow(0.1)]}
    found = {("1", 2): [cow(0.1)["bbox"], cow(0.31)["bbox"], [0.8, 0.8, 0.9, 0.9]], ("1", 3): []}
    q = detector_quality(frames, found)
    assert q["iou0.5"]["found"] == 2 and q["iou0.5"]["missed"] == 2 and q["iou0.5"]["extra"] == 1
    assert abs(q["iou0.5"]["recall"] - 0.5) < 1e-9 and abs(q["iou0.5"]["precision"] - 2 / 3) < 1e-9
    rows = [{"gt_posture": "lying", "gt_activity": "none", "posture": "lying", "activity": "none"},
            {"gt_posture": "standing", "gt_activity": "feeding", "posture": "standing", "activity": "none"},
            {"gt_posture": "lying", "gt_activity": "none", "posture": None, "activity": None,
             "missed_by_detector": True}]
    e = errors(rows)
    assert abs(e["exact_error"] - 2 / 3) < 1e-9          # the missed cow is an error
    assert abs(e["exact_error_found_cows"] - 0.5) < 1e-9  # and is left out here
    assert e["missed"] == 1
