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


def test_detector_tiles_zoom_nms():
    import random
    from PIL import Image
    import detector as d
    assert d.tile_boxes(1920, 1080) == [(0, 0, 1080, 1080), (840, 0, 1920, 1080)]
    assert d.tile_boxes(1080, 1080) == []                          # not wide: nothing to gain
    img, boxes = d.zoom_crop(Image.new("RGB", (1920, 1080)), [[0.1, 0.1, 0.2, 0.3], [0.45, 0.4, 0.55, 0.6]],
                             random.Random(0))
    assert all(0 <= v <= 1 for b in boxes for v in b) and img.size[0] <= 1920
    assert len(d.nms([[0, 0, .5, .5, .9], [.01, 0, .5, .5, .8], [.6, .6, .9, .9, .7]])) == 2

    def fake_predict(model, proc, imgs, keep=0.05):              # one cow in the middle, half a cow at the right edge
        return [[[0.4, 0.4, 0.6, 0.6, 0.9], [0.95, 0.3, 1.0, 0.5, 0.8]] for _ in imgs]
    real, d.predict = d.predict, fake_predict
    try:
        out = d.predict_tiled(None, None, [Image.new("RGB", (1920, 1080))])[0]
    finally:
        d.predict = real
    # the left tile's half cow at its cut is dropped; the same cow from the right tile merges with the whole frame's
    assert len([b for b in out if b[0] > 0.9]) == 1
    assert len(out) == 4                                           # frame centre + one per tile + the edge cow
