"""show-dets: annotations found next to pictures, and what each box is counted as."""
import _paths  # noqa: F401
import os
import tempfile

import show_dets as s


def test_yolo_label_beside_or_in_labels_folder():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        os.makedirs(os.path.join(d, "labels"))
        open(os.path.join(d, "images", "a.jpg"), "wb").close()
        open(os.path.join(d, "labels", "a.txt"), "w").write("0 0.5 0.5 0.2 0.4\n")
        pics, vids = s.find_sources([d])
        assert vids == [] and len(pics) == 1
        assert [round(v, 6) for v in pics[0][1][0]] == [0.4, 0.3, 0.6, 0.7]


def test_score_found_loose_missed_extra():
    gts = [[0.1, 0.1, 0.3, 0.5], [0.5, 0.1, 0.7, 0.5], [0.8, 0.1, 0.95, 0.5]]
    dets = [[0.1, 0.1, 0.3, 0.5, 0.9],        # found
            [0.55, 0.1, 0.78, 0.5, 0.6],      # the same cow, boxes disagree (IoU ~0.45)
            [0.0, 0.7, 0.1, 0.9, 0.5]]        # on no cow
    r = s.score(dets, gts)
    assert [i for i, _, _ in r["good"]] == [0] and [i for i, _, _ in r["loose"]] == [1]
    assert r["extra"] == [2] and r["missed"] == [2]
