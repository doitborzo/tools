#!/usr/bin/env python3
"""v14 (A): every score in one RESULTS.md (and RESULTS.json) for the results tar."""

from __future__ import annotations

import argparse
import json
import os
import sys


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def num(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="the v14 folder (idnet/, assoc/, system/ inside)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    idn = load(os.path.join(a.work, "idnet", "eval_id.json"))
    asc = load(os.path.join(a.work, "assoc", "eval_assoc.json"))
    sys_a = load(os.path.join(a.work, "system", "system_assoc.json"))
    sys_b = load(os.path.join(a.work, "system", "system_iou_per_camera.json"))
    sp = (idn or {}).get("split") or {}
    by_cows = sp.get("by") == "cows"
    ver = "B" if ((sys_a or {}).get("settings") or {}).get("way") == "B" else "A"
    L = [f"# Version 14 ({ver}): cow identification" + (", version 14 (A)'s weights (one barn, MmCows)" if ver == "B" and not idn else ", one barn (MmCows): different cows for training and testing"
                                                   if by_cows else ", trained on MultiCamCows2024 + MmCows"), "",
         "Nothing from CBVD-5. Detector: version 10's, not retrained. DINOv2-S frozen. The ID transformer "
         "and the association transformer trained from scratch.", ""]
    if ver == "B":
        L += ["Version 14 (B) uses version 14 (A)'s weights, nothing retrained. At onboarding it measures the "
              "barn's average cow in the setup hour and takes it out of every fingerprint before the album is built; "
              "the album threshold is set from the setup hour too, without labels (the same track a minute later = "
              "the same cow, two tracks seen together = two cows). The 'not sure' model was fitted on A's "
              "similarities, so B always answers. Way A is scored on the same run.", ""]
    if idn:
        t, prf, alw = idn["test"], idn["test_id_with_not_sure"], idn["test_id_always_answer"]
        if by_cows:
            L += ["## 1. ID transformer on MmCows cows it never saw", "",
                  f"Trained on cows {sp['train']} ({len(sp['train'])}); epoch, 'not sure' model and enrolment "
                  f"threshold picked on cows {sp['val']}; tested on cows {sp['test']} ({len(sp['test'])}, never "
                  f"seen): album from before 12:00, queries every sequence after 15:00 (labelled frames a minute "
                  f"apart). With {len(sp['test'])} cows, guessing gets {1 / max(1, len(sp['test'])):.0%} top-1.", ""]
        else:
            L += ["## 1. ID transformer on MultiCamCows2024 (a day it never saw)", "",
                  f"Trained on {', '.join(sp['train'])}; epoch and 'not sure' model picked on {sp['val']}; "
                  f"album = {sp['val']}, queries = every sequence of {sp['test']}.", ""]
        L += [
              "| measure | value |", "|---|---|",
              f"| top-1 (right cow first) | {pct(t['top1'])} |", f"| top-5 | {pct(t['top5'])} |",
              f"| mAP | {num(t['mAP'])} |", f"| queries | {t['queries']} |",
              f"| ID precision, with 'not sure' | {pct(prf['precision'])} |",
              f"| ID recall ('not sure' = miss) | {pct(prf['recall'])} |",
              f"| **ID F1** | **{pct(prf['f1'])}** |", f"| answered | {pct(prf['answered_share'])} |",
              f"| ID F1, always answering | {pct(alw['f1'])} |",
              f"| model | {idn['model']['parameters'] / 1e6:.2f} M parameters, epoch {idn['model']['epoch']} |", ""]
    if asc:
        w, r = asc["test_window"], asc["baseline_plain_dinov2"]
        L += ["## 2. Association transformer on MmCows (12:00-15:00, never trained on)", "",
              "| measure | transformer | plain DINOv2 similarity |", "|---|---|---|",
              f"| same cow on two cameras: F1 (best threshold) | {num(w['cross_camera']['f1'])} | {num(r['cross_camera']['f1'])} |",
              f"| same: F1 at the threshold picked in training | {num(w['cross_camera_at_train_threshold']['f1'])} | - |",
              f"| same: AUC | {num(w['cross_camera_auc'])} | {num(r['cross_camera_auc'])} |",
              f"| track continued with the right cow (top-1) | {pct(w['track_top1'])} | {pct(r['track_top1'])} |",
              f"| overlap map AUC | {num(w['overlap_auc'])} | - |", ""]
    for name, s in (("association transformer", sys_a), ("baseline: each camera alone, box overlap", sys_b)):
        if not s:
            continue
        d, tr, al = s["detector"], s["tracking"], s["album"]
        i1, i2 = s["identification"]["with_not_sure"], s["identification"]["always_answer"]
        L += [f"## 3. Onboarding test on MmCows, tracking by {name}", "",
              f"Setup {s['settings']['setup_min']} min from {s['settings']['start']:.0f}:00 (album built), then "
              f"identification until {s['settings']['end']:.0f}:00; {s['labelled_sightings']} labelled sightings of "
              f"{s['true_cows']} cows.", "",
              "| measure | value |", "|---|---|",
              f"| detector recall / precision | {pct(d['recall'])} / {pct(d['precision'])} |",
              f"| detector recall, setup hour (never trained on) / test hours | {pct(d['recall_setup_hour'])} / {pct(d['recall_test_hours'])} |",
              f"| **IDF1** (same cow kept the same track) | **{pct(tr['idf1'])}** |",
              f"| track kept: same camera / camera change | {pct(tr['kept_same_camera'])} / {pct(tr['kept_camera_change'])} |",
              f"| tracks | {tr['tracks']} |",
              f"| album cows (true cows) | {al['album_cows']} ({al['true_cows']}) |",
              f"| album purity / coverage / duplicates | {pct(al['purity'])} / {pct(al['coverage'])} / {al['duplicates']} |",
              f"| ID precision / recall, with 'not sure' | {pct(i1['precision'])} / {pct(i1['recall'])} |",
              f"| **ID F1** | **{pct(i1['f1'])}** |",
              f"| ID F1, always answering | {pct(i2['f1'])} |", ""]
        st = s.get("settings") or {}
        if st.get("way"):
            way_txt = {"A": "A (fingerprints as they come)", "B": "B (the barn's average cow taken out first)"}
            L.insert(len(L) - 1, f"| album built by way | {way_txt.get(st['way'], st['way'])}, threshold {st.get('enrol_threshold')} |")
            cal = st.get("self_calibration")
            if cal:
                L.insert(len(L) - 1, f"| threshold from the setup hour (no labels) | same-track pairs {cal.get('same_pairs')} "
                                     f"(10th pct {cal.get('same_p10')}), seen-together pairs {cal.get('different_pairs')} "
                                     f"(99th pct {cal.get('different_p99')}) |")
        o = s.get("other_way")
        if o:
            oa, oi, oi2 = o["album"], o["identification"]["with_not_sure"], o["identification"]["always_answer"]
            L.insert(len(L) - 1, f"| **way {o['way']}, same run**: album cows / purity / coverage | {oa['album_cows']} / "
                                 f"{pct(oa['purity'])} / {pct(oa['coverage'])} |")
            L.insert(len(L) - 1, f"| **way {o['way']}, same run**: ID F1 (always answering) | {pct(oi['f1'])} ({pct(oi2['f1'])}) |")
        for g, v in (s["identification"].get("by_cow_group") or {}).items():
            L.insert(len(L) - 1, f"| ID F1, {g.replace('_', ' ')}: cows {v['cows']} | {pct(v['with_not_sure']['f1'])} "
                                 f"(always answering {pct(v['always_answer']['f1'])}) |")
    L += ["## Notes", "",
          "- MultiCamCows2024 (UK Non-Commercial Government Licence) and MmCows (CC BY-NC-SA 4.0): non-commercial.",
          "- The detector was fine-tuned on MmCows' labelled frames except the hours 0, 6, 12 and 18: its numbers "
          "for 13:00-15:00 may be flattering; the setup hour is clean.",
          ("- One barn: the test cows are new to the ID transformer, but the pen, light and cameras are not; with "
           "few cows, picking the right one is easier than in a real herd. The association transformer trained on "
           "all 16 cows outside 12:00-15:00 (it learns which boxes are one cow, not who the cow is)."
           if by_cows else
           "- MultiCamCows crops are torsos from a walkway; MmCows crops are whole cows in a pen: the ID transformer "
           "meets a new kind of crop in the onboarding test, as it would on a new farm."), ""]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    json.dump({"idnet": idn, "assoc": asc, "system_assoc": sys_a, "system_iou_per_camera": sys_b},
              open(os.path.join(a.out, "RESULTS.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
