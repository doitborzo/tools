# herd — the full guide

For anyone opening this code for the first time: what the system does, which part of
the code does what, how to train and run it, what has been checked, and what is left
to do.

`README.md` is the short overview; this is the long one.

---

## 1. What the system is

A barn of up to 60 cows, 4–5 IP cameras looking down (their views partly overlap),
the video streamed to the cloud. The system must:

- **identify every cow without RFID** — the network names the cows itself, and says
  **NaN** when it is not sure;
- **every second**, record each cow's posture and activity: lying / standing, feeding /
  drinking / nothing;
- **once a minute**, take 7 seconds of video (a burst) and from it detect
  **rumination**, confirm **who the cow is** (gait, coat pattern) and, once labels
  exist, **lameness**;
- **every 3 days**, produce a CSV per cow; **alert** the vet when a cow lies or
  ruminates clearly differently from her usual; collect the **vet's verdicts** and
  count how many alerts were false;
- survive the **Tuesday 10:00** change-over, when some cows leave and new ones arrive.

Until the barn's own footage exists, everything is trained and checked on the public
**CBVD-5** dataset (side-view cow videos, 10 s per clip, posture and activity
annotated on one frame a second). What CBVD-5 cannot provide is built as a ready
scaffold that switches on with the barn's footage (section 8).

---

## 2. Map: requirement → where in the code → state

| Requirement | Where | State |
|---|---|---|
| Find the cows in a frame | RT-DETRv2 detector (`cowbench/detector.py`, class `Live`) | works; on CBVD-5 finds 74% of cows (51% of the far ones), see section 6 |
| Cameras overlap | a mask per camera in the config: a cow counts where her box centre is (`common.in_mask`) | works |
| Far cows are not tried at all | working zone (`common.in_zone`): boxes under `[zone] min_box_pct` (% of the frame) / `min_box_side_px` are dropped before tracking; optional `exclude` polygons are greyed out before the detector | works; `eval-det` suggests both cuts (section 4.0) |
| Follow a cow between frames | `pipeline.Tracker` (box overlap, once a second) | works |
| Posture and activity every second | `model.FrameHeads` on the DINOv2 vector of the crop (`pipeline.Camera.second`) | works; error 2.4–2.6% posture, 17–19% activity (on annotated boxes) |
| A 7 s, 25 fps burst a minute, cameras in turn | `pipeline.Camera.burst`, `step_burst` | works, tested with 5 cameras (section 6) |
| A small Temporal Transformer attends to the burst's frames | `model.TemporalModel` (3 layers, d=256) | works and is trained |
| The cow's embedding (fingerprint) = weighted average over the frames | `quality` head → frame weights → `fingerprint` | works; weights are learnt (directly with `--quality 1`) |
| Good burst / bad burst (mud, hidden behind another cow) | `--quality 1`: spoilt frames (`degrade.py`) teach the frame `quality` head; the NaN model judges the burst from the match and the frame qualities | **built, to be trained on the pod** (section 5, step 6) |
| Gait, identification across days | scaffold `barn_dataset.py` + `train.py --extra-train` | **waits for barn footage** (section 8.1) |
| K-means over the herd's week, the cluster closest to the fingerprint | `gallery.py`: per cow up to 6 k-means centres of her bursts over 7 days | works (tested on synthetic data) |
| NaN when unsure | `abstain.py`: p(right) from similarity, margin, quality; a cut-off | works: on CBVD-5 val answers ~86–91% of the time at ~0.3–1% wrong |
| New cows on Tuesdays 10:00 | `gallery.migration_due`, `enroll`, `retire` | works (synthetic data) |
| Rumination | `rumination` head on the burst + motion rhythm (`motion.py`) | works, but weak: F1 ≈ 0.39 on CBVD-5 (section 8.5) |
| Lameness | `lameness` head exists, **switched off** | **no labels** (section 8.3) |
| Time lying, standing, feeding, drinking, ruminating, idle | `report.py daily`, `store.py` | works |
| CSV every 3 days | `herd.py report csv` | works |
| Alerts: lame cows, lying and rumination deviations | `report.py alerts` (against the cow's own week) | works; which direction of lying is "worse" is a **setting** (section 8.4) |
| Vet verdicts, false alarms | `report.py verdict`, `alert-stats` | works |
| Is one GPU enough | `herd.py stress` | tested: 5 cameras, GPU busy 18% (section 6) |

---

## 3. How it runs in the barn (data flow)

```
IP cameras (RTSP)  ->  per camera a ring buffer of the last ~10 s (Full HD)
                       per camera 3 threads: reader, once-a-second, bursts

once a second (every camera):
   frame -> detector -> camera mask -> tracker -> a 224x224 crop of each cow
         -> DINOv2-S (frozen) -> FrameHeads -> posture, activity -> table seconds

once a minute (cameras in turn, 60/N s apart):
   the buffer's last 7 s (175 frames) -> detector on 5 of every 25 frames -> box chains,
   interpolated in between -> 175 crops per cow
         -> DINOv2-S -> 175 vectors
         -> pixel rhythm (motion.py, on the CPU alongside)
         -> TemporalModel -> weight of every frame, fingerprint (512 numbers),
                             rumination, posture, activity, lameness
         -> gallery: nearest cow + NaN model -> confirmed / tentative / unknown
         -> table bursts (who, ruminating or not, quality ...)
         -> (if enabled) every N-th burst into the training cache

at night (02:00): the gallery rebuilds its k-means centres from confirmed bursts,
                  enrols new cows, retires the ones that left
Tuesday 10:00:    the change-over starts

reports: track -> cow (if >= 2/3 of the track's confirmed bursts name one cow, else NaN)
         -> minutes per cow and day -> 3-day CSV, alerts, verdicts
```

Why it is built this way:

- **Only small models, only on small crops.** The detector and DINOv2-S are light. A
  large vision-language model (Muse) was tried before: about 2 frames a second, too
  slow for 5 cameras.
- **A burst per camera, not per cow:** one 7 s window gives every cow in view at once.
- **The GPU is shared batch by batch,** and the once-a-second work goes first, so a
  large burst never delays a tick by more than one batch (`pipeline.GpuLock`).
- **A cow is counted by one camera:** each camera's mask covers only the floor it owns.
- **Reports do not pretend to see everything:** they carry observed minutes and
  shares, not just hours.

---

## 4. The models one by one

### 4.0 Working zone — far cows are not tried

A bigger detector did not fix far cows: in a side view they are a few dozen pixels,
too small to find reliably, read a posture from or identify. So the system does not
try. The cut is the box's size — it works for any camera and angle, with no line to
draw:

- `[zone] min_box_pct` — boxes smaller than this percentage of the frame are dropped
  before tracking (0.6 = 0.6% of the frame);
- `[zone] min_box_side_px` — and/or boxes whose short side is under this many pixels
  of the kept frame (`frame_width`, Full HD by default): what a crop has to go on;
- both can be set again per camera; a dropped box never becomes a track, a posture,
  an identity or a NaN (counted as `boxes_outside_zone` in the camera stats);
- optional per camera `exclude` polygons, greyed out before the detector, for what
  size alone does not catch (a neighbour's pen behind a rail);
- the camera `mask` (box centre) as before: which camera owns which floor.

Reports show observed minutes, so time a cow spent too far from every camera is "not
observed", not NaN.

How to choose the cut: `herd.py eval-det` prints two sweeps — by % of the frame and by
short side in pixels. Each drops the smallest cows step by step (a tenth more at each
step) and shows the share still covered, the detector's recall on them and the error of
the whole path, and suggests the smallest cut at which the detector finds
`--target-recall` (90%) of the cows left. `--min-box-pct` / `--min-box-side-px` then
score that zone: cows outside it are neither missed nor wrong. A cow is scored only if
clearly inside (1.25x the % cut, 1.12x the pixel cut), so one just over the line whose
detected box is a little smaller is not counted as missed.

To see it: `python herd/show_zone.py --root /workspace/cbvd5 --clip 371 --ts 5
--min-box-pct 0.8 [--detector D]` draws a frame with every cow kept (green) or dropped
as too small (orange; red for an `exclude` zone), above the frame the detector actually
gets; `--config barn.toml --camera cam1 --image cam1.jpg` shows a barn camera's own
zone.

### 4.1 Detector (`cowbench/detector.py`)

RT-DETRv2 (a ResNet-50 CNN + a transformer), fine-tuned on the CBVD-5 boxes. Returns
cow boxes with a score. Settings that matter:

- **threshold** — lower finds more far cows and adds more boxes that are no cow;
  `herd.py eval-det` prints the error of the whole system at each threshold
  (section 6);
- **tiles** — besides the whole frame the detector also looks at two square halves; a
  far cow is 1.8x wider there; about 3x the work;
- a new detector for Full HD: `bash herd/run_pod.sh detector` (1088 input, zoom crops
  in training, threshold chosen by F2 — a missed cow costs more than an extra box).
  It trains the deeper R101 backbone by default (`PekingU/rtdetr_v2_r101vd`, ~76M
  parameters, Apache-2.0, never tried before); `DET_MODEL=PekingU/rtdetr_v2_r50vd`
  trains the R50 (~42M) used so far, for a like-for-like comparison.

RF-DETR (Roboflow; sizes N/S/M/L are Apache-2.0, XL/2XL are not) can replace
RT-DETRv2: `DET_ARCH=rfdetr bash herd/run_pod.sh detector` (`cowbench/rfdetr_det.py`)
trains it on the same CBVD-5 boxes and held-out clips and writes a `best/` folder the
rest of the system loads like any detector (`detector.Live` reads `"kind": "rfdetr"`
from `det_train_meta.json`). Its backbone is DINOv2 too, but it does not replace the
frame encoder below: the detector's features are tuned to find boxes at frame
resolution, where a far cow is a few patches, while the encoder sees each cow at
224x224.

**Box calibration.** CBVD-5's annotators drew boxes with a margin, the detector draws
them tight: for a small far cow the margin alone pushes the overlap under 0.5, so a
cow the detector found counts as missed, and the frame heads (trained on annotated
crops) get tighter crops than they learnt on. `herd.py calibrate --detector D`
measures, on the detector's own held-out train clips (never val), the median ratio of
the annotation's box to the detector's and the shift of its centre, per size of box,
and writes `<detector>/box_calib.json`; `detector.Live` applies it to every box from
then on (pipeline, keys, eval-det, show_zone). `run_pod.sh` does it in step 7
(`CALIB=1`); `eval-det --calib 0` scores the raw boxes for comparison.

**Public cow boxes (`cowbench/extra_det.py`).** Three more sets can join RF-DETR's
training, all downloaded on the pod itself (`fetch` reads the zips through HTTP Range
requests and takes only the frames it needs; a small zip is downloaded whole):

| set | licence | view | what it adds |
|---|---|---|---|
| `zenodo` (HAMK, Zenodo 20426189) | CC BY 4.0 | angled barn camera, 11 videos | 769 frames, ~8.7k whole-cow boxes, crowded stalls |
| `mmcows` (Purdue, MmCows) | CC BY-NC-SA 4.0 | 4 high cameras over one pen, 4480x2800 | one day, a frame per 2 min per camera (`MM_EVERY=8`): ~2.5k frames, whole-cow boxes, day and night |
| `opencows` (Bristol, OpenCows2020) | UK Non-Commercial Government Licence | straight down: indoor walkway, drone over a field | 3.7k real frames; boxes are the body **without the head**, cows cut by the frame edge often unboxed |

Two are non-commercial: fine for herd as a research system, not for a product. Each
set is split into train / valid without neighbouring frames on both sides (zenodo by
video, mmcows by hour, opencows by its authors' fold 0) and stored as COCO (one
class, `cow`), images at most 1920 px on the longer side.

    EXTRA=zenodo,mmcows DET_ARCH=rfdetr bash herd/run_pod.sh detector
    EXTRA=zenodo,mmcows,opencows DET_ARCH=rfdetr bash herd/run_pod.sh detector   # then compare

This downloads the sets to `WORK/extra_det`, trains `detector_rfdetr_large_<sets>`
(train = CBVD-5 + the sets' train; the checkpoint is chosen on CBVD-5's held-out clips
+ the sets' valid; the threshold on CBVD-5 alone), scores it and every other
`detector_*/best` on each set's valid frames (`WORK/extra_det/eval_extra.json`: recall
and precision at IoU 0.5 and 0.3 at the detector's own threshold, best F1), calibrates
its boxes to CBVD-5's habit and runs eval-det on CBVD-5 val. Keep the new detector only
if CBVD-5 val holds up and the other sets improve; opencows' head-less boxes may make
the boxes shorter, which is why it is tried separately. `EXTRA_MAX=N` caps the train
frames taken from each set.

On a new pod, `bash herd/run_pod.sh setup` first (the environment and CBVD-5 only,
steps 1-2), then copy the earlier detector's whole `best/` folder back
(`checkpoint.pth`, `det_train_meta.json`, `box_calib.json`).

**Seeing the boxes.** `herd.py show-dets --src <folder / pictures / videos> --detector
<best/> [--detector <another best/>]` draws every box on every picture into one HTML
gallery (`<out>/index.html`); with an annotation beside the pictures (COCO
`_annotations.coco.json`, or YOLO `.txt` beside them / in `labels/`) found, boxes
disagreeing, missed and extra are coloured as in eval-det, `--pick worst` shows the
pictures with the most misses and extras first, `--cbvd <root>` takes CBVD-5 val.

**Fine-tuning a detector trained before.** `DET_INIT=<its best/>` starts RF-DETR from
that detector's weights instead of rfdetr's COCO ones (same size; its input side is
kept), with fewer epochs and a lower learning rate (`DET_EPOCHS=12`, `DET_LR=5e-5`),
and `_ft` in the tag:

    EXTRA=zenodo,mmcows DET_INIT=/workspace/herd/detector_rfdetr_large/best DET_ARCH=rfdetr bash herd/run_pod.sh detector

A stopped RF-DETR run (Ctrl+C, `stop detector`, a lost pod) goes on from its last
finished epoch when the same command is run again: rfdetr writes `rf/last.ckpt`
after every epoch (`--fresh` starts over). To start a run with other data from a
stopped one instead, point `DET_INIT` at its `rf/checkpoint_best_total.pth`.

CBVD-5 stays in the training set, so the detector does not forget it while it learns
the new sets; it is ~2.5x quicker than training afresh. Compare both ways with
`eval_extra.json` and eval-det.

### 4.2 Frame encoder — DINOv2-S (`model.FrameEncoder`)

Frozen. Turns a 224x224 crop into a vector: CLS + the patch tokens pooled over a
GRID x GRID grid (GRID=2: 384·5 = 1920 numbers). The grid keeps "what is where" in
the crop (head / body). Training ("Stage A") runs on vectors computed once in advance,
which is why it is fast.

### 4.3 Frame heads — `FrameHeads`

Posture (standing / lying) and activity (feeding / drinking / none) from one vector —
the once-a-second path. With `POS=1` they also get the box's position in the frame
(the feed barrier is a place in a fixed camera's picture). With `DET_KEYS=1` they also
learn on the detector's boxes, not only on the annotation's.

### 4.4 Temporal Transformer — `TemporalModel`

Input: the burst's 175 frame vectors + the real time of each frame (a sinusoidal
encoding, so dropped frames do not break the rhythm). Every frame attends to every
other frame of the burst. Outputs:

| Output | What | How it is trained |
|---|---|---|
| `quality` (per frame) | how useful the frame is → frame weights (softmax) | through the ID loss; with `--quality 1` also directly: spoilt frames (`degrade.py`) must score low |
| `frame_ids` (per frame) | one frame's fingerprint (512) | ID loss |
| `fingerprint` | **the weighted average of the frame fingerprints** — the burst's fingerprint | SupCon: two pieces of one burst are one cow, the rest of the batch other cows. With barn data: all bursts of one cow |
| `rumination` | probability of rumination | from the CBVD-5 labels; with `MOTION=1` it also gets the motion rhythm |
| `posture`, `activity` | posture, activity over the burst | from the labels |
| `lameness` | lameness score | **no labels** — off |

### 4.5 Motion rhythm (`motion.py`)

Rumination is chewing about once a second. Over the burst's 175 crops, the spectrum of
every pixel's change (grey, 56x56) over the 7 s is computed; on a 4x4 grid, the power
and the share of motion in bands around 1 Hz are taken. On CBVD-5 (side view, from
afar) it barely helped: the jaw is a couple of pixels.

### 4.6 Gallery (`gallery.py`) — k-means per cow

Every cow has up to 6 k-means centres (spherical, cosine) of her confirmed
fingerprints over the last 7 days: lying, walking, dirty and clean look different, and
one average would blur them. A burst's fingerprint is compared with every centre of
every cow:

- the nearest cow, its similarity, **the margin to the second-best cow**, the burst's
  quality → NaN model → p(right);
- p ≥ cut-off → **confirmed** (the cow's name);
- similar to someone but p below the cut-off → **tentative** (NaN in reports);
- similar to no one → **unknown** → pooled; a group of tracks seen often enough becomes
  a new cow `new-<date>-<n>`.

On the first start the gallery is empty and enrols the whole herd this way. Every
night the centres are rebuilt from confirmed bursts only, so a mistake is not learnt.
From Tuesday 10:00 the gallery expects new cows and retires those not seen for 48 h
after it. `names.json` (optional) maps the names to ear tags in reports.

The difference from the letter of the brief: the brief says "one k-means over the
whole herd". Here every cow has her own k-means, and names come from enrolment — the
same "nearest cluster to the fingerprint", only sturdier.

### 4.7 NaN model (`abstain.py`)

A small logistic regression: p(the answer is right) from the similarity, the margin to
the second cow, frame quality and cow size.
It is fitted on the dev clips (spoilt bursts included, when they have been made). The
cut-off is set so that at most `MAX_ERROR` (1% by default) of the answers given are
wrong. "Not sure" is a NaN in the reports.

### 4.8 One cow, one track across cameras (`mct.py`, `[mct]`)

Without it each camera has its own IoU tracker: a cow walking from one camera to the
next gets a new track, and the two tracks are joined only if the gallery names both.
What holds even without `[mct]`:

- **seen together = two cows.** Two tracks one camera sees together for
  `cannot_link_min_s` (5) seconds with boxes apart (IoU < `cannot_link_max_iou`, 0.3),
  and the chains of one burst, are written to `cannot_link` (`mct.CoSeen`). One second
  was not enough: two boxes on one cow made 3–5% of the pairs wrong on MmCows. A burst's cows are matched to the
  gallery together (best assignment, no cow twice), and a cow confirmed on one track
  is held by it for `claim_s` (120 s): another track seen at the same time gets her
  only with a burst surer by `claim_steal_margin` (0.08), else the next cow
  (`claim-kept` events).
- **new cows** are grouped by average linkage of whole tracks, never joining two
  tracks seen together; with `max_cows` set (the herd's size) no new ids are made at
  the limit — the bursts wait (`held_back_herd_full` in the nightly event).
- a cow's split profile can be merged into another by an alias (`cow_aliases`;
  `store.alias_cow` / `unalias_cow`, reversible); reports follow the aliases.

With `[mct] enabled = true` one `GlobalTracker` serves all cameras. Every box gets a
**frame id** (`model.frame_fingerprint`): the herd model's own temporal transformer
run on a burst of that one frame — a vector in the burst fingerprint's space, with no
extra training and no other dataset. (Later, on the barn's own data, `herd.py
frame-id --features F --out RUN` can teach a small head to do the same faster; the
model then uses the head.) A box gets its global track (`gid`):

1. from the same camera — IoU with the frame id (`w_iou`), and a cow hidden for up to
   `reacquire_s` comes back near where she was if her frame id matches (`reacquire_sim`);
   a cow lying still whose box comes back at the same place (IoU ≥ `still_iou`) within
   `still_s` (60 s) is hers whatever she looks like;
2. **by place on the floor** — with calibrated cameras: per camera `floor_points` (≥ 4
   marks on the floor: where they are in the frame, 0–1, and their measured X, Y in cm,
   the same axes for all cameras; the box's bottom centre is mapped) or `proj_mat`
   (a 3×4 projection matrix, as MmCows has; the box centre at the cow's body height,
   55 cm lying / 90 cm standing by the posture head). A cow another camera saw in the
   last `handover_s`, within `handover_dist` (100 cm) of this box's floor point, the
   next one at least `handover_dist_margin` (50 cm) further. `apart_dist` > 0: cows
   two cameras see that far apart at the same second are two cows (`cannot_link`);
3. from a neighbour (`neighbours` = pairs of cameras with shared floor) — a cow the
   neighbour saw in the last `handover_s`, by frame id ≥ `handover_min_sim` and
   `handover_margin` ahead of the next cow (otherwise a new track: two look-alikes are
   not guessed). `handover_zone` (per camera, optional) is where that may happen;
4. through a portal (`[[mct.portals]]`: a blind passage `from` → `to` taking `min_s` to
   `max_s`) — at `handover_min_sim + portal_extra_sim`;
5. otherwise a new `gid`.

**One canvas (`[mct] mode = "fused"`, calibrated cameras).** Instead of tracking in each
camera and handing over, every second all cameras are put on one floor plan
(`mct.FloorTracker`): each box becomes a floor point, points of different cameras within
`fuse_dist` (160 cm) are one cow — never two of one camera — and one tracker follows the cows
on the floor (gate `fuse_gate` + `max_speed` × time lost; a lost cow keeps her place for
`still_s`). Each box takes its cow's track. The spread is cut where it comes from: a box cut by
the frame's edge weighs `edge_weight`; a point off `floor_area` (the pen's floor, a polygon in
floor units) is a phantom — a box on two cows, a cow of the next pen — and stays with its
camera's own IoU tracking; and the height: a lying cow's centre is ~55 cm up, a standing one's
~90, and a wrong posture throws the point a metre off, so each point is kept at both heights and
two cameras' points are one cow at the height where they agree, the posture head only a prior
(`height_penalty`). On MmCows' labelled boxes with 30% of postures wrong: 1.18 clusters a cow and
13% mixed, against 1.31 and 20% with the posture taken as it comes. Live, the cameras' threads
meet once a second (`mct.FusedSeconds`; a camera down does not hold the others).

**Joined tracks (`[mct] mode = "link"`, calibrated cameras).** On MmCows the one canvas
put a cow seen by two cameras on one track 77% of the time (handover by place: 23%), but its
floor tracker, matching points a second at a time, swapped cows lying a metre or two apart
(33 switches a cow an hour, tracks 50% pure). Within a camera boxes are reliable (IoU tracks
93% pure), so `mct.TrackletLinker` keeps both: each camera follows its own cows, and a track of
one camera is joined to a track of another when for `link_min_s` (5) seconds they were each
other's nearest on the floor within `link_dist` (160 cm, at the height where they agree), in
`link_share` (60%) of the seconds both were seen — never two tracks one camera saw at the same
time. Joins are written to `track_links`; reports read every track under the one it was finally
joined to. MmCows (60 min, 4 cameras): IDF1 over all cameras 0.499 (the system without `[mct]`
0.244, handover by place 0.351), 6.9 tracks a cow (24.2), a cow in two cameras on one track 44%,
but tracks 73% pure - some joins chain wrong cows (A=B, B=C, so A=C). Against that,
`link_margin` (50 cm, on by default): a pair counts as each other's nearest only with the next
cow that much further — wrong joins 2330 → 1309, tracks 78% pure, IDF1 the same (0.498).
`link_contra` (no join putting together two tracks of other cameras seen together
`link_contra_s` seconds yet near in less than that share of them) barely helped and is off.
**For calibrated cameras this is the mode to use** (`mode = "link"`): IDF1 over all cameras
0.244 → 0.498, tracks a cow 24 → 9. With the identity model set `reacquire_sim = 0.9` (its space is tighter: at 0.7 a
camera took back the wrong cow).

**By look alone, cows are not passed between cameras.** MmCows (60 min, 4 cameras):
a cow seen by two cameras at once got one track in 0.1–1.4% of the cases, and every
looser `handover_min_sim` only merged different cows (1285 false merges at 0.5) —
from opposite corners a cow shows another side, and the fingerprint (trained without
cow ids) knows the view, not the cow. By place: one cow seen from two cameras is
within 1 m of herself in 63% of the pairs, two different cows in 0.7%. So: calibrate
the cameras (`floor_points`) before turning `[mct]` on; without calibration keep
handover off (`handover_min_sim = 0.8`).

Each camera then tracks in its `tracking_mask` (overlapping its neighbours; default
its `mask`) but writes seconds and bursts only for boxes in its own `mask`, so a cow in
an overlap is counted once, under one track. Across cameras `cannot_link` is added only
for `disjoint` pairs (cameras with no shared floor): without a floor map, one cow in an
overlap is seen by two cameras at once. Handovers are logged in `track_handovers`.
Not yet: a Kalman filter on the floor point, a nightly merge of split profiles, and
identity training with real cow ids across views (the fingerprint does not tell cows
apart across views today).

**Checking it — MmCows** (`mct_eval.py`, `mct_pod.sh`): four synchronised cameras over
one pen of 16 cows, a frame a second (7/25 12:00–15:00), each cow's number in every
camera every 15 s, the cameras' projection matrices and the cows' positions.
`bash herd/mct_pod.sh` (tmux `herd-mct`) needs only the detector and RUN's `model.pt`
(nothing is trained): `fetch` (`MINUTES=60`, up to 180), `embed` (GPU), `score` and
`show`:

| measure | what it says |
|---|---|
| `idf1_camera` | IDF1 within each camera (one track per cow per camera) |
| `idf1_global` | IDF1 over all cameras together — one track per cow across cameras |
| `tracks_per_cow`, `id_switches_per_cow_hour` | how often a cow's track breaks |
| `cross_cam_same_track` | a cow seen by two cameras at once: share on one track |
| `false_merges` | two different cows on one track at the same second |
| `false_cannot_link` | track pairs marked "two cows" that are one cow |
| `enroll` | new cows from the tracks (7 s pseudo-bursts of frame ids), with and without `cannot_link`: cows made of 16, ids per cow, purity |

The trackers compared: `iou` (the system without `[mct]`), `fid/cam` (GlobalTracker
per camera, as stage B), `fid/still` (+ cows lying still held), `fid/geo` (+ passed on
by the floor point), `fid/geo+apart500` (+ cows 5 m apart are two), `fid/geo@<cm>`
(other `handover_dist`); `--appearance` adds handing over by look, `--dino` DINOv2's
CLS as the vector. The 1 s frames show the neighbouring pens too; each camera is
limited to its pen (the hull of its labelled cows), as a barn camera to its mask.
MmCows is CC BY-NC-SA 4.0 — non-commercial.

**Watching it.** `mct_eval.py show --data D --variant fid/geo --minutes 10` (the pod
runs it on the best tracker) draws the four cameras and the floor, a frame a second:
every box in its track's colour with the track's number (`#12`), `= cow N` — the cow
that track mostly is — and on the labelled seconds `true: cow N`, with a red frame and
`WRONG` when the two differ; on the floor panel the tracks as dots and the true cows
as rings. Out: `show/<variant>/ids.mp4` (MPEG-4: VLC) and `index.html` with a still
every 30 s. For the barn itself: `herd.py show-ids --config barn.toml --cam cam1
--video cam1.mp4 --start <the run's --start>` draws what a run on that recorded file
wrote — each box's cow (or NaN), its track, the last burst decision, posture and
activity — in the cow's colour, the same in every camera.


### 4.9 Identity taught on cows with ids (`reid.py`, `reid_pod.sh`, `[model] identity`)

The herd model's fingerprint learnt on CBVD-5, which names no cows: it tells "the same cow within
7 s" and knows the view more than the cow. `reid.py` teaches a **copy** of its temporal transformer
on three public sets that do name their cows (the herd model itself — posture, activity,
rumination — is not touched):

| set | cows | what | licence |
|---|---|---|---|
| MultiCamCows2024 (Bristol) | 90 | 3 ceiling cameras, 7 days, 101k crops (tracklets at 5 Hz), the same folder number every day | NCGL (non-commercial) |
| Cows2021 (Bristol), identification test set | 181 | top-down crops over 4 weeks | NCGL |
| OpenCows2020, identification part | 46 | top-down crops | NCGL |

- `fetch` reads only the crops from the zips (MultiCamCows: 3.6 GB of 36.5);
- `extract` runs DINOv2 on them as the barn does (square crop, the model's margin and grid) and
  makes pseudo-bursts — 7 s of one camera's tracklet (35 crops at 5 Hz) or a handful of one cow's
  stills of a day — stretched to the barn's 175 frames, each with its cow's id; a sixth of each
  set's cows is held out (never trained on), as the barn's cows are new cows;
- `train`: supervised contrastive over cow ids — one cow on other days and from other cameras
  are positives — with a quarter of the views cut to a single frame (mct.py tracks by one frame
  a second). Scored on the held-out cows: the same cow on another day (`other_day_top1`), from
  another camera the same day (`other_camera_top1`), and from one frame (`frame_other_day_top1`),
  before and after; the NaN model is refitted on these fingerprints (`abstain.json` next to
  `identity.pt`).

`bash herd/reid_pod.sh` does all of it, then MmCows (`mct_pod.sh` with `IDENTITY=` the new model:
the trackers, and `look_cross_cam_top1` — a cow in one camera, the nearest frame id among another
camera's cows at that second is hers). In the barn: `[model] identity = ".../identity.pt"` — the
gallery and mct.py take its fingerprint, the gallery its `abstain.json`; a gallery built with the
old fingerprints must be enrolled again (its prototypes are in the old space). All three sets are
non-commercial: for a commercial product the barn's own data (`[training_cache]`) is the clean path.

**Results of the first model** (held-out cows of the three sets): another day 77.3 → 97.7%, another
camera 61 → 95.2%, one frame 59.8 → 88.7%. But on MmCows a cow seen from one high corner camera was
found among another camera's cows only 17.3% of the time (chance 9.8%): the three sets look straight
down, MmCows (and a barn) looks at an angle, from the side, at cows lying.

**Training on the barn's view** (`reid_pod.sh` trains from scratch — a copy of the herd model's
transformer — on all six sets at once; fine-tuning the first identity model on top, `INIT=`, risks
over-fitting what it has already fitted): three more sources, all with an angled or side view:

| set | cows | what | licence |
|---|---|---|---|
| MmCows, 7/25 (human labels) | 16 | the evaluation's own pen and 4 corner cameras; outside mct_eval's 12:00-15:00; cows 1-10 trained on, **11-16 held out** | CC BY-NC-SA |
| MmCows, other days (`--mm-uwb-days`, default 7/22, 7/28, 8/01, a frame every 30 s) | 10 | labelled by the cows' UWB neck tags (below) | CC BY-NC-SA |
| CoBRA youngstock (Zenodo 15018518) | 48 | heifers from a roof camera, standing / lying left / lying right (the posture is the "day") | CC BY 4.0 |
| SideViewCows2026 (Zenodo 21605650) | 110 | from the side: parlor camera, handheld barn video, snapshots (some lying in cubicles) | CC BY 4.0 |

- **Labels from the neck tags** (`reid.py label`, `tag_tracks`): MmCows' other days have no human
  labels, but cows 1-10 wear a UWB tag whose position (cm, the cameras' world) is logged every 15 s.
  The tag sits on the neck, ~1 m from the body's centre along the stall, and a box's floor point
  is ~1 m off too, so one second is not enough (74-85% right). Per camera: an IoU tracker on the
  detector's boxes (lying cows keep their box for hours); every second the tags in view go to the
  boxes by the Hungarian assignment on floor distance (the distance along the stall forgiven by
  60 cm); a track takes a tag only when that tag won it at 90% of its seconds. Checked on 7/25
  against the human labels: **95% right** (96% at a frame every 15 s), about a seventh of the
  tagged cows' boxes kept; the mistakes are mostly an untagged cow lying next to a tagged one.
  `label` repeats the check on its 7/25 frames and prints it.
- `train` (from scratch, lr 1e-4) takes the top-down and the angled sets together; the best epoch
  is chosen on the held-out cows of every set **and** MmCows' cows 11-16 from another camera, and
  training stops when that has not improved for 6 epochs. `train --init <identity.pt>` (lr 3e-5,
  `INIT=` in the pod script) fine-tunes an existing model instead.
- `mct_eval` now splits its look check: cows 1-10 (trained on, other hours of the day) and cows
  11-16 (never seen) — the second is the honest number for a new barn.

**What that run showed, and what changed.** Trained from scratch on five sets (SideViewCows not yet
in), the model knew a held-out cow on another day 96% of the time and MmCows' cows 11-16 from
another camera 57.5% by burst — but one frame against one frame at the same second (what the
tracker asks) only 15.5% (chance 9.8%), no better than the first model, and its similarities sat
high for every cow (same cow 0.96, two cows 0.9+), so the tracker's fixed thresholds let everything
through: within-camera purity 88 → 77%, IDF1 over all cameras 0.499 → 0.413. Three changes:

- **Frame against frame in training** (`--frame-w 1`, `--moments 8`): besides the bursts, each step
  takes 8 of MmCows' seconds and every cow every camera saw then, one frame each — the same cow from
  another camera at that second is the positive, the cows beside her the negatives (cows 1-10; cows
  11-16 held out). Scored as `moment_cross_cam_top1` (frame vs frame, the look check on held-out
  cows), and the best epoch counts it.
- **Thresholds per model** (`thresholds.json` next to `identity.pt`, `reid.calibrate`): on the held-out
  cows, each threshold sits above 98% of the pairs of two different cows (handover 95%, the gallery's
  merge 99.5%) — `reacquire_sim` (one frame, same camera), `handover_min_sim` (another camera, same
  second), `late_sim` (means of 3 frames), `new_cow_similarity` and `merge_similarity` (bursts); the log
  prints how many same-cow pairs still pass. `pipeline.run` and `mct_eval` take them wherever the
  config keeps the default. The NaN model of a one-day set (MmCows) is now fitted on its other-camera
  queries too.
- **Late reacquire** (`mct.py`, `[mct] late_*`): a new track, once it has 3 frame ids over 3 s, is
  compared by its own fingerprint (the mean of its frame ids) with the tracks its camera lost in the
  60 s before it began, near where it began; the clearly nearest above `late_sim` is the same cow — the
  new gid continues the old one (`track_links`, the reports follow). Off without a calibration;
  `mct_eval` scores `fid/still-nolate` and `fid/link-final-nolate` to compare.

**The result** (MmCows, 60 min, the model trained with the frame loss and its own thresholds):

| | first identity model | from scratch, no frame loss | **with frame loss + thresholds** |
|---|---|---|---|
| by look, cows 11-16 (never seen) | — | 19.6% | **51.1%** |
| by look, all cows (chance 9.8%) | 17.3% | 15.5% | **41.9%** |
| IDF1 in a camera (`fid/still`; first model `~0.9`) | 0.579 | 0.505 | **0.593** |
| track purity in a camera | 88% | 77% | **88%** |
| IDF1 over all cameras, `link-final` at link_dist 120 | 0.499 | 0.388 | 0.496 |
| ... at link_dist 160 | — | 0.411 | **0.540** |
| purity / wrong joins at 120 | 73% / 2330 | 61% / 3269 | 72% / 2095 |
| ... at 160 | — | 62% / 2919 | **77% / 1682** |

The late reacquire changed little (`-nolate`: the same IDF1, purity 1-2 points higher without it);
link_dist 160 is now the default. The first calibration was too strict for the gallery (98% of
two-cow pairs: one cow's bursts grouped only a fifth of the time, most left in the pool) and for the
late check (its same-cow pairs were any two pieces of the day, not a cow lost and back): now
`new_cow_similarity` is the best split of same and two cows (at least above 90% of two cows) and the
late check's positives are a cow's consecutive pieces.
---

## 5. Training on the pod (CBVD-5)

### 5.1 Everything with one command

```bash
cd /tools && git pull
RUN=run7 POS=1 bash herd/run_pod.sh     # runs in tmux session "herd"
bash herd/run_pod.sh log                 # follow the log
```

Every step resumes where it stopped; what is already computed is not recomputed.

| Step | What it does | Output |
|---|---|---|
| 1 | Python environment `/workspace/herd/.venv` | |
| 2 | CBVD-5 with its videos | `/workspace/cbvd5` |
| 3–4 | DINOv2 vectors of every burst crop and annotated keyframe crop | `/workspace/herd/features/{train,val}/<clip>.npz`, `index.jsonl`, `meta.json` |
| 5 | motion rhythm of every burst (CPU) — `MOTION=1` | `<clip>.motion.npz` |
| 6 | spoilt burst frames (GPU) — `QUALITY=1` | `<clip>.degraded.npz` |
| 7 | crops from the detector's boxes — `DET_KEYS=1`, needs a detector | `features/train/keys_aug.*` |
| 8 | training and evaluation | `/workspace/herd/<RUN>/model.pt`, `eval_val.json`, `heads.json`, `eval-val/` |
| 9 | NaN model | `<RUN>/abstain.json` |
| 10 | evaluation on the detector's boxes | `<RUN>/eval_det.json`, `<RUN>/eval-val-det/` |

Knobs (`NAME=value bash herd/run_pod.sh`):

| Knob | Default | Meaning |
|---|---|---|
| `RUN` | run1 | run name (folder) |
| `EPOCHS` | 40 | maximum epochs (early stopping applies) |
| `POS` | 0 | 1: the heads know where the cow is in the frame (1 recommended) |
| `MOTION` | 1 | motion rhythm for rumination |
| `QUALITY` | 1 | frame quality taught directly on spoilt frames |
| `DET_KEYS` | 1 | frame heads also learn on the detector's boxes |
| `CALIB` | 1 | calibrate the detector's boxes to the annotation's habit first (section 4.1) |
| `GRID` | 2 | DINOv2 patch grid (4: finer, features 3.4x larger) |
| `MARGIN` | 0.1 | context around the box in the crop (0.5 was worse: neighbouring cows) |
| `MAX_ERROR` | 0.01 | error budget for NaN |
| `DET` | auto | the detector's `best/` folder (else the newest trained one) |

### 5.2 Other pod commands

```bash
bash herd/run_pod.sh detector    # a new Full HD detector, R101 (hours), then eval-det of the newest run
DET_MODEL=PekingU/rtdetr_v2_r50vd bash herd/run_pod.sh detector   # the same with R50, to compare
DET_SIZE=960 DET_ZOOM=0 DET_TILES=0 DET_SELECT=f1 DET_TAG=r101_960 bash herd/run_pod.sh detector
                                 # R101 with the first detector's settings: only the backbone differs
bash herd/run_pod.sh stress      # 5 cameras in real time on one GPU (~7 min)
CAMERAS=20 RUN=run7 bash herd/run_pod.sh stress
bash herd/run_pod.sh stop
bash herd/run_pod.sh setup       # a new pod: the environment and CBVD-5 only
bash herd/test_pod.sh            # a new pod: upload herd_models_*.tgz to WORK, then the detectors on the
                                 # public sets' fully boxed valid frames + galleries, and the stress test
                                 # (VIDEOS=<barn videos>, else CBVD-5's val clips as cameras), a summary
bash herd/rum_pod.sh             # rumination without CBVD-5: Kaggle + CVB -> DINOv2 every frame -> transformer
bash herd/export_pod.sh          # three archives in WORK/export: models (detectors' best/, RUN's
                                 # model.pt + abstain.json, DINOv2 from the HF cache), results, galleries
```

### 5.3 Single commands (`python herd/herd.py ...`)

| Command | What |
|---|---|
| `extract --split train\|val` | DINOv2 vectors |
| `motion --split ...` | motion rhythm |
| `degrade --split ...` | spoilt frames for `--quality` |
| `keys --split train --detector D` | crops from the detector's boxes for `--det-keys` |
| `train --features F --out R [--pos 1 --motion 1 --quality 1 --det-keys 1] [--extra-train ...]` | training (runs eval at the end) |
| `eval --features F --out R [--extra-val ...]` | evaluate again |
| `abstain --features F --run R` | NaN model |
| `calibrate --detector D` | the detector's boxes reshaped to the annotation's habit (`box_calib.json`) |
| `eval-det --run R [--tiles 1] [--threshold t] [--min-box-pct p] [--min-box-side-px n]` | the whole path on the detector's boxes, threshold sweep, size sweeps for the working zone |
| `stress --model M --detector D` | load test |
| `run --config barn.toml` | run in the barn |
| `report --config barn.toml csv\|alerts\|verdict\|alert-stats\|daily` | reports |
| `gallery --config barn.toml` | the cows the system knows |
| `barn sheet\|build ...` | barn data for training (section 8.1) |

One table of every run: `python cowbench/summary.py /workspace/herd /workspace/lora-runs`.

---

## 6. Reading the results, and what has been achieved

### 6.1 Evaluation files

- `eval_val.json`:
  - `reid_top1_clip`, `reid_top1_all` — was the same cow found again 3.5 s later
    (among the clip's cows / among all);
  - `frame_posture_error`, `frame_activity_error` — once-a-second errors on annotated
    boxes;
  - `rumination_*` — rumination from bursts (`_f1`, `_best_f1`; `_share_true` and
    `_share_called_time` — the true share of rumination and the share reports will
    show);
  - `frame_quality_auc` — does the quality head tell spoilt frames from clean ones
    (1.0 always, 0.5 chance);
  - `reid_spoilt_weighted` and `reid_spoilt_uniform` — re-identification from spoilt
    bursts with the frame weights and with a plain average; the difference is what the
    weights win;
  - `barn_reid_cross_track_top1` — barn data only: identification across tracks and
    days.
- `eval-val/report.md` — the full cowbench report (the same 2532 cows as the LoRA
  runs).
- `eval_det.json` and `eval-val-det/` — the same with the detector's boxes, a missed cow
  counted wrong; plus the detector's recall and precision, recall by cow size, the
  threshold sweep.
- `abstain.json` — how often it answers at which error budget (dev, val, val_spoilt).
- `stress_herd_5cam.md` — whether the GPU keeps up.

### 6.2 Results on CBVD-5 (before `QUALITY` / `DET_KEYS` / the new detector)

| | value |
|---|---|
| Error "posture and activity both right", annotated boxes | 19.7% (run4), 21.4% (run5) |
| Posture / activity | 2.4–2.6% / 17–19% |
| The same on the detector's boxes (misses are errors) | 45.4%; on the cows it found 26.6% |
| Detector recall / precision (IoU 0.5) | 74.4% / 76.0% (at IoU 0.3: 83.7% / 85.6%) |
| Recall by cow size: smallest → largest | 50.7% → 95.6% |
| Same cow found again 3.5 s later (all val cows) | 97.5–98.4% |
| Rumination F1 | ≈0.39 (the motion rhythm did not help) |
| NaN: cut-off for 0.5% wrong on dev → on val wrong / answered | 0.3% / 86% (run2), 1.0% / 91% (run4) |
| 5 cameras, one RTX PRO 6000: keeps up / GPU busy / cameras estimate | yes / 18% / ~22 (1280 px frames, 960 detector without tiles — recheck with Full HD and tiles) |

High re-identification "3.5 s later" does not mean "recognised tomorrow": CBVD-5 has
no cow ids and every cow is in a single 10 s clip. Identification across days can only
be checked on the barn's data.

---

## 7. Running in the barn

Step by step, from an empty server (environment, models, masks, systemd, cron, backups):
`SETUP.md`. In short:

1. `cp herd/config.example.toml barn.toml` and fill in:
   - `[farm]` — time zone, the change-over day and hour;
   - `[[cameras]]` — `id`, `url` (RTSP), `mask` — the polygon of floor that belongs to
     this camera alone (0–1 coordinates). Every patch of floor must be in exactly one
     mask;
   - `[model] checkpoint`, `[detector] weights` (and `threshold` from `eval-det`);
   - the working zone: `[zone] min_box_pct` and/or `min_box_side_px` (from `eval-det`'s
     size sweeps), section 4.0;
   - `[store] path` — where the SQLite file, the gallery and the cache go;
   - `[identity] max_cows` — the herd's size (no new ids beyond it);
   - several cameras over one herd: `[mct]` (section 4.8) — `enabled = true`,
     `neighbours`, a `tracking_mask` per camera reaching into the neighbour's floor,
     `portals` for blind passages.
2. Run: `python herd/herd.py run --config barn.toml`. For recorded files instead of
   RTSP: `--start 2026-10-07T06:00` and `url` = the file's path.
3. In the first days the gallery enrols the herd by itself (`new-<date>-<n>`). For ear
   tags in the reports, create `names.json` next to the store:
   `{"new-2026-10-07-3": "UA1234"}`.
4. Reports:
   ```bash
   python herd/herd.py report --config barn.toml csv                 # the last 3 days
   python herd/herd.py report --config barn.toml alerts --day 2026-10-07
   python herd/herd.py report --config barn.toml verdict --alert 12 --confirmed yes --diagnosis "..."
   python herd/herd.py report --config barn.toml alert-stats         # how many alerts were confirmed
   ```
   CSV columns: `period_start, period_end, cow_id, label, observed_min, lying_min,
   standing_min, feeding_min, drinking_min, ruminating_min, idle_min, lying_share,
   rumination_share, lameness_score, lameness_bursts, bursts`. The `NaN` row is the time
   on tracks no cow could be given.
5. To train on your own cows, set `[training_cache] enabled = true` (section 8.1).

Load: up to ~20 cameras on one RTX PRO 6000 by the GPU estimate; about 1.5 GB of RAM
per camera at Full HD. With `tiles` the detector takes about 3x longer — rerun
`stress`.

---

## 8. What is unfinished and how to finish it

### 8.1 Identification across days, and gait — needs barn footage

**Why it is not done:** CBVD-5 has no cow ids, so the model only learns "the same cow a
few seconds later". That task can be solved by appearance alone, so the model is never
pushed to learn gait.

**What is ready:** the cache in the pipeline, the dataset builder, identity-aware
training, evaluation across tracks and days.

**How to finish:**

1. In `barn.toml`: `[training_cache] enabled = true` (by default every 10th burst,
   ~250 MB an hour, capped at 200 GB). Run for 1–2 weeks.
2. `python herd/herd.py barn sheet --cache <store_dir>/training_cache --out tracks.html` —
   a page per day, one row per track: thumbnails of the cow and the gallery's guess.
3. From the page, write `merges.csv`: lines `identity,track` — tracks that are one cow
   (identity: her ear tag or any name). A few dozen cows over a few days are enough.
   Without `merges.csv` a track counts as a cow by itself (already minutes to hours, in
   different postures).
4. Build the dataset, the last days held out for evaluation:
   ```bash
   python herd/herd.py barn build --cache <...>/training_cache --out /workspace/herd/features_barn \
       --merges merges.csv --val-from 2026-11-01
   ```
5. Train together with CBVD-5 and evaluate:
   ```bash
   python herd/herd.py train --features /workspace/herd/features --out /workspace/herd/run_barn1 \
       --pos 1 --motion 1 --quality 1 --extra-train /workspace/herd/features_barn/train
   python herd/herd.py eval --features /workspace/herd/features --out /workspace/herd/run_barn1 \
       --extra-val /workspace/herd/features_barn/val
   python herd/herd.py abstain --features /workspace/herd/features --run /workspace/herd/run_barn1
   ```
   Watch `barn_reid_cross_track_top1` in `eval_val.json`: the share of bursts whose
   nearest burst from another track (and day) is the same cow.
6. Put the new model in `[model] checkpoint`. Start the gallery afresh by deleting the
   `gallery/` folder next to the store: old fingerprints come from another model and do
   not compare with new ones.

`--gallery-labels 1` also joins tracks the gallery confidently named as one cow. More
data, but the model also learns the gallery's mistakes. Use it only once the NaN
cut-off is strict.

### 8.2 A detector for cameras looking down

**Why:** the detector is trained on CBVD-5 (side view). From above, cows look
different, though they overlap less.

A first step needs no labelling: `EXTRA=...` (section 4.1) adds public sets taken
from above or at a steep angle (MmCows, OpenCows2020). Labelled barn frames are still
the only fair test of the barn's own cameras.

**How to finish:**

1. Cut frames from the barn's videos and draw boxes around the cows in CVAT. The tools
   are in `cowbench/label/`: `prelabel.py` (pre-labels with the current detector),
   `cvat2ava.py` (exports to the CBVD-5 format), and the labelling manual
   `manual_labeling_uk.docx`. A few hundred frames from different cameras and times of
   day are enough.
2. Train: `python cowbench/detector.py train --root <dataset> --out /workspace/herd/detector_top
   --size 1088 --zoom 0.5 --tiles 1 --select f2`.
3. `herd.py eval-det` on the barn's labelled frames → the threshold into the config.
   Then `stress`.

### 8.3 Lameness — needs labels

The `lameness` head exists in the model but is switched off
(`[lameness] enabled = false`): there are no lameness labels in CBVD-5 or anywhere
else yet.

**How to finish:**

1. Collect labels: which cow was lame and when. The vet's locomotion scores (1–5) will
   do; they can be tied to bursts through the verdicts.
2. Add a `lameness` field (a number) to the burst rows, and in `train.py`, next to
   `l_post`, add a loss on `o["lameness"]` (regression; skip rows without a label, as
   `masked_ce` does). Right now this head takes no part in training at all.
3. Set `[lameness] enabled = true` and choose `alert_score`.

### 8.4 Which lying deviation is "worse"

It is not known whether to alert when a cow lies more than usual, less, or both. Right
now alerts are raised both ways: `[alerts] lying_directions = ["low", "high"]`. After a
few weeks of verdicts, check `report alert-stats` and keep the direction that the vet
confirms.

### 8.5 Rumination

On CBVD-5, F1 ≈ 0.39 is this dataset's ceiling: few ruminating bursts in val (~67),
side view from afar. Rumination minutes in reports are currently overstated (≈25–30%
against 15%). Since alerts compare a cow with her own week, a constant overstatement
hurts them less.

**How to finish:** label rumination on the barn's footage (7 s pieces: ruminating /
not; CVAT can do it as an activity) and fine-tune. If the jaw is not visible from
above, feed a larger crop of the head and neck instead of the whole cow.

**A rumination model without CBVD-5 (`herd/rumination.py`, `bash herd/rum_pod.sh`).**
DINOv2-S embeds every frame of a cow's crop (the barn's own encoder settings: CLS +
2 x 2 patch pool, 224 px, margin 0.1); a temporal transformer (`model.TemporalModel`,
real times, so the chewing rhythm is never stretched) reads 7 s of them and says
ruminating or not, the behaviour (rumination, feeding, drinking, resting, moving,
other) and the posture. It learns on two sets at once:

| set | where | what | labels |
|---|---|---|---|
| Kaggle "Beef Cattle Behavior" | barn, 1 camera, 6 cows, day + night | 4,337 clips of one cow, 10 s, 25 fps, 224 px | ruminate 1,544, lie 1,362, stand 638, eat 546, drink 247 |
| CSIRO CVB (CC BY-NC-SA 4.0) | pasture | 502 clips, 15 s, 30 fps, 1080p, every cow boxed and tracked | 11 behaviours per frame, ruminating standing / lying among them |

Validation never shares a recording with training (Kaggle: recording
00000000514000000 and every 5th short sequence; CVB: its authors' lists). Two
variants are trained, frames only and frames + the motion rhythm (motion.py), and
compared per set (`WORK/rum/model*/eval.json`). CVB's cows are small (boxes ~70 px
in 1080p), so its crops are upscaled; its 225k single frames are the slow download
(`CVB_STEP=2` takes 15 fps).

### 8.6 Coat colour and NaN

Solid-coloured cows (no pattern) are harder to identify. The system should say NaN for
them more often by itself: their margin to the second cow and their frame quality are
lower. This can only be checked in the barn: look at `abstain.json` on barn data and at
how often such cows get NaN. If needed, give them a separate error budget.

### 8.7 Feed and water zones

Feeding / drinking depends strongly on where the cow is. With `POS=1` the model learns
this from the box position, but only for the CBVD-5 cameras. For the barn, either label
activities on its own frames (as in 8.5) or add explicit feed-barrier and trough zones
to the camera config (not done).

### 8.8 Stage B — fine-tuning the encoder itself

DINOv2 is frozen for now. If Stage A plateaus on barn data, the next step is LoRA on
the encoder and training end to end from images. Slower (images in the loop), but the
model sees fine detail: coat pattern, the jaw.

---

## 9. Files

| File | What is in it |
|---|---|
| `herd.py` | the one entry point (every command) |
| `common.py` | default config, geometry (masks, IoU, crop, interpolation) |
| `model.py` | DINOv2 encoder, FrameHeads, TemporalModel, losses |
| `cbvd_bursts.py` | CBVD-5 → bursts and keyframes → vectors; commands `extract`, `motion`, `degrade`, `keys` |
| `motion.py` | the burst's motion rhythm (rumination) |
| `degrade.py` | spoiling frames (occlusion, mud, blur, dark) |
| `train.py` | training and evaluation (re-ID, quality, rumination, barn data) |
| `abstain.py` | NaN model |
| `gallery.py` | the cows: k-means centres, matching, enrolment, Tuesday, night |
| `pipeline.py` | real time: cameras, detector, tracker, bursts, cache |
| `mct.py` | one track per cow across cameras (GlobalTracker) |
| `mct_eval.py`, `mct_pod.sh` | that tracker scored on MmCows' four cameras |
| `show_ids.py` | `herd.py show-ids`: who is who drawn on a recorded file (from the store) |
| `reid.py`, `reid_pod.sh` | identity taught on sets with cow ids; MmCows labelled by its neck tags (section 4.9) |
| `store.py` | SQLite: seconds, bursts, events, alerts, verdicts |
| `report.py` | days per cow, CSV, alerts, verdicts |
| `eval_det.py` | the whole path on the detector's boxes, threshold sweep |
| `stress.py` | load test |
| `barn_dataset.py` | the barn cache, the track sheet, building the identity training set |
| `show_crops.py` | a picture of what the model sees (boxes, crops, a burst) |
| `show_zone.py` | a picture of the working zone: which cows are kept, which are not tried |
| `run_pod.sh` | everything on a GPU pod with one command |
| `config.example.toml` | example barn config |
| `tests/` | `python herd/tests/run_all.py` |
| `../cowbench/` | CBVD-5 loading, the detector, cowbench-format reports, `summary.py` |

Store tables (`store.py`):

- `seconds` — track and second: box, posture, activity;
- `bursts` — track and burst: the gallery's decision (`state`, `cow`, `sim`, `margin`,
  `p`), `rumination_p`, `ruminating`, burst posture and activity, lameness, frame
  quality;
- `events` — enrolments, retirements, change-overs, nightly rebuilds, `claim-kept`;
- `cannot_link` — track pairs seen at the same time (two cows); `cow_aliases` — one
  cow's split profile merged into another; `track_handovers` — a track passed between
  cameras (`overlap` / `portal`, with the frame id similarity);
- `alerts`, `verdicts` — alerts and the vet's answers.

---

## 10. Common problems

- **`no trained detector`** — set `DET=/path/to/best` or copy the `detector/best`
  folder (it holds `det_train_meta.json`) under `/workspace`.
- **`--quality 1 needs the spoilt frames`** — step 6 (`herd.py degrade`) has not run.
- **`--motion 1 needs the motion features`** — step 5 has not run.
- **`bursts differ from the extracted ones`** — the features were made from different
  annotations; rerun `extract`.
- **The log shows an old run** — `bash herd/run_pod.sh log` follows the newest log; for
  a given one: `RUN=run7 bash herd/run_pod.sh log`.
- **The cowbench report is stale** — after `eval`, rerun `cowbench.py ... score` and
  `report`.
- **GPU out of memory next to other jobs** — the detector falls back to the CPU (`stress`
  shows it), but is then slow.

## 11. Glossary

| Term | Meaning |
|---|---|
| burst | 7 s of video at 25 fps, once a minute per camera |
| crop | a cow's box cut out and resized to 224x224 |
| fingerprint | 512 numbers describing a cow from a burst; compared by cosine |
| track | the same cow on one camera until the tracker loses her |
| gallery | the known cows and their k-means centres |
| confirmed / tentative / unknown | surely this cow / looks like her but not sure (NaN) / none of the known cows |
| NaN | the system honestly does not know who it is |
| re-ID top-1 | the share of cases where the nearest fingerprint is the same cow |
| AUC | 1.0 separates perfectly, 0.5 is chance |
| IoU | overlap of two boxes (0–1) |
| detector recall / precision | share of cows found / share of boxes that are really cows |
