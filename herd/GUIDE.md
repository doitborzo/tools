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
| Follow a cow between frames | `pipeline.Tracker` (box overlap, once a second) | works |
| Posture and activity every second | `model.FrameHeads` on the DINOv2 vector of the crop (`pipeline.Camera.second`) | works; error 2.4–2.6% posture, 17–19% activity (on annotated boxes) |
| A 7 s, 25 fps burst a minute, cameras in turn | `pipeline.Camera.burst`, `step_burst` | works, tested with 5 cameras (section 6) |
| A small Temporal Transformer attends to the burst's frames | `model.TemporalModel` (3 layers, d=256) | works and is trained |
| The cow's embedding (fingerprint) = weighted average over the frames | `quality` head → frame weights → `fingerprint` | works; weights are learnt (directly with `--quality 1`) |
| Good burst / bad burst (mud, hidden behind another cow) | `--quality 1`: spoilt frames (`degrade.py`) + `burst_quality` head | **built, to be trained on the pod** (section 5, step 6) |
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
                             burst quality, rumination, posture, activity, lameness
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
| `burst_quality` | 0–1: will this burst find its own cow (good / bad burst) | with `--quality 1`: the target is whether the burst's fingerprint found its own cow in the batch |
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
the second cow, frame quality, cow size and (when the model has it) `burst_quality`.
It is fitted on the dev clips (spoilt bursts included, when they have been made). The
cut-off is set so that at most `MAX_ERROR` (1% by default) of the answers given are
wrong. "Not sure" is a NaN in the reports.

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
| `QUALITY` | 1 | good / bad burst taught directly |
| `DET_KEYS` | 1 | frame heads also learn on the detector's boxes |
| `GRID` | 2 | DINOv2 patch grid (4: finer, features 3.4x larger) |
| `MARGIN` | 0.1 | context around the box in the crop (0.5 was worse: neighbouring cows) |
| `MAX_ERROR` | 0.01 | error budget for NaN |
| `DET` | auto | the detector's `best/` folder (else the newest trained one) |

### 5.2 Other pod commands

```bash
bash herd/run_pod.sh detector    # a new Full HD detector (hours), then eval-det of the newest run
bash herd/run_pod.sh stress      # 5 cameras in real time on one GPU (~7 min)
CAMERAS=20 RUN=run7 bash herd/run_pod.sh stress
bash herd/run_pod.sh stop
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
| `eval-det --run R [--tiles 1] [--threshold t]` | the whole path on the detector's boxes, threshold sweep |
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
  - `burst_quality_auc` — does `burst_quality` tell bursts that find their cow from
    those that do not;
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

1. `cp herd/config.example.toml barn.toml` and fill in:
   - `[farm]` — time zone, the change-over day and hour;
   - `[[cameras]]` — `id`, `url` (RTSP), `mask` — the polygon of floor that belongs to
     this camera alone (0–1 coordinates). Every patch of floor must be in exactly one
     mask;
   - `[model] checkpoint`, `[detector] weights` (and `threshold` from `eval-det`);
   - `[store] path` — where the SQLite file, the gallery and the cache go.
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

### 8.6 Coat colour and NaN

Solid-coloured cows (no pattern) are harder to identify. The system should say NaN for
them more often by itself: their margin to the second cow and their `burst_quality` are
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
| `store.py` | SQLite: seconds, bursts, events, alerts, verdicts |
| `report.py` | days per cow, CSV, alerts, verdicts |
| `eval_det.py` | the whole path on the detector's boxes, threshold sweep |
| `stress.py` | load test |
| `barn_dataset.py` | the barn cache, the track sheet, building the identity training set |
| `show_crops.py` | a picture of what the model sees (boxes, crops, a burst) |
| `run_pod.sh` | everything on a GPU pod with one command |
| `config.example.toml` | example barn config |
| `tests/` | `python herd/tests/run_all.py` |
| `../cowbench/` | CBVD-5 loading, the detector, cowbench-format reports, `summary.py` |

Store tables (`store.py`):

- `seconds` — track and second: box, posture, activity;
- `bursts` — track and burst: the gallery's decision (`state`, `cow`, `sim`, `margin`,
  `p`), `rumination_p`, `ruminating`, burst posture and activity, lameness, frame
  quality, `burst_quality`;
- `events` — enrolments, retirements, change-overs, nightly rebuilds;
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
