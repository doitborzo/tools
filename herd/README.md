# herd

Подробная инструкция на русском (что к чему, как работает, что не завершено и как
завершить): [GUIDE.md](GUIDE.md).

The barn system from the brief (up to 60 cows, 4-5 IP cameras looking down,
partly overlapping): identify every cow by itself, record lying, standing,
feeding, drinking, idle time every second and rumination every minute, report
every 3 days and alert the vet. Trained and tested on CBVD-5 until the barn's
own footage exists.

```
cameras (RTSP)  ->  ring buffer per camera (last ~10 s)
   every 1 s    ->  RT-DETRv2 (CNN backbone + transformer) -> camera mask -> tracker
                ->  crop -> DINOv2-S (transformer #1, frozen) -> frame heads: posture, activity
   every 60 s   ->  7 s burst at 25 fps, every cow in view: 175 crops -> DINOv2-S
   (staggered)  ->  temporal transformer (#2) -> per-frame quality + ID
                ->  fingerprint = quality-weighted average; rumination, posture, activity, lameness
                ->  + the rhythm of the crops (motion.py: chewing ~1/s) -> ruminating yes / no
                ->  gallery: k-means prototypes per cow -> confirmed / tentative / unknown (NaN)
SQLite (seconds, bursts) -> track -> cow from confirmed bursts -> per-cow days
                -> 3-day CSV, alerts vs the cow's own week, vet verdicts -> alert precision
```

## Decisions

| Question | Decision |
|---|---|
| Who runs in real time | Small trainable models only (RT-DETRv2, DINOv2-S, a 3-layer temporal transformer); Muse cannot do 5 frames/s (≈2/s measured) |
| Burst | Per camera, not per cow: one 7 s burst gives every cow in view; cameras take turns |
| What comes from where | posture, feeding, drinking: every second; rumination: bursts (one still frame cannot show it); lameness: bursts, once labels exist |
| Rumination in a burst | the temporal model's head plus the rhythm of the burst's own crops (motion.py: per-pixel spectrum over the 7 s, bands around 1 chew a second, per cell of a 4x4 grid); ruminating = p over the cut-off calibrated with the model (heads.json) and not feeding or drinking; written per burst (`bursts.ruminating`), summed into minutes by the reports |
| Identity between bursts | the tracker carries it; a track gets the cow most of its confirmed bursts say (2/3), else its time is NaN |
| k-means | per cow (her own looks: lying, walking, dirty), rebuilt nightly from confirmed bursts only |
| NaN | a small model of p(right) from similarity, margin to the next cow, burst quality, cow size; cut-off so that ≤1% of answers are wrong |
| New cows (no RFID) | unknown bursts are pooled; tracks are grouped; a group seen often becomes `new-<date>-<n>`; first start enrols the herd this way. Tuesday 10:00 starts a change-over; cows unseen 48 h after it retire. `names.json` (optional) maps ids to ear tags for reports |
| Overlapping cameras | each camera's mask covers only the floor it owns; a cow counts where her box centre is |
| Cameras do not see everything | reports carry observed minutes and shares, not just hours |

Open, kept as settings until known: lameness labels (`[lameness] enabled = false`),
which way lying is "worse" (`lying_directions = ["low", "high"]`), coat colours
(affect only how often NaN is said).

## On CBVD-5 now

```bash
bash herd/run_pod.sh          # GPU pod: venv, CBVD-5 with videos, features, train, eval, NaN model
bash herd/run_pod.sh log
```

or by hand:

```bash
python herd/herd.py extract --split train      # 175 crops per burst -> DINOv2 vectors on disk
python herd/herd.py extract --split val
python herd/herd.py motion --split train       # the rhythm of every burst (chewing), CPU only
python herd/herd.py motion --split val
python herd/herd.py train --motion 1           # minutes: no images, no big encoder in the loop
python herd/herd.py abstain                    # NaN model + cut-off -> abstain.json
python cowbench/cowbench.py --out /workspace/herd/run1/eval-val score   # same 2532 cows as the LoRA runs
python herd/herd.py eval-det --run /workspace/herd/run1 # the same on RT-DETRv2's boxes: its misses are errors
python cowbench/cowbench.py --out /workspace/herd/run1/eval-val-det score
```

`eval-det` also sweeps the detector's cut-off and prints the error of the
whole path at each (lower finds more far cows, adds boxes that are no cow):
put the lowest-error one in the barn config (`[detector] threshold`).
`--tiles 1` tries the whole frame plus two square tiles (far cows ~1.8x wider,
~3x the detector's work) on the detector as it is.

Two error figures, as for the LoRA runs: `eval-val` scores the heads on the
annotated boxes (how good they are when the cow is found); `eval-val-det` on
the detector's boxes, a cow it misses counted wrong - the error the barn sees.
`eval_det.json` has the detector's own recall / precision (IoU 0.5 and 0.3, by
cow size) and both errors side by side; `cowbench/summary.py` lists both runs
with "missed by detector" and "error on found cows".

What CBVD-5 can and cannot show:

- posture / activity: the same val keyframes as the LoRA runs, from the once-a-second
  heads only (feeding / drinking / none; a ruminating cow counts as "none" there).
  Rumination stays on the bursts: scored in eval_val.json, reported as minutes from
  bursts, never mixed into the per-frame answers;
- rumination: 7 s of motion per cow, the first time it is learnable at all here;
- identity: CBVD-5 has no cow ids, so a cow within one clip is one identity. That
  teaches "same cow over seconds, other cows apart", not across days or in a top
  view - the barn's own footage is needed for that (and the side-view CBVD-5 is
  not a top view).

## Detector and the once-a-second heads on its boxes

On CBVD-5 val the detector misses about one cow in four (half of the far,
small ones) and draws loose boxes; the heads err more on its boxes (26.6%)
than on the annotation's (21.4%). Two remedies:

```bash
bash herd/run_pod.sh detector    # a new RT-DETRv2 for Full HD: 1088 input (a 1080x1080 tile at ~native
                                 # size), zoom crops in training, whole frame + tiles, chosen by F2
RUN=run6 POS=1 bash herd/run_pod.sh   # step 6: keyframe crops from that detector's boxes (+ jittered
                                      # annotated boxes) -> the frame heads learn on them (--det-keys)
```

Cameras are kept at Full HD (`frame_width = 1920`): a far cow's crop has 1.5x
the pixels it had at 1280.

## One GPU, 5 cameras, in real time

```bash
bash herd/run_pod.sh stress                       # newest run's model, /workspace/lora-runs/detector/best
RUN=run4 CAMERAS=5 DURATION=300 bash herd/run_pod.sh stress
python herd/herd.py stress --model M --detector D --root /workspace/cbvd5 --out OUT   # by hand
```

Two tests on CBVD-5 val videos with the real models: **a burst alone** (7 s at
25 fps, nothing else running: seconds per burst split into detect / crop /
encode / temporal / gallery, cows and crops per burst, one 1 fps tick) and
**live** (5 cameras at once, each a video at its real 25 fps through the same
code as the barn, a tick a second and a burst a minute each, staggered; 30 s
warm-up, 300 s measured). Result: `stress_herd_5cam.md` / `.json` - the detector at its live settings on
the val keyframes from the videos (recall, precision, cows missed, extra boxes,
by size, ms a frame), ticks done,
tick lag p50-p99, bursts done and how late, GPU busy share by part, memory, CPU,
and how many cameras one GPU could take. Keeps up when >= 98% of ticks are done,
tick lag p99 <= 2 s, every burst is done, and each ends before its camera's next.

Live, each camera runs three threads (reader, 1 fps, bursts); the GPU is taken
batch by batch and the 1 fps work goes first, so a burst never holds a tick up
by more than one batch.

## In the barn

```bash
cp herd/config.example.toml barn.toml          # cameras, masks, paths
python herd/herd.py run --config barn.toml                         # live
python herd/herd.py run --config barn.toml --start 2026-10-07T06:00 # recorded files instead of RTSP
python herd/herd.py report --config barn.toml csv                  # last 3 days
python herd/herd.py report --config barn.toml alerts --day 2026-10-07
python herd/herd.py report --config barn.toml verdict --alert 12 --confirmed yes --diagnosis "..."
python herd/herd.py report --config barn.toml alert-stats
python herd/herd.py gallery --config barn.toml
```

CSV columns: `period_start, period_end, cow_id, label, observed_min, lying_min,
standing_min, feeding_min, drinking_min, ruminating_min, idle_min, lying_share,
rumination_share, lameness_score, lameness_bursts, bursts`; the `NaN` row is
time on tracks no cow could be given.

## Files

| | |
|---|---|
| `common.py` | config (TOML over defaults), masks, crops, box interpolation |
| `model.py` | DINOv2 frame encoder, frame heads, temporal transformer, losses |
| `motion.py` | the rhythm of a burst's crops: what rumination (chewing) looks like |
| `degrade.py` | bursts spoilt on purpose (occlusion, mud, blur, dark): what the quality heads learn on |
| `barn_dataset.py` | the barn's own bursts (training cache) -> identity training across tracks and days |
| `cbvd_bursts.py` | CBVD-5 -> bursts + keyframes -> Stage A vectors |
| `train.py` | Stage A training, re-ID / posture / rumination scoring, cowbench export |
| `eval_det.py` | the 1 fps heads on the detector's boxes: detector error included, cut-off sweep |
| `abstain.py` | the NaN model |
| `gallery.py` | prototypes, matching, enrolment, change-over, retirement |
| `pipeline.py` | cameras -> 1 fps + bursts -> store |
| `stress.py` | a burst alone, then N cameras live on one GPU: lag, burst time, GPU share |
| `store.py`, `report.py` | SQLite; per-cow days, CSV, alerts, verdicts |
| `tests/` | `python herd/tests/run_all.py` |

Stage B (LoRA on the frame encoder, end to end on images) is the next step if
Stage A plateaus; it lets the model see finer detail - coat patterns, jaw
movement.
