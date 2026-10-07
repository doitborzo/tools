# herd

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
python herd/herd.py train                      # minutes: no images, no big encoder in the loop
python herd/herd.py abstain                    # NaN model + cut-off -> abstain.json
python cowbench/cowbench.py --out /workspace/herd/run1/eval-val score   # same 2532 cows as the LoRA runs
```

What CBVD-5 can and cannot show:

- posture / activity: the same val keyframes as the LoRA runs, comparable directly;
- rumination: 7 s of motion per cow, the first time it is learnable at all here;
- identity: CBVD-5 has no cow ids, so a cow within one clip is one identity. That
  teaches "same cow over seconds, other cows apart", not across days or in a top
  view - the barn's own footage is needed for that (and the side-view CBVD-5 is
  not a top view).

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
| `cbvd_bursts.py` | CBVD-5 -> bursts + keyframes -> Stage A vectors |
| `train.py` | Stage A training, re-ID / posture / rumination scoring, cowbench export |
| `abstain.py` | the NaN model |
| `gallery.py` | prototypes, matching, enrolment, change-over, retirement |
| `pipeline.py` | cameras -> 1 fps + bursts -> store |
| `store.py`, `report.py` | SQLite; per-cow days, CSV, alerts, verdicts |
| `tests/` | `python herd/tests/run_all.py` |

Stage B (LoRA on the frame encoder, end to end on images) is the next step if
Stage A plateaus; it lets the model see finer detail - coat patterns, jaw
movement.
