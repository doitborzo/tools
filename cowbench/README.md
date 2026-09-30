# cowbench

Measures how well a vision-language model served by vLLM reads cow behaviour on
the [CBVD-5 dataset](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset)
([paper](https://www.nature.com/articles/s41598-024-65953-x)), and writes a
report with the example count, the error rate, the model name, the test date,
the quantization and the dataset link.

## What one "example" is

One annotated cow on one keyframe. The model sees the full 1920x1080 keyframe
with that cow outlined in bright green, and answers two fields.

## Stills or video

Both. `--frames 1` (the default) sends the annotated keyframe from
`labelframes/`. `--frames 5 --span 2.0` decodes that many frames from
`videos/videos/<id>.mp4`, evenly spread over the given number of seconds and
centred on the keyframe, and sends them as a sequence.

The clips are 1920x1080, 25 fps, 10 s. The keyframe
`labelframes/<id>_<t>.jpg` is exactly frame `t*25` of `<id>.mp4` — checked by
pixel comparison, the only difference is JPEG recompression noise — so the two
paths describe the same moment and their numbers are comparable.

Worth doing at least once, because of what `rumination` is: chewing cud, a
movement. On a single still frame a ruminating cow and an idle one are the same
picture, and an error there says more about the frame than about the model. The
cost is that the box is annotated on the middle frame only, so a cow that walks
drifts off its own outline — which is why the span is short by default and the
prompt tells the model to follow the animal rather than the rectangle.

## The annotation is not a 5-way label

CBVD-5 ships AVA-format annotations, which are multi-label: one box appears as
several CSV rows, one per action id.

```
393,4,0.330,0.230,0.417,0.365,2,1   <- lying down
393,4,0.330,0.230,0.417,0.365,5,1   <- rumination     (same box)
```

`labelmap.txt` numbers the classes 1-5 (the dataset card's "0-4" is wrong), and
they form two independent axes:

| axis | values | per box |
|---|---|---|
| posture | `1` stand, `2` lying down | exactly one, always |
| activity | `3` foraging, `4` drinking water, `5` rumination | zero or one |

So the model is asked for a posture *and* an activity, and three error rates are
reported: posture, activity, and exact match. Scoring this as a flat 5-way
problem would mark "lying" wrong on a box annotated `lying + rumination`, which
inflates the error rate for no reason.

## Which split

`annotations/ava_val_v2.1.csv` — 2533 boxes across 50 clips.

`ava_test_v2.1.csv` is `ava_val_v2.1.csv` with every row duplicated: 7522 rows,
the same 2533 boxes, nothing new. It is not used.

One box in val carries both `stand` and `lying down`. There is no correct answer
to grade against, so it is dropped before the run and listed in the report.

## Usage

```bash
pip install -r requirements.txt

python cowbench.py plan  --clips 10          # -> out/manifest.jsonl
python cowbench.py run                       # -> out/results.jsonl
python cowbench.py score --vote              # -> out/metrics-voted.json
python cowbench.py report                    # -> out/report.md
```

Sample **whole clips** (`--clips`), not individual boxes (`--limit`). Per-clip
exact-match error runs from 0% to 100% across the 50 val clips, so the clip is
the unit the variance lives in; and a random subset of boxes shreds the tracks
`--vote` depends on — 4% coverage on a 300-box sample against 92% on a 10-clip
one.

## Free accuracy: voting over the track

The same cow is boxed on up to six keyframes of a clip, one second apart. The
annotation gives no animal identity, but the camera is fixed and overlap links
the boxes — and the labels confirm the links, staying constant along 98% of the
tracks for posture and 91% for activity.

So the model answers the same question up to six times. At temperature 0 the
disagreements are not sampling noise but per-frame difficulty: an animal walking
through, an awkward moment. `score --vote` replaces each answer with the
majority over its track and throws those away. On the full val split:

```
exact-match error   30.8%  ->  28.4%
posture error        9.8%  ->   8.9%
activity error      25.6%  ->  23.5%
        412 tracks, 123 fixed / 52 broken, McNemar p = 8e-8
```

No extra model calls — it is post-processing over `results.jsonl`.

## Comparing two runs

```bash
python cowbench.py compare --a runs/<baseline> --b runs/<variant>
```

Paired McNemar over the shared ids, not two error rates side by side. The runs
answer the same cows, so only the examples that changed carry information about
the difference, and a few hundred examples cannot resolve a few points any
other way. It is what showed that five frames of video did not help: the
apparent doubling of `lying` recall was the model shifting its threshold
(7 -> 17 `lying` answers) while standing recall fell, net p = 1.000.

Clip **371** is the default target: 120 boxes (over the 100 minimum) from a
single clip, and the only val clip containing all five classes —
`stand` 96, `lying` 24, `foraging` 78, `rumination` 18, `drinking` 6.

Only `run` talks to the model. `score` and `report` are pure functions over
`results.jsonl`, so metrics and wording can be reworked without re-querying.
`run` is resumable: it skips ids already present in `results.jsonl`.

### Reaching a model on a pod

```bash
ssh -N -L 8000:127.0.0.1:8000 root@<pod>
python cowbench.py run --base-url http://127.0.0.1:8000 --model muse-glimmer
```

The dataset stays on the local machine; frames go over the tunnel as base64
data URLs. Clip 371 needs six JPEGs, about 440 KiB each after the downscale.

## Options that matter

| flag | default | why |
|---|---|---|
| `--mode` | `marked` | `marked` keeps the scene; `crop` makes the cow bigger but hides the trough, and `drinking` vs `feeding` is decided by what the head is over. `both` sends two images and lets the report separate "cannot see the cow" from "cannot read the scene". |
| `--frames` / `--span` | `1` / `2.0` | `>1` decodes from the mp4 instead of the keyframe; see above |
| `--max-width` | `1920` | the keyframes' native width, i.e. no downscale. Measured against 1280 on 300 paired examples: 21 fixed, 10 broken, and the gain falls off monotonically with box size — the model was short of pixels on distant cows |
| `--min-width` | off | enlarge images narrower than this. A crop of a distant cow is ~113x130px, a few dozen patches for the vision encoder; upscaling adds no information but spends more patches on it, which is the only way to ask for a close look at something small |
| `--temperature` | `0.0` | with `seed=0`, reruns are comparable |
| `--max-tokens` | `4096` | Muse Glimmer reasons before answering and reasons longer with more images. At 2048 a five-frame request is truncated mid-thought and returns nothing; on `finish_reason: length` the client retries once at double the budget |
| `--concurrency` | `4` | vLLM batches these; raise it if the GPU is idle |
| `--limit` / `--seed` | off / `0` | seeded subsample, recorded in the manifest |

## Output

`run` answers through vLLM's structured output (`response_format: json_schema`),
so a malformed answer is not a thing that can happen — a parser would otherwise
be one more source of error indistinguishable from the model being wrong:

```json
{"posture": "standing|lying", "activity": "feeding|drinking|ruminating|none", "confidence": 0.0}
```

`results.jsonl` keeps the raw response and, because Muse Glimmer is served with
`--reasoning-parser`, its `reasoning_content` too — useful for reading back why
a given box went wrong.

`report.md` carries the required header (model, served name, quantization, test
date, dataset link, split, clips, example count, exact-match error rate, vLLM
version, sampling params, image mode, prompt hash) plus:

- three error rates with 95% Wilson confidence intervals;
- the **majority-class baseline** for each axis. 80% of boxes are `standing`, so
  a constant answer already scores well, and an error rate quoted without that
  comparison says nothing;
- per-class precision / recall / F1 and both confusion matrices;
- failed requests and excluded boxes, counted separately and never hidden.

`drinking` has 6 boxes in clip 371 (53 in all of val). Any per-class number for
it is indicative only — that is what the confidence intervals are there to show.

## Two more tests: finding the cows, and many cameras at once

Both run on a plan made with `plan`, like the bench.

**Finding the cows** (`detect.py`). The bench hands the model a box from the
annotation; on a farm nobody does. `detect` sends each keyframe bare and asks
for every cow as `[x1, y1, x2, y2]` on a 0-1000 scale (the model's own
convention: asked for pixels, it answered 0-1000 anyway), with posture and activity.
`detect-score` matches found cows to annotated ones by IoU (0.5 and 0.3) and
reports recall, precision, the count per frame, recall by cow size, the
behaviour error on the cows found, and the end-to-end error: found *and* both
answers right, over every annotated cow. It also shows recall with the boxes
read as pixels or 0-1, in case the model answers another way. Precision
is a lower bound: a cow the annotators skipped counts as an extra.

    python cowbench.py --out out-val detect
    python cowbench.py --out out-val detect-score

**Many cameras** (`stress.py`). `stress --streams 12` runs 12 cameras, each
replaying its own val clip (the busiest clips first). `--task classify` asks
one question per annotated cow, the cows of a frame in parallel - the bench's
question, add `--answer-now` for a LoRA; `--task detect` asks one question per
frame. `--interval 0` measures the most each camera gets; `--interval S` paces
every camera at one frame per S seconds and says whether the server keeps up.

    python cowbench.py --out out-val stress --streams 12 --duration 300
    python cowbench.py --out out-val stress --streams 12 --interval 10 --model pose638 --answer-now --max-width 896
