# herd in Docker Desktop

One image holds the code, its Python environment and the models. Nothing is installed on the computer
besides Docker Desktop. Training stays on a GPU pod (`herd/reid_pod.sh`, `herd/run_pod.sh`). The
image runs everything that happens in the barn:

- the barn live;
- recording the cameras;
- the two dataset tools: naming the cows, and correcting the detector's boxes;
- reports, `show-ids`, `show-zone` and the stress test.

## 1. Docker Desktop

- **Windows** (an NVIDIA card): install a current NVIDIA driver for Windows, then Docker Desktop with
  the WSL 2 backend (its default). Nothing else is needed for the GPU: Docker Desktop passes the card to
  containers.
- **Mac**, or a computer without an NVIDIA card: Docker Desktop alone; use `compose.cpu.yaml` (below).
  Everything runs on the CPU: fine for the labelling pages and a few recordings, too slow for a live
  barn.
- Docker Desktop → Settings → Resources: give it memory. A live barn wants ~1.5 GB a camera plus ~4 GB;
  on Windows the limit is in `%UserProfile%\.wslconfig` (`[wsl2]` → `memory=24GB`).

## 2. Put the models in

Copy the archives into `docker/models/` (the folder this file is in):

| archive | needed | made by |
|---|---|---|
| `herd_models_*.tgz` | yes | `herd/export_pod.sh` on the pod: the herd model, the detectors, DINOv2 |
| `herd_reid_*.tgz` | no | `herd/reid_pod.sh`: the identity model (`reid/id_model_all/`) |
| `herd_rumination_*.tgz` | no | the rumination model |

## 3. Build the image

In a terminal (PowerShell on Windows), in this `docker/` folder:

```bash
docker compose build                                 # an NVIDIA card (RTX 20-40)
TORCH_BACKEND=cu128 docker compose build             # RTX 50 series (PowerShell: $env:TORCH_BACKEND="cu128"; docker compose build)
docker compose -f compose.cpu.yaml build             # no NVIDIA card
```

The build unpacks the models into the image and loads each of them once. The finished image runs
without the internet. The build stops with a message when `docker/models/` has no
`herd_models_*.tgz`. The image is large (the CUDA build is around 10 GB).

Check it:

```bash
docker compose run --rm herd models
```

It lists the models in the image and the GPU. `GPU: none` with an NVIDIA card means the driver or
Docker Desktop does not pass the card: update both, restart Docker Desktop.

With `compose.cpu.yaml`, put `-f compose.cpu.yaml` after `docker compose` in every command below.

## 4. The data folder

`docker/data/` on the computer is `/data` in the container. Everything the commands read or write
lives there:

```
docker/data/barn.toml         the config
docker/data/rec/              the recordings (cam1_2026-10-20_06-00-00.mp4 ...)
docker/data/barn/             the live barn: herd.sqlite, gallery/, names.json
docker/data/barn_video/       recordings played for naming the cows (tracks.html)
docker/data/boxes/            frames for the detector (boxes.html)
```

The config, with the image's model paths filled in:

```bash
docker compose run --rm herd init                    # -> docker/data/barn.toml
                                                     # (again on an existing one: fills in model paths
                                                     #  that point at nothing, e.g. after a new build)
```

Then edit `docker/data/barn.toml`:

- `[farm] timezone_offset_hours`;
- the `[[cameras]]`: `id`, the `url` (`rtsp://...`), the `mask`.

See `herd/SETUP.md` sections 3–4. Paths in it are container paths (`/models/...`, `/data/...`).
Docker Desktop's network reaches cameras on the farm's network as the computer does.

**The zones, drawn on the cameras' pictures.** With recordings in `docker/data/rec/` (named as in
section 5):

```bash
docker compose run --rm herd zones page --config /data/barn.toml --videos /data/rec --cameras cam1 cam2 cam3 cam4 \
    --out /data/zones
```

(`--cameras`: the camera ids, also those not in `barn.toml` yet; the recordings are found by them.)

1. Open `docker/data/zones/zones.html`. It shows each camera's picture with the cows the detector
   found, coloured by the zone rule.
2. Draw:
   - `exclude`: neighbouring pens, the yard, cows that are not this pen's;
   - `mask`: the floor this camera counts;
   - the `url`.
3. **Save cameras.toml** and move the file to `docker/data/`.
4. Apply it:

   ```bash
   docker compose run --rm herd zones apply --config /data/barn.toml --cameras /data/cameras.toml
   ```

A still from a camera, and its zone drawn:

```bash
docker compose run --rm --entrypoint ffmpeg herd -rtsp_transport tcp -i "rtsp://user:pass@10.0.0.11:554/stream1" -frames:v 1 /data/cam1.jpg
docker compose run --rm herd show-zone --config /data/barn.toml --camera cam1 --image /data/cam1.jpg
```

## 5. The datasets

**Recordings.** Every camera at once, an hour:

```bash
docker compose run --rm herd record --config /data/barn.toml --out /data/rec --minutes 60
```

The files are named as the tools read them (`rec/cam1_2026-10-20_06-00-00.mp4`). Recordings made
elsewhere work too: copy them into `docker/data/rec/`, named by camera id and local start time.

**Damaged recordings.** A recorder's file can play in a player and still be read wrong by the tools:
`h264 ... error while decoding` in the log, one camera making no bursts, the clock of a camera standing
still. Encode them again into clean .mp4 (same names, steady frame rate, broken frames dropped) and use
that folder:

```bash
docker compose run --rm herd fix-videos --videos /data/rec --out /data/rec_fixed
```

`--copy` only re-wraps the stream (fast; enough when just the container is broken). Then pass
`--videos /data/rec_fixed` to the commands below (and to `zones page`).

**Naming the cows** (`herd/SETUP.md` section 11):

```bash
docker compose run --rm herd barn video --config /data/barn.toml --videos /data/rec --out /data/barn_video
docker compose run --rm herd barn sheet --cache /data/barn_video/training_cache --out /data/barn_video/tracks.html
```

1. Open `docker/data/barn_video/tracks.html` in your browser. Name the tracks, then **Save
   merges.csv**.
2. Move the saved file from Downloads to `docker/data/merges.csv`.
3. Export:

   ```bash
   docker compose run --rm herd barn export --cache /data/barn_video/training_cache --merges /data/merges.csv
   ```

**The detector's boxes** (`herd/SETUP.md` section 12):

```bash
docker compose run --rm herd boxes frames --config /data/barn.toml --videos /data/rec --out /data/boxes
```

1. Open `docker/data/boxes/boxes.html` in your browser. Correct the boxes, then **Save
   labels.json**.
2. Move the saved file to `docker/data/labels.json`.
3. Export:

   ```bash
   docker compose run --rm herd boxes export --data /data/boxes --labels /data/labels.json
   ```

They write `docker/data/herd_barn_feat_<date>.tgz` and `docker/data/herd_det_barn_<date>.tgz`: these go to
the pod's `/workspace`. Training writes new archives
(`herd_reid_*.tgz`, a detector in `herd_models_*.tgz`). Put those in `docker/models/` (move the old
ones out) and build again.

**Cows named on the boxes** (instead of, or besides, naming tracks). On the box page select a box,
press `I`, type her ear tag, `Enter`: the next box without an id is selected. `P` takes the ids of this
camera's frame before for boxes in the same place (a lying cow keeps her place): check them and fix
the few that moved. The same cow seen by two cameras at once gets the same tag on both: that is what the
model needs most. Save labels.json, then:

```bash
docker compose run --rm herd boxes ids --data /data/boxes --labels /data/labels.json
```

Faster: type each cow's tag once, on one frame of a camera, and press `F`. The tag is carried through
all of that camera's frames, forwards and backwards, while her box clearly continues: the most similar
box on the next frame, and she the most similar to it, not across more than 5 minutes. Filled-in ids
are blue; check them. A typed id is never overwritten; fix a wrong blue one by typing it, and press `F`
again. "only frames I edited" limits it to frames you have opened.

More crops between the cut frames, from the video itself (`--fill 2`: a crop every 2 s between two
frames 10 s apart, where the cow's two boxes overlap, i.e. she stayed in place; her box moved
linearly):

```bash
docker compose run --rm herd boxes ids --data /data/boxes --labels /data/labels.json --fill 2 --config /data/barn.toml --videos /data/rec_fixed
```

It writes `docker/data/herd_barn_ids_<date>.tgz`: each named box cut as the barn cuts a cow, a sixth of
the cows held out. Every frame with named boxes counts, done or not (`--done-only`: only done ones). Put it in the pod's `/workspace`; `reid_pod.sh` adds the set `barn_ids` to training.
The same labels.json also makes the detector's set (`boxes export`).

**One id, one cow on every camera?** Cross-camera training and checking only work when a cow has the
same id on every camera. See it, and fix it:

```bash
docker compose run --rm herd boxes gallery --data /data/boxes --labels /data/labels.json
```

Open `docker/data/boxes/ids.html`: each id is a group, each row in it that id on one camera with her
crops. **match cameras** (e.g. name the cows of cam3 by those of cam1): the left column is the first
camera's cows, the right the second's; click the cow on the right that the selected left one is, or
"new cow". A camera with the same cow under two ids is flagged. Or type in a row the cow it really is (rows with the same name become one cow on every camera;
`x` removes the id), **Save labels.json**, move it to `docker/data/`, then `boxes ids` again (and on
the box page, Load labels.json).

**Check the models on your labels** (before training anything on them - none of these frames was
ever seen by the models):

```bash
docker compose run --rm herd check --config /data/barn.toml
```

It takes the newest `herd_det_barn_*.tgz` and `herd_barn_ids_*.tgz` in `docker/data/` (or `--det`,
`--ids`):

- **the detector** on every labelled frame: the share of the cows it finds and of its boxes that are
  right, at the config's threshold and at the best one for this barn (put that one in
  `[detector] threshold`);
- **the identity model** (and the herd model's own fingerprint, to compare) on the named cows: a cow
  from another camera, is the nearest of all the cows her - from a group of crops and from one crop.

`docker/data/check/check.json` keeps the numbers. After training on the pod, put the new models in
`docker/models/`, build again and run `check` again: the same frames, before and after.

**One camera's zone changed** (`zones page` → `zones apply` again). Redo only that camera:

```bash
docker compose run --rm herd barn video --config /data/barn.toml --videos /data/rec_fixed --out /data/barn_video --cameras cam2
docker compose run --rm herd barn sheet --cache /data/barn_video/training_cache --out /data/barn_video/tracks.html
docker compose run --rm herd boxes frames --config /data/barn.toml --videos /data/rec_fixed --out /data/boxes --every 10 --cameras cam2 --again
```

- `barn video --cameras`: that camera's earlier bursts are removed and its recordings played again;
  the other cameras' bursts and names stay. Its tracks get new names on the page, so name them again.
- `boxes frames --cameras ... --again`: that camera's frames are cut again at the same moments with
  the new grey zone, and the detector looks again. Your boxes on them stay, minus those now in the
  grey. The page marks each of them "cut again - check it": look, press Enter, Save labels.json. Until
  then, `boxes export` leaves them out.

## 6. A dress rehearsal on the recordings

Before the barn goes live: the whole system with `barn.toml` - zones, masks, detector, tracker, posture,
the gallery - played on the recordings:

```bash
docker compose run --rm herd trial --config /data/barn.toml --videos /data/rec_fixed --out /data/trial
```

A gallery fills over a barn's first hours; a quarter of an hour of recordings would end with every
cow NaN. So the files are played twice. The first pass (a burst every 20 s) is followed by the night's
maintenance, which enrols every cow seen in 6 or more bursts. The second pass then runs with those
cows: who is who. It writes:

- `docker/data/trial/show_<cam>/index.html` and `ids.mp4`: every camera's first 10 minutes with each
  box's cow, track, posture and activity (one colour per cow, the same on every camera; grey = NaN).
  Check:
  - every cow has a box (missed ones: the detector, its threshold, `min_box_pct`);
  - no box sits on the neighbouring pen (`exclude`);
  - a cow in the cam1/cam2 overlap is counted by one camera only;
  - a cow keeps her colour while she lies.
- `docker/data/trial/trial.json`, and a line a camera in the log: the cow-seconds counted, the tracks,
  the share given a cow, posture and activity.

**Cameras that share floor, calibrated without measuring anything.** A cow seen by two cameras should
be one cow. `[mct] link` joins her two tracks by where she stands on one floor plan. Marks measured by
hand are not needed: a cow that gets up or walks does it at the same moment on both cameras. The
detector looks at every second of the recordings, on each camera's whole picture (the rehearsal's
store will not do: the masks cut the shared floor away from one of the two cameras). Then
`autocalib` does four things:

- finds those cows and the two cameras' clock difference;
- fits the floor mapping between the cameras and each wide lens's bend;
- checks it on every cow both see;
- writes it all into `barn.toml`.

```bash
docker compose run --rm herd autocalib --config /data/barn.toml --videos /data/rec_fixed --out /data/calib --pairs cam1:cam2 --apply
```

The log says, per camera pair:

- how many moving cows agree;
- what share of the other camera's cows lands on one of this camera's cows (above ~60% is good), and
  what share would by chance alone (in a full pen any point is near some cow): the first has to be
  well above the second;
- "no shared floor" for cameras that share none.

Look at `docker/data/calib/cam1_cam2.jpg`: the other camera's cows (red crosses) should sit on this
camera's cows (green circles). Then run `trial` again: a cow in the overlap should now have one id on
both cameras. The boxes are kept in `docker/data/calib/boxes_<cam>.json`: a second run on the same
recordings skips the detector. `docker/data/calib/cam1_cam2_moving.jpg` is drawn whatever came of
it: both cameras side by side, the paths of the cows that walked (orange), the pairs found (red,
numbered alike) - it shows whether the recording had enough cows walking. If no pair is found, too few cows moved through the overlap: record
an hour or more (the feeding time is best) and run `autocalib` on that. A few minutes of cows lying
is not enough: their boxes only twitch, and a twitch is no evidence of where a cow stands.

**No cows walking: click the same spots on both pictures.** Nothing is measured; you only show
which spot of one picture is which spot of the other:

```bash
docker compose run --rm herd autocalib --config /data/barn.toml --videos /data/rec_fixed --out /data/calib --pairs cam1:cam2 cam3:cam4 --pick
```

1. Open `docker/data/calib/pick.html` in a browser: each camera pair's pictures of one moment side by side.
2. Click a spot on one picture, then the same spot on the other (they get the same number). Spots **on
   the floor**: the foot of a post, a corner of the trough where it stands on the floor, a drain or a
   mark on the floor - not a cow's back or the top rail. 6-10 pairs, spread over the floor both
   cameras see (7 or more: the wide lens's bend is found too). `+` / `-` zoom; click a number to
   remove it.
3. **Save points.json** (it goes to your Downloads) and put it into `docker/data/calib/`.
4. Run:

```bash
docker compose run --rm herd autocalib --config /data/barn.toml --videos /data/rec_fixed --out /data/calib --points /data/calib/points.json --apply
```

The log says per pair how many spots one floor mapping agrees on and how close it puts them to your
clicks (a few px is good). A spot that disagrees with the others (clicked on two different places) is
left out and named; fewer than 5 spots that agree, or spots along one line (one fence), are refused
and nothing is written for that pair. Cameras whose shared floor is only a strip along a fence cannot
be calibrated this way: take that pair out of `[mct] disjoint`, put it in `neighbours`, and let the
gallery join their cows by look.
`calib/cam1_cam2_spots.jpg` shows the spots numbered as on the page: green agree, red are off (with
the px), grey were left out. Fix them on the page (it remembers your clicks), save, run again. Then the check on the cows (share, and what chance alone would give) and the
pictures `calib/cam1_cam2.jpg`, `calib/cam3_cam4.jpg`. With `--apply` the pairs also leave `[mct]
disjoint`; two groups of cameras that share no floor are kept far apart on the plan.

## 7. The barn, live

```bash
docker compose --profile live up -d barn            # starts, and starts again with Docker Desktop
docker compose logs -f barn                         # its log
docker compose --profile live down                  # stop it
```

Reports, while it runs or after:

```bash
docker compose run --rm herd report --config /data/barn.toml csv
docker compose run --rm herd report --config /data/barn.toml alerts --day 2026-10-20
docker compose run --rm herd gallery --config /data/barn.toml
```

`herd/SETUP.md` sections 8–9 say what they mean. In Docker Desktop → Settings → General, turn on
"Start Docker Desktop when you sign in": the barn then comes back by itself after a restart.

## 8. Other commands

`docker compose run --rm herd help` lists them. Anything `herd.py` takes runs as
`docker compose run --rm herd <command> ...`; `bash` opens a shell in the container.

## Updating

After `git pull`, run `docker compose build` again. Only the code is rebuilt: the models' layers stay
cached unless `docker/models/` changed. Then `docker compose --profile live up -d barn` restarts the
live barn on the new image.
