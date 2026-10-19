# herd — setting up a barn, step by step

From an empty server to daily reports. Example paths: the code in `/opt/herd/tools`,
models and the database in `/srv/herd`. How the system works inside: `GUIDE.md`.

**Docker Desktop instead** (Windows or Mac, no Python set up): `docker/README.md` — one image with the
code and the models; it replaces steps 1–2 and 7, and every command below runs as
`docker compose run --rm herd <command>` with the paths under `/data`.

## 0. What is needed

- **A Linux server** (Ubuntu 22.04 / 24.04) with an **NVIDIA GPU** and its driver
  (`nvidia-smi` shows the card). How many cameras one card carries is measured in step 5;
  an RTX 2000 Ada kept up with 5 cameras at 51% GPU busy (estimate: ~7).
- **RAM**: about 1.5 GB per camera at Full HD, plus a few GB for the models.
- **Disk**: the database grows every day (a row per cow per second) — plan tens of GB and
  watch its size in the first days.
- **Cameras**: IP cameras with **RTSP**, Full HD and 25 fps if possible, looking down or
  down-and-aside. Neighbouring cameras may overlap: then a cow is not lost between them.
- **Network**: the server reaches every camera's RTSP stream.

## 1. Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # Python manager
export PATH="$HOME/.local/bin:$PATH"
sudo mkdir -p /opt/herd /srv/herd && sudo chown $USER /opt/herd /srv/herd
cd /opt/herd && git clone -b claude/relaxed-hypatia-j4d7aa https://github.com/hitomihiumi/tools.git
uv venv --python 3.12 --seed --managed-python /opt/herd/venv
PY=/opt/herd/venv/bin/python
uv pip install --python $PY torch torchvision --torch-backend=auto
uv pip install --python $PY "transformers>=5.15" opencv-python-headless numpy scipy pillow requests safetensors rfdetr
$PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True and the card's name
```

## 2. Models

Put the `herd_models_*.tgz` archive on the server (`export_pod.sh` makes it on a pod), and
`herd_rumination_*.tgz` if there is a rumination model:

```bash
tar -xzf herd_models_*.tgz -C /srv/herd
tar -xzf herd_rumination_*.tgz -C /srv/herd                  # if there is one
export HF_HOME=/srv/herd/hf-cache HF_HUB_OFFLINE=1           # DINOv2 from the archive, no internet needed
ls /srv/herd/herd/run9                                        # model.pt, abstain.json, heads.json
ls /srv/herd/herd/detector_*/best                             # the detector
```

- the herd transformer: `herd/run9/model.pt` (`abstain.json` and `heads.json` next to it
  are picked up by themselves);
- the detector: `herd/detector_rfdetr_large_zenodo_mmcows_opencows_ft/best`;
- rumination: `rum/model_motion/rumination.pt`.

## 3. Cameras: frames and zones

1. **A still from each camera:**
   ```bash
   ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@10.0.0.11:554/stream1" -frames:v 1 /srv/herd/cam1.jpg
   ```
2. On each still, outline as a polygon the **floor this camera alone counts** (`mask`).
   Coordinates are 0–1: x = pixel / width, y = pixel / height (any image editor shows the
   pixels). **Every patch of floor must be in exactly one mask** — otherwise a cow is
   counted twice or not at all. A neighbouring pen behind a rail goes into `exclude`: it
   is greyed out before the detector.
3. **Check the zone** — which cows are kept and which are not:
   ```bash
   $PY herd/show_zone.py --config /srv/herd/barn.toml --camera cam1 --image /srv/herd/cam1.jpg \
       --detector /srv/herd/herd/detector_rfdetr_large_zenodo_mmcows_opencows_ft/best
   ```
   Green — counted; orange — too small (`min_box_pct` / `min_box_side_px`); red — in the
   excluded zone. Set the size cut so that only cows whose posture can no longer be made
   out are dropped.

## 4. The config `/srv/herd/barn.toml`

Start from `herd/config.example.toml`. What must be filled in:

```toml
[farm]
timezone_offset_hours = 2        # local time
migration_weekday = 1            # the cow change-over day (0 = Monday)
migration_hour = 10
nightly_hour = 2                 # the gallery's nightly maintenance

[model]
checkpoint = "/srv/herd/herd/run9/model.pt"
rumination = "/srv/herd/rum/model_motion/rumination.pt"      # if there is one

[detector]
weights = "/srv/herd/herd/detector_rfdetr_large_zenodo_mmcows_opencows_ft/best"

[identity]
max_cows = 60                    # the herd's size: no more ids are made beyond it

[store]
path = "/srv/herd/barn/herd.sqlite"   # gallery/ and names.json go next to it

[[cameras]]
id = "cam1"
url = "rtsp://user:pass@10.0.0.11:554/stream1"
mask = [[0.0, 0.0], [0.55, 0.0], [0.55, 1.0], [0.0, 1.0]]
# exclude = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.15], [0.0, 0.15]]
# min_box_pct = 0.4

[[cameras]]
id = "cam2"
url = "rtsp://user:pass@10.0.0.12:554/stream1"
mask = [[0.05, 0.0], [1.0, 0.0], [1.0, 1.0], [0.05, 1.0]]
```

Everything else (alerts, identification thresholds, how often bursts are taken) has
working defaults; each key is explained in `config.example.toml`.

## 5. How many cameras the server carries

Record 5–10 minutes from each camera into one folder and run the stress test on them:

```bash
ffmpeg -rtsp_transport tcp -i "rtsp://…/stream1" -t 600 -c copy /srv/herd/rec/cam1.mp4      # each camera
cd /opt/herd/tools/herd
$PY herd.py stress --model /srv/herd/herd/run9/model.pt --detector <best/> --videos /srv/herd/rec \
    --cameras 5 --duration 300 --out /srv/herd/stress
```

The report says whether it keeps up (seconds' lag, burst time, GPU busy). Repeat with a
larger `--cameras` to find the limit.

## 6. A trial run on recordings

In a copy of the config, set each camera's `url` to its recorded file, and give the time
the recording started:

```bash
$PY herd.py run --config /srv/herd/barn_test.toml --start 2026-10-20T06:00:00
$PY herd.py show-ids --config /srv/herd/barn_test.toml --cam cam1 --video /srv/herd/rec/cam1.mp4 \
    --start 2026-10-20T06:00:00 --minutes 10 --out /srv/herd/show_cam1
```

Open `show_cam1/index.html` (or `ids.mp4` in VLC) and check: the boxes sit on the cows,
nothing is counted twice at the mask borders, posture and activity look right. In the
first hours most cows are NaN — that is expected, the gallery is still empty.

## 7. Running for good (systemd)

`/etc/systemd/system/herd.service`:

```ini
[Unit]
Description=herd barn monitoring
After=network-online.target

[Service]
User=herd
WorkingDirectory=/opt/herd/tools/herd
Environment=HF_HOME=/srv/herd/hf-cache HF_HUB_OFFLINE=1
ExecStart=/opt/herd/venv/bin/python herd.py run --config /srv/herd/barn.toml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

`User=herd` is a system user that owns `/srv/herd` and `/opt/herd` — create it, or put your
own user there. Then:

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now herd
journalctl -u herd -f            # the log
```

A camera that drops out does not stop the system: its stream reconnects every 5 s. The
gallery's nightly maintenance runs inside the program — no cron is needed for it.

## 8. The first days: the herd and its names

- The gallery enrols the cows by itself as `new-<date>-<n>`. A group needs ~20 bursts to
  become a new cow; at one burst a minute that takes the first hours to a day.
- What it knows: `$PY herd.py gallery --config /srv/herd/barn.toml`.
- To tie ids to ear tags: `show-ids` writes each cow's id next to her. Find the cow on the
  video, read her tag, and write `/srv/herd/barn/names.json`:
  ```json
  {"new-20261020-03": "UA1234", "new-20261020-07": "UA1251"}
  ```
- If one cow was split into two ids, both still show in the reports. They can be merged
  with `cow_aliases` (`store.alias_cow`; reversible, see `GUIDE.md` 4.8).

## 9. Daily work

```bash
H="$PY /opt/herd/tools/herd/herd.py"
$H report --config /srv/herd/barn.toml alerts --day $(date -d yesterday +%F)   # yesterday's alerts
$H report --config /srv/herd/barn.toml csv                                     # CSV of the last 3 days
$H report --config /srv/herd/barn.toml daily                                   # per cow per day
$H report --config /srv/herd/barn.toml verdict --alert 12 --confirmed yes --diagnosis "ketosis"
$H report --config /srv/herd/barn.toml alert-stats                             # how many alerts were confirmed
```

Alerts start once a cow has 3 or more days of history. To run them automatically, add a
cron line (`crontab -e`), e.g. at 06:30:

```
30 6 * * * cd /opt/herd/tools/herd && /opt/herd/venv/bin/python herd.py report --config /srv/herd/barn.toml alerts --day $(date -d yesterday +\%F) >> /srv/herd/alerts.log
```

The vet's answers (`verdict`) are what tell whether the alerts are worth anything.

## 10. Optional

- **One track across cameras (`[mct]`)** — only after calibration:
  1. put 4 or more marks on the floor in each camera's view (tape, posts);
  2. measure their X, Y in cm from one origin shared by all cameras;
  3. write them into the camera: `floor_points = [[u, v, X, Y], …]`, where u, v is the
     mark's place on the still, 0–1;
  4. set `[mct] enabled = true`, `mode = "link"` (each camera tracks on its own; tracks
     are joined once their floor points agree for a few seconds — the best mode on
     MmCows), and give each camera a `tracking_mask` slightly wider than its `mask`.

  Without calibration do not turn it on: by look alone cows are not passed between
  cameras. How many handovers happened shows in the `track_handovers` table (kind
  `floor`); there is no tool yet to check the calibration itself.
- **A better identity model** (sets with cow ids, `GUIDE.md` 4.9): on a pod, with
  `herd_models_*.tgz` in `/workspace`, `bash herd/reid_pod.sh`; then copy `reid/id_model_all/`
  (identity.pt, abstain.json) to the server and set
  `[model] identity = "/srv/herd/reid/id_model_all/identity.pt"`. Do it before the herd is enrolled
  (or move `gallery/` aside and let it enrol again): the fingerprints change.
- **Training on your own cows** — the main way to make identification better: section 11.

## 11. Your own cows: a labelled dataset

The public sets teach what a cow looks like. Only the barn's own cows teach this barn's cameras, light
and herd — and the check on them is the only honest one. The dataset is made in the barn: the server
keeps bursts, a person names the cows on a page, the names go to the pod with the bursts.

What is kept: with `[training_cache]` on, every 10th burst of every track — its frame vectors (the
encoder's output, not video), the track, the gallery's guess and a 112 px thumbnail. About 250 MB an
hour at 60 cows. Only the vectors leave the barn; the thumbnails stay on the server.

**1. Collect.** In `barn.toml`, then `sudo systemctl restart herd`:

```toml
[training_cache]
enabled = true
every_n = 10          # every 10th burst
max_gb = 200          # it stops keeping at this size
```

Let it run 1–2 weeks. Check: `ls /srv/herd/barn/training_cache` — a folder per day, each with
`index.jsonl` and a folder per camera. With `[mct]` on, one track follows a cow across cameras;
without it, a track is one camera's.

**1, from recordings instead.** Files recorded from the barn's cameras give the same cache without
touching the running barn. Put them in one folder, each named by its camera's id (as in `barn.toml`)
and the local time it started:

```
/srv/herd/rec/cam1_2026-10-20_06-00-00.mp4      or   /srv/herd/rec/cam1/20261020T060000.mp4
/srv/herd/rec/cam2_2026-10-20_06-00-00.mp4
/srv/herd/rec/cam1_2026-10-21_14-00-00.mp4
```

(any of `2026-10-20_06-00-00`, `20261020_060000`, `20261020T060000`; .mp4 .mkv .avi .mov .ts). Then:

```bash
$PY herd.py barn video --config /srv/herd/barn.toml --videos /srv/herd/rec --out /srv/herd/barn_video \
    --gallery /srv/herd/barn/gallery         # optional: the barn's gallery copied in, its guesses on the page
```

It plays the files through the whole pipeline (detector, tracker, cameras joined as `[mct]` says),
every camera's recordings that started together as one session, and keeps **every** burst, one a
camera every 20 s (`--burst-every`). Everything goes to `--out` — its own database, gallery and
`training_cache/`; the barn's are not touched. A file played is not played again, so add recordings
and run it again. It needs the GPU for about as long as the recordings last: run it at a quiet time
(the live barn shares the card) or on a pod. Then use `--cache /srv/herd/barn_video/training_cache`
below. Days count more than hours: a few hours on each of 3–5 days beat a whole single day.

**2. Make the labelling page:**

```bash
cd /opt/herd/tools/herd
$PY herd.py barn sheet --cache /srv/herd/barn/training_cache --out /srv/herd/barn/tracks.html \
    --names /srv/herd/barn/names.json            # --names: the gallery's guess shown as her ear tag
```

Open `tracks.html` in a browser. It reads the thumbnails from `training_cache/` next to it. To label
on another computer, copy the page with the thumbnails only, keeping the layout:

```bash
rsync -a --include='*/' --include='*.jpg' --include='tracks.html' --exclude='*' \
    server:/srv/herd/barn/ ./barn/              # then open barn/tracks.html
```

**3. Name the cows.** One row is one track: the camera, the time, how many bursts, thumbnails spread
over the track, the gallery's guess.

- Type the cow's ear tag (or any name that stays hers) and press **Enter**: the next track.
- One name on many tracks makes them one cow — across cameras and days.
- **= last** repeats the last name. **= UA…/gallery** takes the gallery's guess: check it on the
  thumbnails first.
- **x** leaves the track out: two cows in one track (the tracker swapped them), or no cow.
- **Not sure — leave it empty.** Unnamed tracks are not exported; a wrong name does more harm than
  none.
- Filters: a day, a camera, **only unnamed**, and **find a name** (every track of that cow: check
  they all look alike).
- The line under the buttons: every cow named so far, with her tracks and days.

How to tell the cows apart: the thumbnails are too small for the ear tag, but a Holstein's
black-and-white pattern is her own. Name first the cows you know for sure (a striking pattern), and
the same cows on as many days as you can. When the pattern does not settle it (an all-black cow),
leave the track empty.

The names are kept in this browser while you work, but **Save merges.csv** is the result — press it
often (it downloads `merges.csv`: `identity,track` per line, `x` for a track left out). To go on
later, or on another computer: **Load merges.csv**, or make the page again with
`--merges merges.csv`.

How much: at least 20 cows, each on 3 or more days. A sixth of the cows are held out to check the
model, so with fewer cows the check says little. A cow named on one day only teaches little: the
model must learn the same cow on another day.

**4. Export:**

```bash
$PY herd.py barn export --cache /srv/herd/barn/training_cache --merges merges.csv
```

It writes `herd_barn_feat_<date>.tgz` (frame vectors, no pictures) and says how many cows and bursts
went in and which cows are held out. Held-out cows (a sixth, picked by `--seed`) are never trained on:
they score the model and set its thresholds. Options: `--min-bursts 4` (a cow with fewer is left
out), `--val-share`, `--frames 35` (frames kept per burst).

**5. Train on the pod.** Put `herd_barn_feat_*.tgz` in `/workspace` next to `herd_models_*.tgz`
(and `herd_reid_feat_*.tgz` from an earlier run, if you have it: then no public set is downloaded),
then:

```bash
bash herd/reid_pod.sh          # bash herd/reid_pod.sh log - follow it
```

The log shows `barn: N bursts of the barn's own cows (M of them held out)`. The model trains on the
public sets and the barn's cows together; `eval.json` scores the held-out cows of every set, the
barn's among them. `BARN=0` trains without it. The barn's vectors must come from the same herd model
as the pod's `RUN`; otherwise the set is left out, with a message in the log.

**6. Into the barn.** As in section 10: copy `reid/id_model_all/` (identity.pt, abstain.json,
thresholds.json) to the server, set `[model] identity`, move `gallery/` aside and restart. The gallery
enrols the herd again, so `names.json` has to be written again too.

**7. Again.** Keep the cache on. Every few weeks name the new days (`--merges merges.csv` keeps the
old names), export, train. Each round adds days, and days are what identification needs.

The herd model itself (posture, activity) can also learn from these names: `barn build` and
`train.py --extra-train` (`GUIDE.md` 8.1).

## 12. Your cameras' boxes: a dataset for the detector

The detector was trained on side views (CBVD-5) and public top-down sets. A few hundred frames of the
barn's own cameras with their cow boxes checked by hand teach it this barn: its angle, light, pens
and dirt. It needs the same recordings as above (named by camera and time).

**1. Cut frames, with the detector's guess:**

```bash
$PY herd.py boxes frames --config /srv/herd/barn.toml --videos /srv/herd/rec --out /srv/herd/boxes
```

A frame every 30 s (`--every`), at most 300 a camera (`--max`) spread over all its recordings;
frames where nothing moved are skipped. Each frame is cut as the detector sees it in the barn: Full
HD, the camera's `exclude` zone grey. The current detector draws its boxes on every frame as a
first guess. It writes `frames/`, `frames.json` and the page `boxes.html` into `--out`. Run it again
with more recordings: frames already cut are kept.

**2. Correct the boxes** — open `/srv/herd/boxes/boxes.html` in a browser (on another computer: copy
the whole `boxes/` folder). Green boxes are the detector's; dashed orange ones are cows it was unsure
of.

- **every cow gets a box**, or the part of her that is visible (a cow behind a rail: the visible part;
  half out of the frame: the half inside);
- drag on empty space — a new box; drag a box — move it; drag its corners — resize;
  **Del** — remove the selected box;
- a click on a dashed box keeps it, **A** keeps all of them;
- no boxes in the grey zone (the barn never shows it to the detector);
- **Enter** — the frame is right: marked done, the next one opens. Only frames marked done are
  exported; a frame with no cow at all, marked done, teaches "nothing here" — keep some;
- **S** — skip a frame (spoilt, or you are not sure); **←/→** — move without marking;
  **Ctrl+Z** — undo; "Detector's boxes" — start the frame again;
- filters: one camera, only frames not done.

The work is kept in this browser, but **Save labels.json** is the result — press it often. To go on
later or elsewhere: **Load labels.json** (or `herd.py boxes page --data /srv/herd/boxes --labels
labels.json`).

How much: 200–400 frames over all cameras, from different times of day (morning, night light,
feeding) and days. Boxes drawn tightly, the same way on every frame.

**3. Export:**

```bash
$PY herd.py boxes export --data /srv/herd/boxes --labels labels.json
```

It writes `herd_det_barn_<date>.tgz`: COCO `train/` and `valid/`. A fifth of the frames are held out
to check the detector, by whole hours of a camera (`--block-min 60`), so the check never sees a frame
taken a minute after a training one. With frames from a single hour it stops and asks for more.

**4. Train on the pod.** The detector is fine-tuned together with CBVD-5 (so it keeps what it
knew), so the pod needs CBVD-5 and the current detector:

```bash
bash herd/run_pod.sh setup                        # the environment and CBVD-5, once per pod
tar -xzf /workspace/herd_models_*.tgz -C /workspace   # the current detector -> /workspace/herd/detector_*/
# and herd_det_barn_*.tgz uploaded to /workspace (left packed: the next command takes it in)
DET_ARCH=rfdetr DET_INIT=/workspace/herd/detector_rfdetr_large_zenodo_mmcows_opencows_ft/best \
    EXTRA=zenodo,mmcows bash herd/run_pod.sh detector
bash herd/run_pod.sh log detector                 # follow it
```

The barn's frames join training three times each (`BARN_REPEAT=3`: a few hundred frames against
CBVD-5's thousands). After training, the new and the old detectors are scored on the barn's held-out
frames: `/workspace/extra_det/eval_extra.json`, row `barn`, and a table in the log: recall,
precision and the **best F1 threshold**. The new detector is
`/workspace/herd/detector_rfdetr_large_zenodo_mmcows_barn_ft/best`. (`opencows` is left out of
`EXTRA` above: its server, data.bris, answers 404 since 2026-10.)

**5. Into the barn.** Copy that `best/` folder to the server, set `[detector] weights` to it and
`[detector] threshold` to the barn's best-F1 threshold, check with `show_zone.py` (section 3) and
restart. Only when its recall and precision on `barn` are better than the old detector's.

## 13. Maintenance

- **Backups**, daily, of `/srv/herd/barn/`: `herd.sqlite`, `gallery/`, `names.json`. The
  gallery is the memory of the cows — without it the herd has to be enrolled again.
- **Disk**: watch `herd.sqlite` grow, and `training_cache/` if enabled (it has a `max_gb`
  limit).
- **Updating the code**: `git pull` in `/opt/herd/tools`, then `sudo systemctl restart herd`.

## 14. Common problems

- **`cannot open rtsp://…`** — wrong address, login or password, or the camera cannot be
  reached from the server. Check with the same `ffmpeg` command.
- **Lag, missed seconds** (`seconds_missed` in the log) — the server does not keep up.
  Fewer cameras on this server, or turn `tiles` off; check with the stress test (step 5).
- **Almost every cow NaN after a few days** — either the gallery has not enrolled the herd
  yet (see `herd.py gallery`), or the cameras see the cows too small (check `show_zone.py`).
- **A cow counted twice** — neighbouring cameras' masks overlap. Redraw them so that every
  patch of floor is in exactly one mask.
