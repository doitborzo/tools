# herd — setting up a barn, step by step

From an empty server to daily reports. Example paths: the code in `/opt/herd/tools`,
models and the database in `/srv/herd`. How the system works inside: `GUIDE.md`.

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
- **A better identity model** (public sets with cow ids, `GUIDE.md` 4.9): on a pod
  `bash herd/reid_pod.sh`, then copy `reid/id_model/` (identity.pt, abstain.json) to the server and
  set `[model] identity = "/srv/herd/reid/id_model/identity.pt"`. Do it before the herd is enrolled
  (or move `gallery/` aside and let it enrol again): the fingerprints change.
- **Training on your own cows**: `[training_cache] enabled = true` keeps every 10th
  burst. After a few weeks:
  1. `herd.py barn sheet` — a page of track thumbnails; mark which tracks are one cow;
  2. `herd.py barn build`;
  3. train with `--extra-train`.

  This is the main way to make identification better.

## 11. Maintenance

- **Backups**, daily, of `/srv/herd/barn/`: `herd.sqlite`, `gallery/`, `names.json`. The
  gallery is the memory of the cows — without it the herd has to be enrolled again.
- **Disk**: watch `herd.sqlite` grow, and `training_cache/` if enabled (it has a `max_gb`
  limit).
- **Updating the code**: `git pull` in `/opt/herd/tools`, then `sudo systemctl restart herd`.

## 12. Common problems

- **`cannot open rtsp://…`** — wrong address, login or password, or the camera cannot be
  reached from the server. Check with the same `ffmpeg` command.
- **Lag, missed seconds** (`seconds_missed` in the log) — the server does not keep up.
  Fewer cameras on this server, or turn `tiles` off; check with the stress test (step 5).
- **Almost every cow NaN after a few days** — either the gallery has not enrolled the herd
  yet (see `herd.py gallery`), or the cameras see the cows too small (check `show_zone.py`).
- **A cow counted twice** — neighbouring cameras' masks overlap. Redraw them so that every
  patch of floor is in exactly one mask.
