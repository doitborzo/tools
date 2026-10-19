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

## 6. The barn, live

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

## 7. Other commands

`docker compose run --rm herd help` lists them. Anything `herd.py` takes runs as
`docker compose run --rm herd <command> ...`; `bash` opens a shell in the container.

## Updating

After `git pull`, run `docker compose build` again. Only the code is rebuilt: the models' layers stay
cached unless `docker/models/` changed. Then `docker compose --profile live up -d barn` restarts the
live barn on the new image.
