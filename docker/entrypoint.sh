#!/usr/bin/env bash
# The container's commands: anything herd.py takes, plus a few of its own (help lists them).
set -euo pipefail
H=/opt/herd/herd
PY=python

found() { # every file of this name under /models, newest first (archives unpack at different depths)
    find /models -name "$1" ${2:+-path "$2"} -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2- || true
}

models() {
    echo "Models in this image (/models):"
    found model.pt | sed 's/^/  herd model   : /'
    found det_train_meta.json '*/best/*' | xargs -r -n1 dirname | sed 's/^/  detector     : /'
    found identity.pt | sed 's/^/  identity     : /'
    found rumination.pt | sed 's/^/  rumination   : /'
    "$PY" -c "import torch; print('  GPU          :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none - running on the CPU (slow)')"
    return 0
}

init() { # /data/barn.toml from config.example.toml with this image's models and /data paths filled in; when it
         # is there already, only its model paths that point at nothing in this image are filled in
    local out=/data/barn.toml det ident
    det="$(found det_train_meta.json '*_ft/best/*' | head -1)"
    [ -n "$det" ] || det="$(found det_train_meta.json '*/best/*' | head -1)"
    ident="$(found identity.pt '*id_model_all*' | head -1)"
    [ -n "$ident" ] || ident="$(found identity.pt | head -1)"
    MODEL="$(found model.pt | head -1)" DET="${det%/det_train_meta.json}" IDENT="$ident" \
    RUM="$(found rumination.pt | head -1)" "$PY" - "$H/config.example.toml" "$out" <<'PYEOF'
import os, re, sys
example, out = sys.argv[1], sys.argv[2]
fresh = not os.path.exists(out)
text = open(example if fresh else out, encoding="utf-8").read()
said = []
def put(key, value):
    """key = "value" in place of the example's (or a commented) line, when the value there points at nothing."""
    global text
    m = re.search(rf'^#? ?{key} = "(.*?)"', text, flags=re.M)
    if not value or not m:
        return
    old = m.group(1)
    if not fresh and not m.group(0).startswith("#") and old and os.path.exists(old):
        return
    text = text[:m.start()] + f'{key} = "{value}"' + text[m.end():]
    said.append(f"  {key} = \"{value}\"" + (f"   (was \"{old}\")" if not fresh else ""))
put("checkpoint", os.environ["MODEL"])
put("weights", os.environ["DET"])
put("identity", os.environ["IDENT"])
put("rumination", os.environ["RUM"])
if fresh:
    text = re.sub(r'^path = ".*?herd\.sqlite"', 'path = "/data/barn/herd.sqlite"', text, count=1, flags=re.M)
open(out, "w", encoding="utf-8").write(text)
print(("-> " if fresh else "updated ") + out + " (docker/data/barn.toml on your computer):")
print("\n".join(said) if said else "  its model paths all point at models in this image - nothing changed")
if not os.environ["MODEL"]:
    print("!! no herd model (model.pt) in this image: put herd_models_*.tgz into docker/models/ and build again")
PYEOF
    echo "   Then: [farm] timezone_offset_hours and the cameras (zones page) - docker/README.md section 4."
}

case "${1:-help}" in
    help|-h|--help)
        cat <<'TXT'
herd in Docker. On your computer: docker compose run --rm herd <command>   (from the docker/ folder)
docker/data on your computer is /data here: the config, the recordings, everything written.

  models                         the models in this image, and the GPU
  init                           /data/barn.toml from the example, with the models filled in (run again
                                 on an existing one: fills in model paths that point at nothing)
  zones page --config /data/barn.toml --videos /data/rec --cameras cam1 cam2 cam3 cam4 --out /data/zones
                                 a still of every camera -> zones.html: draw masks / excluded zones
  zones apply --config /data/barn.toml --cameras /data/cameras.toml   the drawn zones into barn.toml
  show-zone --config /data/barn.toml --camera cam1 --image /data/cam1.jpg   a camera's zone drawn
  record --config /data/barn.toml --out /data/rec --minutes 60
                                 every camera at once -> /data/rec/<camera>_<time>.mp4
  fix-videos --videos /data/rec --out /data/rec_fixed
                                 damaged recordings (broken metadata, decoder errors) -> clean .mp4
  barn video --config /data/barn.toml --videos /data/rec --out /data/barn_video
                                 recordings -> tracks for naming the cows
  barn sheet --cache /data/barn_video/training_cache --out /data/barn_video/tracks.html
  barn export --cache /data/barn_video/training_cache --merges /data/merges.csv
  boxes frames --config /data/barn.toml --videos /data/rec --out /data/boxes
  boxes export --data /data/boxes --labels /data/labels.json
  trial --config /data/barn.toml --videos /data/rec_fixed --out /data/trial
                                 a dress rehearsal on the recordings: the gallery fills, then who is who
                                 drawn per camera (trial/show_<cam>/index.html)
  autocalib --config /data/barn.toml --videos /data/rec_fixed --out /data/calib --apply
                                 cameras sharing floor calibrated by the cows moving in both ([mct] link)
  check --config /data/barn.toml   the models on your labels: the detector on herd_det_barn_*.tgz, the
                                 identity model on herd_barn_ids_*.tgz (the newest in /data)
  run --config /data/barn.toml   the barn live (better: docker compose --profile live up -d barn)
  show-ids, report, gallery, stress, eval-det ...   as herd.py (docker compose run --rm herd herd-help)
  bash                           a shell in the container
TXT
        ;;
    herd-help) exec "$PY" "$H/herd.py" --help ;;
    models) models ;;
    init) init ;;
    show-zone) shift; exec "$PY" "$H/show_zone.py" "$@" ;;
    bash|sh) exec "$@" ;;
    python) shift; exec "$PY" "$@" ;;
    *) exec "$PY" "$H/herd.py" "$@" ;;
esac
