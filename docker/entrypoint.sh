#!/usr/bin/env bash
# The container's commands: anything herd.py takes, plus a few of its own (help lists them).
set -euo pipefail
H=/opt/herd/herd
PY=python

newest() { # the newest file of these, by time
    ls -t "$@" 2>/dev/null | head -1 || true
}

models() {
    echo "Models in this image (/models):"
    for m in /models/herd/*/model.pt; do [ -f "$m" ] && echo "  herd model   : $m"; done
    for d in /models/herd/detector_*/best; do [ -d "$d" ] && echo "  detector     : $d"; done
    for m in /models/reid/*/identity.pt; do [ -f "$m" ] && echo "  identity     : $m"; done
    for m in /models/rum/*/rumination.pt; do [ -f "$m" ] && echo "  rumination   : $m"; done
    "$PY" -c "import torch; print('  GPU          :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none - running on the CPU (slow)')"
    return 0
}

init() { # /data/barn.toml from config.example.toml, with this image's models and /data paths filled in
    local out=/data/barn.toml
    if [ -f "$out" ]; then echo "$out is there already - not overwritten (delete it to make it again)"; return 0; fi
    local det
    det="$(ls -dt /models/herd/detector_*_ft/best 2>/dev/null | head -1 || true)"
    [ -n "$det" ] || det="$(ls -dt /models/herd/detector_*/best 2>/dev/null | head -1 || true)"
    MODEL="$(newest /models/herd/*/model.pt)" DET="$det" IDENT="$(newest /models/reid/id_model_all/identity.pt /models/reid/*/identity.pt)" \
    RUM="$(newest /models/rum/*/rumination.pt)" "$PY" - "$H/config.example.toml" "$out" <<'PYEOF'
import os, re, sys
src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
def put(pattern, line):
    global text
    text, n = re.subn(pattern, line, text, count=1, flags=re.M)
    assert n, pattern
put(r'^checkpoint = ".*?"', f'checkpoint = "{os.environ["MODEL"]}"')
put(r'^weights = ".*?"', f'weights = "{os.environ["DET"]}"')
put(r'^path = ".*?herd\.sqlite"', 'path = "/data/barn/herd.sqlite"')
if os.environ["IDENT"]:
    put(r'^# identity = ".*?"', f'identity = "{os.environ["IDENT"]}"')
if os.environ["RUM"]:
    put(r'^# rumination = ".*?"', f'rumination = "{os.environ["RUM"]}"')
open(out, "w", encoding="utf-8").write(text)
PYEOF
    echo "-> $out (docker/data/barn.toml on your computer): the models of this image filled in."
    echo "   Now set [farm] timezone_offset_hours and the [[cameras]] (id, url, mask) - SETUP.md sections 3-4."
}

case "${1:-help}" in
    help|-h|--help)
        cat <<'TXT'
herd in Docker. On your computer: docker compose run --rm herd <command>   (from the docker/ folder)
docker/data on your computer is /data here: the config, the recordings, everything written.

  models                         the models in this image, and the GPU
  init                           /data/barn.toml from the example, with the models filled in
  zones page --config /data/barn.toml --videos /data/rec --out /data/zones
                                 a still of every camera -> zones.html: draw masks / excluded zones
  zones apply --config /data/barn.toml --cameras /data/cameras.toml   the drawn zones into barn.toml
  show-zone --config /data/barn.toml --camera cam1 --image /data/cam1.jpg   a camera's zone drawn
  record --config /data/barn.toml --out /data/rec --minutes 60
                                 every camera at once -> /data/rec/<camera>_<time>.mp4
  barn video --config /data/barn.toml --videos /data/rec --out /data/barn_video
                                 recordings -> tracks for naming the cows
  barn sheet --cache /data/barn_video/training_cache --out /data/barn_video/tracks.html
  barn export --cache /data/barn_video/training_cache --merges /data/merges.csv
  boxes frames --config /data/barn.toml --videos /data/rec --out /data/boxes
  boxes export --data /data/boxes --labels /data/labels.json
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
