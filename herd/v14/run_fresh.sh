#!/usr/bin/env bash
# Is version 14 (E)'s album going stale? freshness.py on MmCows' labelled boxes, with E's ID model and
# with plain DINOv2 - no detector, no tracking, no training. Everything this test needs, on a fresh pod:
#
#   1  Python environment                       3  version 14 (E)'s ID model (WEIGHTS, a *_MODELS.tgz link)
#   2  MmCows: labels, labelled frames, their   4  freshness.py with E's ID model, then with plain DINOv2
#      DINOv2 box vectors (every minute of 25      -> /workspace/FRESHNESS_RESULTS.tar
#      July; every 15 s in 12:00-15:00)
#
#   bash herd/v14/run_fresh.sh          start in tmux session "fresh" (about an hour, most of it MmCows)
#   bash herd/v14/run_fresh.sh log      follow the log
#   bash herd/v14/run_fresh.sh stop
#
# Knobs: WORK=/workspace  WEIGHTS (link to V14E_MODELS.tgz; a Google Drive share link works)
#        ALBUM_MIN=10 (minutes of boxes in one album)  EVERY_MIN=30 (minutes between albums)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
V="$WORK/v14"
VENV="$WORK/herd/.venv"
PY="$VENV/bin/python"
LOG="$V/fresh_log.txt"
SESSION=fresh
WEIGHTS="${WEIGHTS:-https://drive.google.com/file/d/1h9z-GnuaRn9oR4YUpM33odlFJl6c1q81/view?usp=sharing}"
ALBUM_MIN="${ALBUM_MIN:-10}"
EVERY_MIN="${EVERY_MIN:-30}"
export HF_HOME="$WORK/hf-cache"

case "${1:-}" in
    log)  [ -f "$LOG" ] || { echo "no log yet: $LOG"; exit 1; }; exec tail -n 80 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"
          for p in $(pgrep -f '^bash .*run_fresh\.sh' || true); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
          pkill -f 'python.* (data_mmcows|freshness)\.py' 2>/dev/null || true
          exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${FRESH_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    mkdir -p "$V"
    knobs=""
    for v in WORK WEIGHTS ALBUM_MIN EVERY_MIN; do knobs+="$v=$(printf '%q' "${!v}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env FRESH_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/run_fresh.sh"); echo; echo '[run_fresh.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log"
    exit 0
fi

mkdir -p "$V"
trap '' HUP
exec > >(tee -a "$LOG") 2>&1
set -E
trap 'rc=$?; echo "!! line $LINENO failed (exit $rc): $BASH_COMMAND"' ERR
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
again() {   # a download step, tried up to 4 times (Hugging Face sometimes cuts a connection)
    for i in 1 2 3 4; do "$@" && return 0; echo "   stopped (try $i of 4) - again in 30 s"; sleep 30; done; return 1; }

step "1/4  Python environment ($VENV)"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
ready() { [ -x "$PY" ] && "$PY" -c "import torch, transformers, numpy, scipy, PIL; assert torch.cuda.is_available()" 2>/dev/null; }
if ! ready; then
    [ -x "$PY" ] || rm -rf "$VENV"
    [ -d "$VENV" ] || uv venv --python 3.12 --seed --managed-python "$VENV"
    uv pip install --python "$PY" torch torchvision --torch-backend=auto
    uv pip install --python "$PY" "transformers>=5.15" opencv-python-headless numpy scipy pillow requests safetensors
    ready || { echo "!! the venv cannot import torch with CUDA / transformers"; exit 1; }
fi
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
cd "$HERE"

step "2/4  MmCows: labels, labelled frames of 25 July, their DINOv2 box vectors"
again "$PY" data_mmcows.py labels --out "$V/mmcows"
again "$PY" data_mmcows.py frames --out "$V/mmcows"
"$PY" data_mmcows.py feats --out "$V/mmcows"

step "3/4  Version 14 (E)'s ID model: $WEIGHTS"
IDP="$V/fresh_weights/idnet.pt"
if [ ! -f "$IDP" ]; then
    id="$(printf '%s' "$WEIGHTS" | sed -nE 's#.*(/d/|[?&]id=)([A-Za-z0-9_-]{20,}).*#\2#p')"
    url="$WEIGHTS"; [ -n "$id" ] && url="https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
    A="$V/fresh_weights.tgz"
    [ -s "$A" ] || curl -L --fail --retry 5 -o "$A" "$url"
    head -c 512 "$A" | grep -qi '<html' && { rm -f "$A"; echo "!! the link gives a web page: share it as 'Anyone with the link'"; exit 1; }
    rm -rf "$V/fresh_weights_x" && mkdir -p "$V/fresh_weights_x" "$V/fresh_weights" && tar -xzf "$A" -C "$V/fresh_weights_x"
    found="$(find "$V/fresh_weights_x" -name idnet.pt | head -n 1)"
    [ -n "$found" ] || { echo "!! no idnet.pt in the archive:"; find "$V/fresh_weights_x"; exit 1; }
    cp "$found" "$IDP"
fi
"$PY" -c "import torch; ck = torch.load('$IDP', weights_only=False); s = ck.get('split', {}); print('ID model: epoch', ck.get('epoch'), '| split', s.get('by'), '| training cows', len(s.get('train', [])))"

step "4/4  Freshness: albums of $ALBUM_MIN min every $EVERY_MIN min across the day, single labelled boxes"
"$PY" freshness.py --data "$V/mmcows" --idnet "$IDP" --out "$V/freshness_idnet" --album-min "$ALBUM_MIN" --every-min "$EVERY_MIN"
"$PY" freshness.py --data "$V/mmcows" --model none --out "$V/freshness_dinov2" --album-min "$ALBUM_MIN" --every-min "$EVERY_MIN"

R="$WORK/FRESHNESS_RESULTS"
rm -rf "$R"; mkdir -p "$R/idnet" "$R/dinov2"
cp "$V/freshness_idnet/FRESHNESS.md" "$V/freshness_idnet/freshness.json" "$R/idnet/"
cp "$V/freshness_dinov2/FRESHNESS.md" "$V/freshness_dinov2/freshness.json" "$R/dinov2/"
cp "$LOG" "$R/"
sync; sleep 2
tar -cf "$WORK/FRESHNESS_RESULTS.tar" -C "$WORK" FRESHNESS_RESULTS
echo
echo "Done: $WORK/FRESHNESS_RESULTS.tar ($(du -h "$WORK/FRESHNESS_RESULTS.tar" | cut -f1))"
