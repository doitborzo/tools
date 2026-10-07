#!/usr/bin/env bash
# herd on a GPU pod, end to end on CBVD-5:
#
#   environment -> dataset (+ the 10 s videos the bursts are cut from)
#   -> Stage A features (DINOv2-S, train + val) -> train -> eval -> NaN model
#
#   bash herd/run_pod.sh          start in tmux session "herd"
#   bash herd/run_pod.sh log      follow the log
#   bash herd/run_pod.sh stop
#
# Every step resumes: features are written clip by clip, so a rerun after a
# crash continues where it stopped. Knobs:
#   WORK=/workspace   RUN=run1   EPOCHS=40   ENCODER=facebook/dinov2-small   GRID=2
#   MAX_ERROR=0.01    the NaN cut-off: at most this share of wrong IDs among answers

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
RUN="${RUN:-run1}"
EPOCHS="${EPOCHS:-40}"
ENCODER="${ENCODER:-facebook/dinov2-small}"
GRID="${GRID:-2}"
MAX_ERROR="${MAX_ERROR:-0.01}"
DATA="$WORK/cbvd5"
FEAT="$WORK/herd/features"
OUT="$WORK/herd/$RUN"
VENV="$WORK/herd/.venv"
LOG="$WORK/herd/log_$RUN.txt"
SESSION=herd
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cbvd-5cow-behavior-video-dataset"
export HF_HOME="$WORK/hf-cache"

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "$SESSION" 2>/dev/null && echo stopped || echo "not running"; exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    mkdir -p "$WORK/herd"
    knobs=""
    for v in WORK RUN EPOCHS ENCODER GRID MAX_ERROR; do knobs+="$v=$(printf '%q' "${!v}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/run_pod.sh"); echo; echo '[run_pod.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$WORK/herd" "$OUT"
trap '' HUP
exec > >(tee -a "$LOG") 2>&1
set -E
trap 'rc=$?; echo "!! line $LINENO failed (exit $rc): $BASH_COMMAND"' ERR
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }

step "1/6  Python environment ($VENV)"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
uv self update >/dev/null 2>&1 || true
PY="$VENV/bin/python"
ready() { [ -x "$PY" ] && "$PY" -c "import torch, torchvision, transformers, cv2, numpy, scipy, PIL; assert torch.cuda.is_available()" 2>/dev/null; }
if ! ready; then
    [ -x "$PY" ] || rm -rf "$VENV"
    [ -d "$VENV" ] || uv venv --python 3.12 --seed --managed-python "$VENV"
    uv pip install --python "$PY" torch torchvision --torch-backend=auto
    uv pip install --python "$PY" "transformers>=5.15" opencv-python-headless numpy scipy pillow requests safetensors
    ready || { echo "!! the venv cannot import torch with CUDA / transformers / cv2"; exit 1; }
fi
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

step "2/6  CBVD-5 with its videos"
# The LoRA runs only needed keyframes; bursts are cut from the 10 s, 25 fps clips.
if [ ! -f "$DATA/annotations/ava_train_v2.1.csv" ] || [ ! -d "$DATA/labelframes" ] || [ ! -d "$DATA/videos/videos" ]; then
    command -v unzip >/dev/null || apt-get install -y -qq unzip
    ZIP="$WORK/cbvd-5cow-behavior-video-dataset.zip"
    unzip -tq "$ZIP" >/dev/null 2>&1 || curl -L --fail --retry 5 -C - -o "$ZIP" "$DATASET_URL" \
        || curl -L --fail --retry 5 -o "$ZIP" "$DATASET_URL"
    prefix="$(unzip -Z1 "$ZIP" | awk '!f && /(^|\/)annotations\/ava_train_v2\.1\.csv$/ {
        sub(/annotations\/ava_train_v2\.1\.csv$/, ""); print; f = 1 } END { exit !f }')"
    mkdir -p "$DATA/_x"
    unzip -q -o "$ZIP" "${prefix}annotations/*" "${prefix}labelframes/*" "${prefix}videos/*" -d "$DATA/_x"
    for d in annotations labelframes videos; do
        [ -d "$DATA/_x/${prefix}$d" ] && { rm -rf "$DATA/$d"; mv "$DATA/_x/${prefix}$d" "$DATA/$d"; }
    done
    rm -rf "$DATA/_x"
fi
echo "keyframes: $(find "$DATA/labelframes" -name '*.jpg' | wc -l), videos: $(find "$DATA/videos" -name '*.mp4' | wc -l)"

cd "$HERE"
step "3/6  Stage A: frame vectors (DINOv2, frozen) - train"
"$PY" herd.py extract --root "$DATA" --out "$FEAT" --split train --encoder "$ENCODER" --grid "$GRID"
step "4/6  Stage A: frame vectors - val"
"$PY" herd.py extract --root "$DATA" --out "$FEAT" --split val --encoder "$ENCODER" --grid "$GRID"

step "5/6  Training: temporal transformer + heads ($EPOCHS epochs)"
if [ -f "$OUT/model.pt" ] && [ -f "$OUT/eval_val.json" ]; then
    echo "trained already: $OUT"
else
    "$PY" herd.py train --features "$FEAT" --out "$OUT" --epochs "$EPOCHS"
fi
"$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val" score
"$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val" report

step "6/6  The NaN model (max error $MAX_ERROR among answers)"
"$PY" herd.py abstain --features "$FEAT" --run "$OUT" --max-error "$MAX_ERROR"

echo
echo "Done. In $OUT:"
echo "  model.pt            the temporal transformer + heads (and the frame heads)"
echo "  eval_val.json       val: re-ID top-1, posture / activity errors, rumination recall"
echo "  eval-val/report.md  the same keyframes as the LoRA runs, cowbench format"
echo "  abstain.json        when to answer NaN, and how often it does"
