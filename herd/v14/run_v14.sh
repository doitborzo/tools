#!/usr/bin/env bash
# Version 14 (A): cow identification only, trained from scratch on MultiCamCows2024 + MmCows.
#
#   bash herd/v14/run_v14.sh          start in tmux session "v14" (hours: downloads + training)
#   bash herd/v14/run_v14.sh log      follow the log
#   bash herd/v14/run_v14.sh stop
#
# Steps (each resumes):
#   1 Python environment                     5 ID transformer: train + test day (MultiCamCows)
#   2 version 10's detector + DINOv2          6 association transformer: train + test window (MmCows)
#   3 MultiCamCows2024 crops -> DINOv2        7 onboarding test on MmCows: association vs camera-alone
#   4 MmCows labels, frames, 1-s stream       8 RESULTS.md, RUN14A_RESULTS.tar, V14A_MODELS.tgz
#
# Knobs: WORK=/workspace  MODELS_URL (version 10's archive)  DET_NAME (its detector folder)
#        LIVE_STEP=2 (s between 1-second frames used)  TRAIN_EVERY=60  WINDOW_EVERY=15
#        SETUP_MIN=60  START=12  END=15  ID_EPOCHS=60  ASSOC_EPOCHS=60

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERD="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
V="$WORK/v14"
VENV="$WORK/herd/.venv"
PY="$VENV/bin/python"
LOG="$V/log.txt"
SESSION=v14
MODELS_URL="${MODELS_URL:-https://drive.usercontent.google.com/download?id=1urVuSsmAapKD8O1SZ-b9CR5ycOZePRe3&export=download&confirm=t}"
DET_NAME="${DET_NAME:-detector_rfdetr_large_zenodo_mmcows_opencows_ft}"
LIVE_STEP="${LIVE_STEP:-2}"
TRAIN_EVERY="${TRAIN_EVERY:-60}"
WINDOW_EVERY="${WINDOW_EVERY:-15}"
SETUP_MIN="${SETUP_MIN:-60}"
START="${START:-12}"
END="${END:-15}"
ID_EPOCHS="${ID_EPOCHS:-60}"
ASSOC_EPOCHS="${ASSOC_EPOCHS:-60}"
export HF_HOME="$WORK/hf-cache"

case "${1:-}" in
    log)  [ -f "$LOG" ] || { echo "no log yet: $LOG"; exit 1; }; exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${V14_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    mkdir -p "$V"
    knobs=""
    for v in WORK MODELS_URL DET_NAME LIVE_STEP TRAIN_EVERY WINDOW_EVERY SETUP_MIN START END ID_EPOCHS ASSOC_EPOCHS; do
        knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env V14_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/run_v14.sh"); echo; echo '[run_v14.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log"
    exit 0
fi

mkdir -p "$V"
trap '' HUP
exec > >(tee -a "$LOG") 2>&1
set -E
trap 'rc=$?; echo "!! line $LINENO failed (exit $rc): $BASH_COMMAND"' ERR
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }

step "1/8  Python environment ($VENV)"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
ready() { [ -x "$PY" ] && "$PY" -c "import torch, torchvision, transformers, cv2, numpy, scipy, PIL; assert torch.cuda.is_available()" 2>/dev/null; }
if ! ready; then
    [ -x "$PY" ] || rm -rf "$VENV"
    [ -d "$VENV" ] || uv venv --python 3.12 --seed --managed-python "$VENV"
    uv pip install --python "$PY" torch torchvision --torch-backend=auto
    uv pip install --python "$PY" "transformers>=5.15" opencv-python-headless numpy scipy pillow requests safetensors
    ready || { echo "!! the venv cannot import torch with CUDA / transformers / cv2"; exit 1; }
fi
"$PY" -c "import rfdetr, pytorch_lightning" 2>/dev/null || uv pip install --python "$PY" "rfdetr[train]"
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
cd "$HERE"

step "2/8  Version 10's detector and DINOv2"
DET="$WORK/herd/$DET_NAME/best"
if [ ! -f "$DET/checkpoint.pth" ]; then
    mkdir -p "$WORK/v10"
    A="$WORK/v10/models.tgz"
    [ -s "$A" ] || curl -L --fail --retry 5 -C - -o "$A" "$MODELS_URL"
    head -c 512 "$A" | grep -qi '<html' && { rm -f "$A"; echo "!! the models link gives a web page: share it as 'Anyone with the link'"; exit 1; }
    tar -xzf "$A" -C "$WORK" "herd/$DET_NAME" "hf-cache" 2>/dev/null || tar -xzf "$A" -C "$WORK"
fi
[ -f "$DET/det_train_meta.json" ] || { echo "!! no $DET - set DET_NAME to a detector folder in the archive"; ls "$WORK/herd"; exit 1; }
echo "detector: $DET"

step "3/8  MultiCamCows2024: crops -> DINOv2 (the videos are not downloaded)"
"$PY" data_mcc.py inspect | tee "$V/mcc_inspect.txt"
"$PY" data_mcc.py fetch --out "$V/mcc"

step "4/8  MmCows: labels, labelled frames (training side + test window), the 1-second stream"
"$PY" data_mmcows.py labels --out "$V/mmcows"
"$PY" data_mmcows.py frames --out "$V/mmcows" --train-every "$TRAIN_EVERY" --window-every "$WINDOW_EVERY" --start "$START" --end "$END"
"$PY" data_mmcows.py inspect --out "$V/mmcows" | tee "$V/mmcows_live_inspect.txt"
"$PY" data_mmcows.py live --out "$V/mmcows" --live-step "$LIVE_STEP" --start "$START" --end "$END"
"$PY" data_mmcows.py feats --out "$V/mmcows"

step "5/8  ID transformer (from scratch): MultiCamCows days 1-5 -> day 6 picks -> day 7 tests"
if [ -f "$V/idnet/idnet.pt" ] && [ -f "$V/idnet/eval_mcc.json" ]; then echo "trained already"; else
    "$PY" idnet.py train --feats "$V/mcc" --out "$V/idnet" --epochs "$ID_EPOCHS"; fi

step "6/8  Association transformer (from scratch): MmCows outside the test window -> the window"
if [ -f "$V/assoc/assoc.pt" ] && [ -f "$V/assoc/eval_assoc.json" ]; then echo "trained already"; else
    "$PY" assoc.py train --data "$V/mmcows" --out "$V/assoc" --epochs "$ASSOC_EPOCHS"; fi

step "7/8  Onboarding test on MmCows (inference only): setup $SETUP_MIN min, then identification"
"$PY" system.py --data "$V/mmcows" --detector "$DET" --idnet "$V/idnet/idnet.pt" --assoc "$V/assoc/assoc.pt" \
    --out "$V/system" --setup-min "$SETUP_MIN" --start "$START" --end "$END" --step "$LIVE_STEP"
"$PY" system.py --data "$V/mmcows" --detector "$DET" --idnet "$V/idnet/idnet.pt" --assoc none \
    --out "$V/system" --setup-min "$SETUP_MIN" --start "$START" --end "$END" --step "$LIVE_STEP"

step "8/8  Results and models"
R="$WORK/RUN14A_RESULTS"
rm -rf "$R"; mkdir -p "$R"
"$PY" summarize.py --work "$V" --out "$R"
for f in idnet/eval_mcc.json idnet/history.json assoc/eval_assoc.json assoc/history.json \
         system/system_assoc.json system/system_iou_per_camera.json mcc_inspect.txt mmcows_live_inspect.txt \
         mcc/meta.json mcc/samples.png mmcows/assoc_meta.json log.txt; do
    [ -f "$V/$f" ] && { mkdir -p "$R/$(dirname "$f")"; cp "$V/$f" "$R/$f"; }
done
cp "$DET/det_train_meta.json" "$R/detector_det_train_meta.json"
tar -cf "$WORK/RUN14A_RESULTS.tar" -C "$WORK" RUN14A_RESULTS
M="$WORK/V14A_MODELS"
rm -rf "$M"; mkdir -p "$M"
cp "$V/idnet/idnet.pt" "$V/assoc/assoc.pt" "$M/"
cat > "$M/HOW_TO_USE.txt" <<EOF
v14 (A) models, $(date '+%F %T')
idnet.pt   ID transformer: crop sequence -> fingerprint (+ 'not sure' model, enrolment threshold)
assoc.pt   association transformer: detections of all cameras -> one cow
detector   version 10's: herd/$DET_NAME/best (not retrained; in the version 10 archive)
DINOv2-S   facebook/dinov2-small, frozen (hf-cache in the version 10 archive)
Run:       python herd/v14/system.py --data <mmcows> --detector <best/> --idnet idnet.pt --assoc assoc.pt --out <dir>
EOF
tar -czf "$WORK/V14A_MODELS.tgz" -C "$WORK" V14A_MODELS
echo
echo "Done:"
echo "  $WORK/RUN14A_RESULTS.tar   ($(du -h "$WORK/RUN14A_RESULTS.tar" | cut -f1))  RESULTS.md + every score"
echo "  $WORK/V14A_MODELS.tgz      ($(du -h "$WORK/V14A_MODELS.tgz" | cut -f1))  idnet.pt, assoc.pt"
