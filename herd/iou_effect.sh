#!/usr/bin/env bash
# The overlap rule's effect on the frame heads, no training: run9's detector and
# model scored on CBVD-5 val twice - detector boxes matched to the annotated cows at
# IoU 0.5 and at IoU 0.3 - then compared (compare_iou.py) and packed into one tar.
#
#   bash herd/iou_effect.sh          start in tmux session "herd-iou" (~30-45 min on a new pod)
#   bash herd/iou_effect.sh log      follow the log
#   bash herd/iou_effect.sh stop
#
# Knobs:
#   WORK=/workspace
#   WEIGHTS_URL   Google Drive link (or any URL) to an archive holding run9's model folder
#                 (model.pt) and its RF-DETR detector folder (best/ with det_train_meta.json)
#   RUN_DIR=...   run9's folder, when already on the pod (skips the download)
#   DET=...       the detector's best/ folder, when already on the pod
#   NAME=RUN9_RESULTS_30_OVERLAP_EFFECT   the tar: WORK/NAME.tar

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
WEIGHTS_URL="${WEIGHTS_URL:-https://drive.google.com/file/d/1UUcoCn-vnr1MjEevl_0qfFMDQ7KBEEHK/view?usp=sharing}"
NAME="${NAME:-RUN9_RESULTS_30_OVERLAP_EFFECT}"
DATA="$WORK/cbvd5"
VENV="$WORK/herd/.venv"
PY="$VENV/bin/python"
WDIR="$WORK/run9_weights"
OUT="$WORK/herd/iou_effect"
LOG="$OUT/log.txt"
SESSION=herd-iou
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cbvd-5cow-behavior-video-dataset"
export HF_HOME="$WORK/hf-cache"

case "${1:-}" in
    log)  [ -f "$LOG" ] || { echo "no log yet: $LOG"; exit 1; }; exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    mkdir -p "$OUT"
    knobs=""
    for v in WORK WEIGHTS_URL RUN_DIR DET NAME; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/iou_effect.sh"); echo; echo '[iou_effect.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log"
    exit 0
fi

mkdir -p "$OUT"
trap '' HUP
exec > >(tee -a "$LOG") 2>&1
set -E
trap 'rc=$?; echo "!! line $LINENO failed (exit $rc): $BASH_COMMAND"' ERR
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }

step "1/6  Python environment ($VENV)"
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
"$PY" -c "import gdown" 2>/dev/null || uv pip install --python "$PY" gdown
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

step "2/6  CBVD-5 (val keyframes and annotations)"
if [ ! -f "$DATA/annotations/ava_val_v2.1.csv" ] || [ ! -d "$DATA/labelframes" ]; then
    command -v unzip >/dev/null || { apt-get update -qq; apt-get install -y -qq unzip; }
    ZIP="$WORK/cbvd-5cow-behavior-video-dataset.zip"
    unzip -tq "$ZIP" >/dev/null 2>&1 || curl -L --fail --retry 5 -C - -o "$ZIP" "$DATASET_URL" \
        || curl -L --fail --retry 5 -o "$ZIP" "$DATASET_URL"
    prefix="$(unzip -Z1 "$ZIP" | awk '!f && /(^|\/)annotations\/ava_val_v2\.1\.csv$/ {
        sub(/annotations\/ava_val_v2\.1\.csv$/, ""); print; f = 1 } END { exit !f }')"
    mkdir -p "$DATA/_x"
    unzip -q -o "$ZIP" "${prefix}annotations/*" "${prefix}labelframes/*" -d "$DATA/_x"
    for d in annotations labelframes; do
        [ -d "$DATA/_x/${prefix}$d" ] && { rm -rf "$DATA/$d"; mv "$DATA/_x/${prefix}$d" "$DATA/$d"; }
    done
    rm -rf "$DATA/_x"
fi
echo "keyframes: $(find "$DATA/labelframes" -name '*.jpg' | wc -l)"

step "3/6  run9's model and detector"
if [ -z "${RUN_DIR:-}" ] || [ -z "${DET:-}" ]; then
    if [ ! -f "$WDIR/.unpacked" ]; then
        mkdir -p "$WDIR"
        A="$WDIR/download"
        # a web page instead of the archive (Drive's sign-in or virus-scan page): not kept
        is_page() { head -c 512 "$1" 2>/dev/null | grep -qi '<!doctype html\|<html'; }
        [ -s "$A" ] && is_page "$A" && rm -f "$A"
        if [ ! -s "$A" ]; then
            ID="$(printf '%s' "$WEIGHTS_URL" | sed -n 's#.*/d/\([A-Za-z0-9_-]*\).*#\1#p; s#.*[?&]id=\([A-Za-z0-9_-]*\).*#\1#p' | head -1)"
            if [ -n "$ID" ]; then      # Google Drive: gdown by id (any version), then Drive's direct link
                "$VENV/bin/gdown" "$ID" -O "$A" || { rm -f "$A"; curl -L --fail --retry 5 -o "$A" \
                    "https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t"; }
            else
                curl -L --fail --retry 5 -o "$A" "$WEIGHTS_URL"
            fi
        fi
        if [ ! -s "$A" ] || is_page "$A"; then
            rm -f "$A"
            echo "!! the download is a web page, not the archive: share the Drive file as 'Anyone with the link',"
            echo "   or copy the archive to the pod yourself and run: RUN_DIR=<run9 folder> DET=<detector best/> bash $0"
            exit 1
        fi
        echo "downloaded $(du -h "$A" | cut -f1)"
        case "$(file -b "$A" 2>/dev/null || echo unknown)" in
            Zip*)          command -v unzip >/dev/null || apt-get install -y -qq unzip; unzip -q -o "$A" -d "$WDIR" ;;
            gzip*|*tar*|POSIX*) tar -xf "$A" -C "$WDIR" ;;
            *)             tar -xf "$A" -C "$WDIR" 2>/dev/null || unzip -q -o "$A" -d "$WDIR" \
                               || { echo "!! $A is neither a tar nor a zip"; exit 1; } ;;
        esac
        touch "$WDIR/.unpacked"
    fi
    RUN_DIR="${RUN_DIR:-$(dirname "$(find "$WDIR" -name model.pt | head -1)")}"
    DET="${DET:-$(dirname "$(find "$WDIR" -name det_train_meta.json -exec grep -l '"rfdetr"' {} + 2>/dev/null | head -1)")}"
fi
[ -f "$RUN_DIR/model.pt" ] || { echo "!! no model.pt found (RUN_DIR=$RUN_DIR); set RUN_DIR=<run9 folder>"; exit 1; }
[ -f "$DET/det_train_meta.json" ] || { echo "!! no RF-DETR best/ found (DET=$DET); set DET=<detector best/>"; exit 1; }
echo "model:    $RUN_DIR"
echo "detector: $DET  (box calibration: $([ -f "$DET/box_calib.json" ] && echo yes || echo 'NO - boxes uncalibrated'))"

cd "$HERE"
for IOU in 0.5 0.3; do
    TAG="iou0${IOU#0.}"
    step "$([ "$IOU" = 0.5 ] && echo 4 || echo 5)/6  eval-det at IoU $IOU (no training)"
    "$PY" herd.py eval-det --run "$RUN_DIR" --detector "$DET" --root "$DATA" --iou "$IOU" --out "$OUT/eval-val-det-$TAG"
    "$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val-det-$TAG" score || echo "(cowbench score skipped)"
    "$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val-det-$TAG" report || echo "(cowbench report skipped)"
done

step "6/6  Compare and pack -> $WORK/$NAME.tar"
"$PY" "$HERE/compare_iou.py" --strict "$OUT/eval-val-det-iou05" --loose "$OUT/eval-val-det-iou03" --out "$OUT"
mkdir -p "$OUT/detector"
cp "$DET/det_train_meta.json" "$OUT/detector/"
[ -f "$DET/box_calib.json" ] && cp "$DET/box_calib.json" "$OUT/detector/"
[ -f "$RUN_DIR/heads.json" ] && cp "$RUN_DIR/heads.json" "$OUT/"
[ -f "$RUN_DIR/eval_det.json" ] && cp "$RUN_DIR/eval_det.json" "$OUT/run9_original_eval_det.json"
rm -rf "$WORK/$NAME"; cp -r "$OUT" "$WORK/$NAME"
tar -cf "$WORK/$NAME.tar" -C "$WORK" "$NAME"
echo
echo "Done: $WORK/$NAME.tar ($(du -h "$WORK/$NAME.tar" | cut -f1))"
echo "  iou_effect.md        the comparison (also printed above)"
echo "  iou_effect.json      the same, all numbers"
echo "  eval-val-det-iou05/  eval-det at IoU 0.5    eval-val-det-iou03/  at IoU 0.3"
