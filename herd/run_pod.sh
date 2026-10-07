#!/usr/bin/env bash
# herd on a GPU pod, end to end on CBVD-5:
#
#   environment -> dataset (+ the 10 s videos the bursts are cut from)
#   -> Stage A features (DINOv2-S, train + val) -> train -> eval -> NaN model
#
#   bash herd/run_pod.sh          start in tmux session "herd"
#   bash herd/run_pod.sh log      follow the log
#   bash herd/run_pod.sh stop
#   bash herd/run_pod.sh stress   one GPU in real time: a burst alone, then CAMERAS (5) cameras at
#                                 1 fps + a 7 s burst a minute each (tmux "herd-stress", ~7 min);
#                                 RUN= the model (default: the newest run), DET= the detector's best/,
#                                 CAMERAS=5 DURATION=300 (s measured)
#   bash herd/run_pod.sh detector a new RT-DETRv2 for Full HD and far cows (tmux "herd-det", hours):
#                                 input DET_SIZE=1088 (a 1080x1080 tile at ~native size), zoom
#                                 crops in training, whole frame + tiles, epoch and threshold by F2
#                                 (a missed cow costs more than an extra box) -> WORK/herd/detector_<DET_TAG>;
#                                 then eval-det of the newest run with it. DET_EPOCHS=24 DET_BATCH=8
#                                 DET_MODEL=PekingU/rtdetr_v2_r101vd (default; ~76M parameters, the
#                                 deepest backbone) - PekingU/rtdetr_v2_r50vd is the one the LoRA runs and
#                                 the first herd runs used (~42M); DET_TAG=fhd_r101 (fhd_r50 for r50vd).
#                                 The first detector's own settings, only the backbone swapped:
#                                 DET_SIZE=960 DET_ZOOM=0 DET_TILES=0 DET_SELECT=f1 DET_TAG=r101_960
#
# Every step resumes: features are written clip by clip, so a rerun after a
# crash continues where it stopped. Knobs:
#   WORK=/workspace   RUN=run1   EPOCHS=40   ENCODER=facebook/dinov2-small
#   GRID=2            patch tokens pooled to GRID x GRID per frame (4: finer, e.g. the jaw;
#                     features 3.4x larger); another GRID extracts into features_g<GRID>
#   MAX_ERROR=0.01    the NaN cut-off: at most this share of wrong IDs among answers
#   MARGIN=0.1        context around each cow in the crops (0.5: twice the box, sees the feed
#                     barrier); another MARGIN extracts into its own features_m<MARGIN>
#   MOTION=1          the rhythm of each burst (chewing, motion.py) to the rumination and activity
#                     heads; computed once from the videos on the CPU (step 5), 0: without
#   QUALITY=1         frame quality taught directly: stretches of every burst spoilt
#                     (degrade.py: another cow in front, mud, blur, dark) and encoded (step 6,
#                     GPU), the quality head learns to weigh them down; 0: without
#   DET_KEYS=1        the frame heads also learn on the detector's boxes (+ jittered annotated
#                     boxes), so they answer as well on what the barn gives them (step 6)
#   POS=0             1: the heads also get where the cow is in the frame (fixed cameras: the
#                     feed barrier is a place in the picture); no new features needed

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
RUN_SET="${RUN:-}"
RUN="${RUN:-run1}"
EPOCHS="${EPOCHS:-40}"
ENCODER="${ENCODER:-facebook/dinov2-small}"
GRID="${GRID:-2}"
MAX_ERROR="${MAX_ERROR:-0.01}"
MARGIN="${MARGIN:-0.1}"
POS="${POS:-0}"
MOTION="${MOTION:-1}"
DET_KEYS="${DET_KEYS:-1}"
QUALITY="${QUALITY:-1}"
DATA="$WORK/cbvd5"
# Features depend on margin and grid: another value gets its own folder
# (a shared one would be skipped as "done" and silently reused).
FEAT="$WORK/herd/features"
[ "$MARGIN" = "0.1" ] || FEAT="${FEAT}_m$MARGIN"
[ "$GRID" = "2" ] || FEAT="${FEAT}_g$GRID"
OUT="$WORK/herd/$RUN"
VENV="$WORK/herd/.venv"
LOG="$WORK/herd/log_$RUN.txt"
SESSION=herd
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cbvd-5cow-behavior-video-dataset"
export HF_HOME="$WORK/hf-cache"

# The trained RT-DETRv2: DET if set; else the newest finished one under WORK
# (best/ next to train_done - a newer detector_fhd wins over the LoRA runs'
# one); else any best/; nothing printed and status 1 if none.
find_det() {
    local d="${DET:-}"
    if [ -z "$d" ]; then
        d="$(find "$WORK" -maxdepth 7 -name det_train_meta.json -path '*/best/*' -printf '%T@ %h\n' 2>/dev/null \
             | sort -rn | cut -d' ' -f2- | while read -r b; do [ -f "$(dirname "$b")/train_done" ] && { echo "$b"; break; }; done)"
    fi
    if [ -z "$d" ]; then
        d="$(find "$WORK" -maxdepth 7 -name det_train_meta.json -path '*/best/*' -printf '%T@ %h\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-)"
    fi
    [ -n "$d" ] && [ -f "$d/det_train_meta.json" ] && echo "$d"
}

case "${1:-}" in
    log)  # without RUN=: the newest run's log, not run1's
          [ -n "${RUN_SET:-}" ] || LOG="$(ls -t "$WORK"/herd/log_*.txt 2>/dev/null | head -1 || true)"
          [ -n "$LOG" ] || { echo "no log yet in $WORK/herd"; exit 1; }
          echo "following $LOG"
          exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "$SESSION" 2>/dev/null && echo stopped || echo "not running"; exit 0 ;;
    stress)
        [ -n "${RUN_SET:-}" ] || RUN="$(basename "$(dirname "$(ls -t "$WORK"/herd/*/model.pt 2>/dev/null | head -1)")")"
        MODEL="$WORK/herd/$RUN/model.pt"
        [ -f "$MODEL" ] || { echo "no model at $MODEL - train first, or RUN=<run>"; exit 1; }
        DET="$(find_det)" || { echo "no trained detector under $WORK (no best/det_train_meta.json)."
                               echo "copy one here or train it: bash cowbench/lora/run_lora.sh (detector steps),"
                               echo "then DET=<.../detector/best> bash $0 stress"; exit 1; }
        echo "detector: $DET"
        SOUT="$WORK/herd/stress_$RUN"
        mkdir -p "$SOUT"
        cmd="$(printf '%q ' "$VENV/bin/python" "$HERE/herd.py" stress --model "$MODEL" --detector "$DET" \
               --root "$DATA" --cameras "${CAMERAS:-5}" --duration "${DURATION:-300}" --out "$SOUT")"
        tmux has-session -t herd-stress 2>/dev/null && { echo "already running: tmux attach -t herd-stress"; exit 1; }
        env -u TMUX tmux new-session -d -s herd-stress -x 200 -y 50 \
            "export HF_HOME=$(printf '%q' "$HF_HOME"); $cmd 2>&1 | tee $(printf '%q' "$SOUT/log.txt"); echo '[stress finished]'; exec bash"
        echo "Started in tmux session 'herd-stress' (model $MODEL).  log: tail -F $SOUT/log.txt"
        echo "result: $SOUT/stress_herd_${CAMERAS:-5}cam.md"
        exit 0 ;;
    detector)
        DET_MODEL="${DET_MODEL:-PekingU/rtdetr_v2_r101vd}"
        DET_TAG="${DET_TAG:-fhd_$(echo "$DET_MODEL" | sed -n 's/.*_\(r[0-9]*\)vd$/\1/p')}"
        DOUT="$WORK/herd/detector_${DET_TAG}"
        mkdir -p "$DOUT"
        LAST="$(basename "$(dirname "$(ls -t "$WORK"/herd/*/model.pt 2>/dev/null | head -1)")" 2>/dev/null || true)"
        q() { printf '%q ' "$@"; }
        train="$(q "$VENV/bin/python" "$REPO/cowbench/detector.py" train --root "$DATA" --out "$DOUT" \
                   --model "$DET_MODEL" --size "${DET_SIZE:-1088}" --zoom "${DET_ZOOM:-0.5}" --tiles "${DET_TILES:-1}" --select "${DET_SELECT:-f2}" \
                   --epochs "${DET_EPOCHS:-24}" --batch "${DET_BATCH:-8}")"
        evald=""
        if [ -n "$LAST" ] && [ -f "$WORK/herd/$LAST/model.pt" ]; then
            evald="&& $(q "$VENV/bin/python" "$HERE/herd.py" eval-det --run "$WORK/herd/$LAST" --detector "$DOUT/best" \
                       --root "$DATA" --out "$WORK/herd/$LAST/eval-val-det-${DET_TAG}")"
        fi
        tmux has-session -t herd-det 2>/dev/null && { echo "already running: tmux attach -t herd-det"; exit 1; }
        env -u TMUX tmux new-session -d -s herd-det -x 200 -y 50 \
            "export HF_HOME=$(printf '%q' "$HF_HOME"); ( $train $evald ) 2>&1 | tee -a $(printf '%q' "$DOUT/log.txt"); echo '[detector finished]'; exec bash"
        echo "Started in tmux session 'herd-det' -> $DOUT/best.  log: tail -F $DOUT/log.txt"
        [ -n "$evald" ] && echo "then eval-det of $LAST with it -> $WORK/herd/$LAST/eval-val-det-${DET_TAG}/eval_det.json"
        exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop|stress|detector]"; exit 2 ;;
esac

if [ -z "${HERD_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    mkdir -p "$WORK/herd"
    knobs=""
    for v in WORK RUN EPOCHS ENCODER GRID MAX_ERROR MARGIN POS MOTION DET DET_KEYS QUALITY; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
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

step "1/10  Python environment ($VENV)"
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

step "2/10  CBVD-5 with its videos"
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
step "3/10  Stage A: frame vectors (DINOv2, frozen) - train"
"$PY" herd.py extract --root "$DATA" --out "$FEAT" --split train --encoder "$ENCODER" --grid "$GRID" --margin "$MARGIN"
step "4/10  Stage A: frame vectors - val"
"$PY" herd.py extract --root "$DATA" --out "$FEAT" --split val --encoder "$ENCODER" --grid "$GRID" --margin "$MARGIN"

step "5/10  The rhythm of every burst (chewing) - train + val, CPU"
if [ "$MOTION" = "1" ]; then
    WORKERS="$(( $(nproc) > 4 ? $(nproc) - 2 : 2 ))"; [ "$WORKERS" -gt 16 ] && WORKERS=16   # each holds one decoded clip in memory
    "$PY" herd.py motion --root "$DATA" --out "$FEAT" --split train --workers "$WORKERS"
    "$PY" herd.py motion --root "$DATA" --out "$FEAT" --split val --workers "$WORKERS"
else
    echo "MOTION=0: skipped"
fi

step "6/10  Spoilt burst frames (occlusion, mud, blur, dark) - train + val, GPU"
if [ "$QUALITY" = "1" ]; then
    "$PY" herd.py degrade --root "$DATA" --out "$FEAT" --split train
    "$PY" herd.py degrade --root "$DATA" --out "$FEAT" --split val
else
    echo "QUALITY=0: skipped"
fi

step "7/10  Keyframe crops from the detector's boxes (+ jittered) - train"
USE_KEYS=0
if [ "$DET_KEYS" = "1" ]; then
    if D6="$(find_det)"; then
        echo "detector: $D6"
        "$PY" herd.py keys --root "$DATA" --out "$FEAT" --split train --detector "$D6"
        USE_KEYS=1
    else
        echo "no trained detector under $WORK - the frame heads learn on annotated boxes only"
    fi
else
    echo "DET_KEYS=0: skipped"
fi

step "8/10  Training: temporal transformer + heads ($EPOCHS epochs)"
if [ -f "$OUT/model.pt" ] && [ -f "$OUT/eval_val.json" ]; then
    echo "trained already: $OUT"
else
    "$PY" herd.py train --features "$FEAT" --out "$OUT" --epochs "$EPOCHS" --pos "$POS" --motion "$MOTION" \
        --det-keys "$USE_KEYS" --quality "$QUALITY"
fi
"$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val" score
"$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val" report

step "9/10  The NaN model (max error $MAX_ERROR among answers)"
"$PY" herd.py abstain --features "$FEAT" --run "$OUT" --max-error "$MAX_ERROR"

step "10/10  On the detector's boxes: detector misses count as errors"
if DET="$(find_det)"; then
    echo "detector: $DET"
    "$PY" herd.py eval-det --run "$OUT" --detector "$DET" --root "$DATA"
    "$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val-det" score
    "$PY" "$REPO/cowbench/cowbench.py" --out "$OUT/eval-val-det" report
else
    echo "no trained detector under $WORK - skipped (DET=<.../detector/best> to point at one)"
fi

echo
echo "Done. In $OUT:"
echo "  model.pt            the temporal transformer + heads (and the frame heads)"
echo "  eval_val.json       also: frame_quality_auc, reid_spoilt_weighted vs _uniform"
echo "  eval_val.json       val: re-ID top-1, posture / activity errors, rumination recall"
echo "  eval-val/report.md  the same keyframes as the LoRA runs, cowbench format"
echo "  abstain.json        when to answer NaN, and how often it does"
echo "  eval_det.json, eval-val-det/report.md   the same on the detector's boxes: its misses count"
