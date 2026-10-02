#!/usr/bin/env bash
# Everything after training, on a fresh pod, in one go:
#
#   dataset -> adapter -> RT-DETRv2 on every val keyframe -> vLLM with the adapter
#   -> bench with the adapter -> 12 cameras at once, detector and model together
#   -> one tar.gz
#
# The cows are those RT-DETRv2 finds, never the annotation's boxes: the model
# answers about what the detector found, the annotation only scores, and a cow
# the detector missed is an error. The detector is run here, from the weights
# run_lora.sh trained (lora-runs/detector/best), in its own venv. vLLM is
# started with DET_RESERVE_MIB of the card left free, so in the camera test the
# detector runs live on every frame on the same GPU, next to the model.
#
# Usage (the repo checked out; from the training pod, the adapter folder and
# the detector's best/ folder - see ADAPTER and DET_DIR below):
#   bash cowbench/run_tests.sh           start in a background tmux session "cowtests"
#   bash cowbench/run_tests.sh log       follow the log (Ctrl-c stops watching only)
#   bash cowbench/run_tests.sh attach    watch it live (detach: Ctrl-b, then d)
#   bash cowbench/run_tests.sh stop      stop the tests (vLLM keeps running in tmux "vllm")
#
# Every step skips what is already done, so after a failure run it again.
# vLLM is started with ../lora_muse.sh in its own tmux session "vllm" and is
# left running afterwards; an already running server with the adapter is reused.
#
# Knobs, all optional:
#   WORK=/workspace                 where the dataset, adapter and results live
#   QUANT=fp8                       the checkpoint vLLM serves (../lora_muse.sh): fp8,
#                                   nvfp4 (RedHatAI W4A4) or nvfp4-nvidia (mixed W4A16/FP8);
#                                   MODEL=<repo> for any other. A running vLLM serving
#                                   another checkpoint is restarted
#   ADAPTER=$WORK/lora-runs/lora4frame_w1920/adapter
#   LORA_NAME=lora4frame            the adapter's model name on the server
#   DET_DIR=$WORK/lora-runs/detector   the detector: best/ holds its weights; the boxes
#                                   it finds go to val_detections.jsonl beside it
#                                   (with det_meta.json: the threshold; det_report.md)
#   BOXES=$DET_DIR/val_detections.jsonl   boxes made elsewhere: used as they are when
#                                   there are no weights to make them from
#   DET_THRESHOLD                   override the detector's score threshold
#   DET_RESERVE_MIB=4096            GPU memory kept free of vLLM for the detector
#   GPU_MEM_UTIL                    vLLM's share of the card; by default 1 minus that
#                                   reserve and 512 MiB of margin (0.94 on 80 GB)
#   LIVE=1                          camera test with the detector live on every frame
#                                   (when its weights are here); 0: its boxes from file
#   MUSE_DETECT=0                   1: also let the base model find the cows by itself
#                                   (the earlier test: 53% found) and stress that
#   WIDTH, UNIT                     how the adapter is asked: by default as it was trained,
#                                   read from the train_meta.json next to the adapter
#                                   (else 896, cow). UNIT=frame: one question per keyframe
#                                   about every cow on it (frame.py), stress task "frame"
#   STREAMS=12                      cameras in the stress tests
#   INTERVAL=10                     seconds between frames per camera, paced test
#   DURATION=300                    seconds measured per stress test
#   OUT=<repo>/cowbench/runs/tests_<LORA_NAME>_w<WIDTH>_det_<QUANT>_<gpu>   results
#   PORT=8000

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
ROOT="$WORK/cbvd5"
QUANT="${QUANT:-fp8}"
case "$QUANT" in   # as in ../lora_muse.sh
    fp8)          MODEL="${MODEL:-RedHatAI/Muse-Glimmer-30B-FP8-block}" ;;
    nvfp4)        MODEL="${MODEL:-RedHatAI/Muse-Glimmer-30B-NVFP4}" ;;
    nvfp4-nvidia) MODEL="${MODEL:-nvidia/Muse-Glimmer-30B-NVFP4}" ;;
    *) echo "QUANT must be fp8, nvfp4 or nvfp4-nvidia"; exit 2 ;;
esac
# A short name of this machine's GPU for the results folder.
gpu_tag() {
    local name
    name="$( { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true; } | head -1)"
    case "$name" in
        *"RTX PRO 6000"*) echo rtxpro6000 ;;
        *A100*) echo a100 ;; *H100*) echo h100 ;; *H200*) echo h200 ;;
        *B200*) echo b200 ;; *B300*) echo b300 ;; *L40S*) echo l40s ;;
        "") echo nogpu ;;
        *) echo "$name" | tr 'A-Z' 'a-z' | sed 's/nvidia//; s/[^a-z0-9]//g' | cut -c1-16 ;;
    esac
}
ADAPTER="${ADAPTER:-$WORK/lora-runs/lora4frame_w1920/adapter}"
LORA_NAME="${LORA_NAME:-lora4frame}"
DET_DIR="${DET_DIR:-$WORK/lora-runs/detector}"
BOXES="${BOXES:-$DET_DIR/val_detections.jsonl}"
DET_THRESHOLD="${DET_THRESHOLD:-}"
MUSE_DETECT="${MUSE_DETECT:-0}"
DET_RESERVE_MIB="${DET_RESERVE_MIB:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
LIVE="${LIVE:-1}"
# The width and unit the adapter was trained at, from its run's train_meta.json
# (adapter/ or adapters/step-N sit one or two levels below it).
trained() {
    local meta
    for meta in "$(dirname "$ADAPTER")/train_meta.json" "$(dirname "$(dirname "$ADAPTER")")/train_meta.json"; do
        if [ -f "$meta" ]; then
            grep -o "\"$1\": *\"\{0,1\}[a-z0-9]*" "$meta" | head -1 | sed 's/.*[" ]//' || true
            return
        fi
    done
}
WIDTH="${WIDTH:-$(trained width)}"; WIDTH="${WIDTH:-896}"
UNIT="${UNIT:-$(trained unit)}"; UNIT="${UNIT:-cow}"
STREAMS="${STREAMS:-12}"
INTERVAL="${INTERVAL:-10}"
DURATION="${DURATION:-300}"
PORT="${PORT:-8000}"
OUT="${OUT:-$HERE/runs/tests_${LORA_NAME}_w${WIDTH}_det_${QUANT}_$(gpu_tag)}"
BASE_URL="http://127.0.0.1:$PORT"
PLAN="$HERE/runs/2026-09-25_val-full_w1920_f1"   # the full-val sample every run used
SESSION=cowtests
LOG="$OUT/log.txt"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/fandaoerji/cbvd-5cow-behavior-video-dataset"

case "${1:-}" in
    log)    exec tail -n 100 -F "$LOG" ;;
    attach) exec tmux attach -t "$SESSION" ;;
    stop)   tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "not running"
            exit 0 ;;
    "") ;;
    *)      echo "usage: $0 [log|attach|stop]"; exit 2 ;;
esac

# Into tmux, so a closed terminal does not end a run that takes an hour.
if [ -z "${COWTESTS_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "already running in tmux session '$SESSION':  bash $0 log"; exit 1
    fi
    knobs=""
    for v in WORK QUANT MODEL ADAPTER LORA_NAME DET_DIR BOXES DET_THRESHOLD MUSE_DETECT DET_RESERVE_MIB GPU_MEM_UTIL LIVE WIDTH UNIT STREAMS INTERVAL DURATION PORT OUT; do
        knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env COWTESTS_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/run_tests.sh"); echo; echo '[run_tests.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION'."
    echo "  log    :  bash $0 log      ($LOG)"
    echo "  watch  :  bash $0 attach"
    exit 0
fi

mkdir -p "$OUT"
trap '' HUP
exec > >(tee -a "$LOG") 2>&1
set -E
trap 'rc=$?; echo "!! line $LINENO failed (exit $rc): $BASH_COMMAND"' ERR
trap 'rc=$?; echo "#### $(date +%H:%M:%S)  run_tests.sh exited with code $rc"' EXIT
step() { echo; echo "=== $*   [$(date '+%Y-%m-%d %H:%M:%S')]"; }

step "1/7  Dataset"
if [ -f "$ROOT/annotations/ava_val_v2.1.csv" ] && [ -d "$ROOT/labelframes/labelframes" ]; then
    echo "found: $ROOT"
else
    command -v unzip >/dev/null || apt-get install -y -qq unzip
    zip="$WORK/cbvd.zip"
    unzip -tq "$zip" >/dev/null 2>&1 || curl -L --fail --retry 5 -o "$zip" "$DATASET_URL"
    # The archive root holds annotations/ and labelframes/; it also holds
    # miniannotations/ and minilabelframes/ (256 px), which a loose pattern takes.
    rm -rf "$ROOT"
    unzip -q "$zip" "annotations/*" "labelframes/*" -d "$ROOT"
    rm -f "$zip"
fi
echo "keyframes: $(find "$ROOT/labelframes" -name '*.jpg' | wc -l)"

step "2/7  Adapter"
if [ ! -f "$ADAPTER/adapter_config.json" ] || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "!! no adapter in $ADAPTER (adapter_config.json + adapter_model.safetensors)."
    echo "!! From a PC:  scp -P <port> -r <adapter folder> root@<ip>:$(dirname "$ADAPTER")/"
    exit 1
fi
ls -la "$ADAPTER"

step "3/7  Cow detector (RT-DETRv2) on every val keyframe"
# Its own venv: torch + transformers + scipy, ~4 GB, kept on $WORK. Not the
# vLLM venv - vLLM pins its own transformers - and not yet there on a fresh pod.
DET_VENV="$WORK/det/.venv"
DET_PY="$DET_VENV/bin/python"
det_env_ok() {
    [ -x "$DET_PY" ] && "$DET_PY" -c "import torch, torchvision, scipy, PIL, requests
from transformers import RTDetrV2ForObjectDetection" 2>/dev/null
}
make_det_env() {
    det_env_ok && return
    echo "setting up the detector's Python in $DET_VENV (first time: a few minutes)"
    if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
    export PATH="$HOME/.local/bin:$PATH"
    uv self update >/dev/null 2>&1 || true
    # A venv whose Python went with a pod restart (it lives under ~/.local)
    # cannot be repaired in place; it is small enough to rebuild.
    [ -x "$DET_PY" ] || rm -rf "$DET_VENV"
    [ -d "$DET_VENV" ] || uv venv --python 3.12 --seed --managed-python "$DET_VENV"
    # torchvision: transformers' fast RT-DETR image processor needs it.
    uv pip install --python "$DET_PY" torch torchvision --torch-backend=auto
    uv pip install --python "$DET_PY" "transformers>=5.15" scipy pillow requests safetensors
    det_env_ok || { echo "!! the detector's venv does not import torch / transformers RT-DETRv2"; exit 1; }
}
gpu_mib() {   # memory.free or memory.total of GPU 0, MiB; empty without a GPU
    { nvidia-smi --query-gpu="memory.$1" --format=csv,noheader,nounits 2>/dev/null || true; } | head -1 | tr -dc '0-9'
}
if [ -f "$DET_DIR/best/det_train_meta.json" ]; then
    make_det_env
    # A server started without the reserve (by hand, or an older run) may
    # leave no room: then the CPU (a few minutes for 292 keyframes; the
    # detector's ms a frame in det_report.md is then a CPU number).
    det_gpu=()
    free_mib="$(gpu_mib free)"
    if [ "${free_mib:-0}" -lt "$DET_RESERVE_MIB" ]; then
        echo "GPU busy (${free_mib:-?} MiB free) - running the detector on the CPU"
        det_gpu=(env CUDA_VISIBLE_DEVICES=)
    fi
    "${det_gpu[@]}" "$DET_PY" "$HERE/detector.py" detect --root "$ROOT" --out "$DET_DIR"
    "${det_gpu[@]}" "$DET_PY" "$HERE/detector.py" score --out "$DET_DIR"
    BOXES="$DET_DIR/val_detections.jsonl"
elif [ -s "$BOXES" ] && [ -f "$(dirname "$BOXES")/det_meta.json" ]; then
    echo "no detector weights in $DET_DIR/best - using the boxes in $BOXES as they are,"
    echo "and in the camera test too (no live detector without its weights)"
    LIVE=0
else
    echo "!! no detector in $DET_DIR/best (and no ready boxes in $BOXES)."
    echo "!! run_lora.sh trains it (step 9) into lora-runs/detector/best on the training pod."
    echo "!! From a PC:  scp -P <port> -r <detector/best folder> root@<ip>:$DET_DIR/"
    exit 1
fi
mkdir -p "$OUT"
BOX_DIR="$(dirname "$BOXES")"
cp "$BOX_DIR/det_meta.json" "$OUT/"
if [ -f "$BOX_DIR/det_report.md" ]; then cp "$BOX_DIR/det_report.md" "$OUT/"; cat "$BOX_DIR/det_report.md"; fi
BOX_ARGS=(--boxes "$BOXES")
[ -n "$DET_THRESHOLD" ] && BOX_ARGS+=(--det-threshold "$DET_THRESHOLD")

step "4/7  vLLM with the adapter, ${DET_RESERVE_MIB} MiB left for the detector"
gpu_total="$(gpu_mib total)"
if [ -z "$GPU_MEM_UTIL" ]; then
    GPU_MEM_UTIL=0.95
    [ -n "$gpu_total" ] && GPU_MEM_UTIL="$(awk -v t="$gpu_total" -v r="$DET_RESERVE_MIB" \
        'BEGIN { u = int(100 * (t - r - 1536) / t) / 100; if (u > 0.95) u = 0.95; printf "%.2f", u }')"
fi
echo "vLLM gets ${GPU_MEM_UTIL} of ${gpu_total:-?} MiB"
server_has_adapter() { curl -sf "$BASE_URL/v1/models" 2>/dev/null | grep -q "\"$LORA_NAME\""; }
# The adapter alone is not enough: the same adapter on another checkpoint is
# another test. vLLM reports the checkpoint as the base model's "root".
serves_model() { curl -sf "$BASE_URL/v1/models" 2>/dev/null | grep -qE "\"root\": *\"$MODEL\""; }
room_for_detector() {   # true too without nvidia-smi: nothing to measure there
    local free; free="$(gpu_mib free)"
    [ "$LIVE" != 1 ] || [ -z "$free" ] || [ "$free" -ge "$DET_RESERVE_MIB" ]
}
start_vllm() {
    rm -f "$WORK/vllm.exit"
    # lora_muse.sh may ask "Continue anyway?" about open files; that concerns
    # data-parallel, which one GPU does not use, so the answer is yes.
    tmux new-session -d -s vllm -x 200 -y 50 \
        "yes y | env LORA_MODULES=$(printf '%q' "$LORA_NAME=$ADAPTER") GPU_MEM_UTIL=$GPU_MEM_UTIL QUANT=$QUANT MODEL=$(printf '%q' "$MODEL") bash $(printf '%q' "$REPO/lora_muse.sh") 2>&1 | tee -a $WORK/vllm.log; echo \$? > $WORK/vllm.exit; exec bash"
    echo "starting vLLM in tmux session 'vllm' (log: $WORK/vllm.log)."
    echo "First start installs vLLM and downloads $MODEL (20-35 GB): up to an hour."
    t0=$SECONDS
    until server_has_adapter && serves_model; do
        if [ -f "$WORK/vllm.exit" ]; then
            echo; echo "!! vLLM stopped. Last lines of $WORK/vllm.log:"; tail -30 "$WORK/vllm.log"; exit 1
        fi
        if [ $((SECONDS - t0)) -gt 5400 ]; then
            echo; echo "!! vLLM not up after 90 min - see: tmux attach -t vllm"; exit 1
        fi
        printf '\r  waiting %4d s   %s' $((SECONDS - t0)) "$(tail -c 300 "$WORK/vllm.log" 2>/dev/null | tr '\r\n' '  ' | tail -c 90)"
        sleep 20
    done
    echo
}
if server_has_adapter && serves_model && room_for_detector; then
    echo "already serving $LORA_NAME on $MODEL at $BASE_URL, $(gpu_mib free) MiB free for the detector"
elif server_has_adapter && ! serves_model && ! tmux has-session -t vllm 2>/dev/null; then
    echo "!! $BASE_URL serves $LORA_NAME on another checkpoint than $MODEL, from a vLLM this"
    echo "!! script did not start. Stop it and rerun - this script then starts its own."
    exit 1
elif server_has_adapter && serves_model && ! tmux has-session -t vllm 2>/dev/null; then
    echo "!! $LORA_NAME is served by a vLLM this script did not start, and only $(gpu_mib free) MiB"
    echo "!! are free: the live detector may not fit. Restart that server with"
    echo "!! GPU_MEM_UTIL=$GPU_MEM_UTIL, or stop it and rerun - this script then starts its own."
else
    if tmux has-session -t vllm 2>/dev/null; then
        if server_has_adapter && ! serves_model; then
            echo "the tmux session 'vllm' serves another checkpoint, not $MODEL - restarting it"
        elif server_has_adapter; then
            echo "the tmux session 'vllm' leaves only $(gpu_mib free) MiB for the detector - restarting it"
        else
            echo "a tmux session 'vllm' exists but does not serve $LORA_NAME - restarting it"
        fi
        tmux kill-session -t vllm
        sleep 10   # the GPU memory is released when the process is gone
    fi
    start_vllm
fi
# vLLM can take more than its share (CUDA context, graphs): on an RTX PRO 6000
# at 0.95 the detector then found no room at all. Measure, and give vLLM less.
for retry in 1 2; do
    if room_for_detector || ! tmux has-session -t vllm 2>/dev/null; then break; fi
    free="$(gpu_mib free)"
    GPU_MEM_UTIL="$(awk -v u="$GPU_MEM_UTIL" -v t="$gpu_total" -v f="$free" -v r="$DET_RESERVE_MIB" \
        'BEGIN { d = (r - f + 1024) / t; d = int(d * 100 + 0.999) / 100; printf "%.2f", u - d }')"
    echo "only $free MiB free next to vLLM, $DET_RESERVE_MIB needed - restarting it at $GPU_MEM_UTIL"
    tmux kill-session -t vllm
    sleep 10
    start_vllm
done
echo "GPU memory free next to vLLM: $(gpu_mib free) MiB (vLLM at $GPU_MEM_UTIL)"
curl -s "$BASE_URL/v1/models" | grep -o '"id": *"[^"]*"' | sed 's/"id": */  model: /'

step "5/7  Python for the bench"
PY=""
for cand in "$WORK/serving/.venv/bin/python" python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import requests, PIL" 2>/dev/null; then
        PY="$cand"; break
    fi
done
if [ -z "$PY" ]; then
    python3 -m pip install -q requests pillow || python3 -m pip install -q --break-system-packages requests pillow
    PY=python3
fi
echo "using $PY"
cd "$HERE"
bench() { "$PY" cowbench.py --out "$OUT" "$@"; }
[ -f "$OUT/manifest.jsonl" ] || cp "$PLAN/manifest.jsonl" "$PLAN/plan_meta.json" "$OUT/"

step "6/7  Bench with the adapter on the detector's boxes"
# The same question run_lora.sh asked in eval-lora-det/ (transformers there,
# vLLM here): the two should be close. Far apart, something in this setup
# differs and the stress numbers below mean less.
echo "asking $LORA_NAME per $UNIT at $WIDTH px about the cows in $BOXES"
bench run --root "$ROOT" --base-url "$BASE_URL" --model "$LORA_NAME" --answer-now \
    --max-width "$WIDTH" --unit "$UNIT" --concurrency "$( [ "$UNIT" = frame ] && echo 8 || echo 16 )" \
    "${BOX_ARGS[@]}"
bench score
bench score --vote
bench report

if [ "$MUSE_DETECT" = "1" ]; then
    step "6b  The base model finding the cows by itself (with reasoning)"
    bench detect --root "$ROOT" --base-url "$BASE_URL" --model muse-glimmer --concurrency 8
    bench detect-score
fi

if [ "$LIVE" = 1 ]; then
    step "7/7  $STREAMS cameras at once: RT-DETRv2 live on every frame, then $LORA_NAME, one GPU"
else
    step "7/7  $STREAMS cameras at once, on the detector's boxes from file"
fi
stress() {   # skip a test whose report is already there
    local tag="$1"; shift
    if [ -f "$OUT/$tag.md" ]; then echo "done before: $OUT/$tag.md"; return; fi
    "${STRESS_PY[@]}" cowbench.py --out "$OUT" stress --root "$ROOT" --base-url "$BASE_URL" \
        --streams "$STREAMS" --duration "$DURATION" "$@"
}
task=classify; [ "$UNIT" = frame ] && task=frame
if [ "$LIVE" = 1 ]; then
    # The detector's venv: it has torch. The detector loads in this process,
    # on the GPU next to vLLM, in the memory vLLM was told to leave free.
    STRESS_PY=("$DET_PY"); live_tag=_live
    CAM_ARGS=(--live-detector "$DET_DIR/best")
    [ -n "$DET_THRESHOLD" ] && CAM_ARGS+=(--det-threshold "$DET_THRESHOLD")
else
    STRESS_PY=("$PY"); live_tag=""; CAM_ARGS=("${BOX_ARGS[@]}")
fi
stress "stress_${task}_${STREAMS}x_max${live_tag}" --task "$task" --model "$LORA_NAME" --answer-now \
    --max-width "$WIDTH" "${CAM_ARGS[@]}"
stress "stress_${task}_${STREAMS}x_${INTERVAL}s${live_tag}" --task "$task" --model "$LORA_NAME" --answer-now \
    --max-width "$WIDTH" --interval "$INTERVAL" "${CAM_ARGS[@]}"
STRESS_PY=("$PY")
[ "$MUSE_DETECT" = "1" ] && stress "stress_detect_${STREAMS}x_max" --task detect --model muse-glimmer

tarball="$WORK/cow_tests_$(basename "$OUT").tgz"
tar czf "$tarball" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Done. Reports in $OUT:"
echo "  det_report.md                     how many cows RT-DETRv2 found"
echo "  report.md                         bench with $LORA_NAME per $UNIT at $WIDTH px on its boxes;"
echo "                                    missed cows count as errors, extras in detections_answered.jsonl"
[ "$MUSE_DETECT" = "1" ] && echo "  detect_report.md                  the base model finding the cows by itself"
echo "  stress_*.md                       $STREAMS cameras at once$( [ "$LIVE" = 1 ] && echo ', the detector live on every frame (_live)')"
echo "Everything in one file: $tarball"
