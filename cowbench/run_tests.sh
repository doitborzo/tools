#!/usr/bin/env bash
# Everything after training, on a fresh pod, in one go:
#
#   dataset -> adapter + detections -> vLLM with the adapter -> bench with the adapter
#   -> 12 cameras at once -> one tar.gz
#
# The cows are those RT-DETRv2 found (lora-runs/detector, made by run_lora.sh),
# never the annotation's boxes: the model answers about what the detector
# found, the annotation only scores, and a cow the detector missed is an error.
#
# Usage (the repo checked out, the adapter and the detections uploaded - see
# ADAPTER and BOXES below):
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
#   ADAPTER=$WORK/lora-runs/lora4frame_w1920/adapter
#   LORA_NAME=lora4frame            the adapter's model name on the server
#   BOXES=$WORK/lora-runs/detector/val_detections.jsonl   the detector's boxes, with
#                                   det_meta.json (threshold) and det_report.md beside it
#   DET_THRESHOLD                   override the detector's score threshold
#   MUSE_DETECT=0                   1: also let the base model find the cows by itself
#                                   (the earlier test: 53% found) and stress that
#   WIDTH, UNIT                     how the adapter is asked: by default as it was trained,
#                                   read from the train_meta.json next to the adapter
#                                   (else 896, cow). UNIT=frame: one question per keyframe
#                                   about every cow on it (frame.py), stress task "frame"
#   STREAMS=12                      cameras in the stress tests
#   INTERVAL=10                     seconds between frames per camera, paced test
#   DURATION=300                    seconds measured per stress test
#   OUT=<repo>/cowbench/runs/tests_<LORA_NAME>_w<WIDTH>_det   results
#   PORT=8000

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
ROOT="$WORK/cbvd5"
ADAPTER="${ADAPTER:-$WORK/lora-runs/lora4frame_w1920/adapter}"
LORA_NAME="${LORA_NAME:-lora4frame}"
BOXES="${BOXES:-$WORK/lora-runs/detector/val_detections.jsonl}"
DET_THRESHOLD="${DET_THRESHOLD:-}"
MUSE_DETECT="${MUSE_DETECT:-0}"
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
OUT="${OUT:-$HERE/runs/tests_${LORA_NAME}_w${WIDTH}_det}"
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
    for v in WORK ADAPTER LORA_NAME BOXES DET_THRESHOLD MUSE_DETECT WIDTH UNIT STREAMS INTERVAL DURATION PORT OUT; do
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

step "1/6  Dataset"
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

step "2/6  Adapter and the detector's boxes"
if [ ! -f "$ADAPTER/adapter_config.json" ] || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "!! no adapter in $ADAPTER (adapter_config.json + adapter_model.safetensors)."
    echo "!! From a PC:  scp -P <port> -r <adapter folder> root@<ip>:$(dirname "$ADAPTER")/"
    exit 1
fi
ls -la "$ADAPTER"
DET_DIR="$(dirname "$BOXES")"
if [ ! -s "$BOXES" ] || [ ! -f "$DET_DIR/det_meta.json" ]; then
    echo "!! no detections in $BOXES (+ det_meta.json beside it)."
    echo "!! They come from run_lora.sh (step 9, RT-DETRv2), in lora-runs/detector on the"
    echo "!! training pod. Only the small files are needed, not the weights:"
    echo "!!   scp -P <port> detector/val_detections.jsonl detector/det_meta.json detector/det_report.md \\"
    echo "!!       root@<ip>:$DET_DIR/"
    exit 1
fi
mkdir -p "$OUT"
cp "$DET_DIR/det_meta.json" "$OUT/"
[ -f "$DET_DIR/det_report.md" ] && cp "$DET_DIR/det_report.md" "$OUT/" && cat "$DET_DIR/det_report.md"
BOX_ARGS=(--boxes "$BOXES")
[ -n "$DET_THRESHOLD" ] && BOX_ARGS+=(--det-threshold "$DET_THRESHOLD")

step "3/6  vLLM with the adapter"
server_has_adapter() { curl -sf "$BASE_URL/v1/models" 2>/dev/null | grep -q "\"$LORA_NAME\""; }
if server_has_adapter; then
    echo "already serving $LORA_NAME on $BASE_URL"
else
    if tmux has-session -t vllm 2>/dev/null; then
        echo "a tmux session 'vllm' exists but does not serve $LORA_NAME - restarting it"
        tmux kill-session -t vllm
    fi
    rm -f "$WORK/vllm.exit"
    # lora_muse.sh may ask "Continue anyway?" about open files; that concerns
    # data-parallel, which one GPU does not use, so the answer is yes.
    tmux new-session -d -s vllm -x 200 -y 50 \
        "yes y | env LORA_MODULES=$(printf '%q' "$LORA_NAME=$ADAPTER") bash $(printf '%q' "$REPO/lora_muse.sh") 2>&1 | tee -a $WORK/vllm.log; echo \$? > $WORK/vllm.exit; exec bash"
    echo "starting vLLM in tmux session 'vllm' (log: $WORK/vllm.log)."
    echo "First start installs vLLM and downloads the model (~33 GB): up to an hour."
    t0=$SECONDS
    until server_has_adapter; do
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
fi
curl -s "$BASE_URL/v1/models" | grep -o '"id": *"[^"]*"' | sed 's/"id": */  model: /'

step "4/6  Python for the bench"
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

step "5/6  Bench with the adapter on the detector's boxes"
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
    step "5b  The base model finding the cows by itself (with reasoning)"
    bench detect --root "$ROOT" --base-url "$BASE_URL" --model muse-glimmer --concurrency 8
    bench detect-score
fi

step "6/6  $STREAMS cameras at once, on the detector's boxes"
stress() {   # skip a test whose report is already there
    local tag="$1"; shift
    if [ -f "$OUT/$tag.md" ]; then echo "done before: $OUT/$tag.md"; return; fi
    bench stress --root "$ROOT" --base-url "$BASE_URL" --streams "$STREAMS" \
        --duration "$DURATION" "$@"
}
task=classify; [ "$UNIT" = frame ] && task=frame
stress "stress_${task}_${STREAMS}x_max" --task "$task" --model "$LORA_NAME" --answer-now --max-width "$WIDTH" \
    "${BOX_ARGS[@]}"
stress "stress_${task}_${STREAMS}x_${INTERVAL}s" --task "$task" --model "$LORA_NAME" --answer-now --max-width "$WIDTH" \
    --interval "$INTERVAL" "${BOX_ARGS[@]}"
[ "$MUSE_DETECT" = "1" ] && stress "stress_detect_${STREAMS}x_max" --task detect --model muse-glimmer

tarball="$WORK/cow_tests_$(basename "$OUT").tgz"
tar czf "$tarball" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Done. Reports in $OUT:"
echo "  det_report.md                     how many cows RT-DETRv2 found"
echo "  report.md                         bench with $LORA_NAME per $UNIT at $WIDTH px on its boxes;"
echo "                                    missed cows count as errors, extras in detections_answered.jsonl"
[ "$MUSE_DETECT" = "1" ] && echo "  detect_report.md                  the base model finding the cows by itself"
echo "  stress_*.md                       $STREAMS cameras at once"
echo "Everything in one file: $tarball"
