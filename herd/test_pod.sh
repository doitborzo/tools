#!/usr/bin/env bash
# A new pod, a trained herd: how good the detector is and how fast the whole chain runs on this GPU
# (e.g. an RTX 2000 Ada, 16 GB). CBVD-5's labels are not used: its frames leave cows unboxed.
#
#   1. the environment (run_pod.sh step 1)
#   2. the models from herd_models_*.tgz (export_pod.sh) uploaded to WORK:
#      the detectors' best/, RUN's model.pt + abstain.json, DINOv2 (hf-cache)
#   3. the usual test - the detector on fully boxed frames: the valid frames of the public sets
#      (SETS=zenodo,mmcows,opencows; only valid is downloaded), every detector unpacked side by
#      side (extra_det.py eval), and a gallery of each set's worst frames for DET (show-dets)
#   4. the stress test - detector -> DINOv2 -> transformer on CAMERAS cameras in real time:
#      the barn's own videos when VIDEOS=<folder>, else CBVD-5's 50 val clips (248 MB, played as
#      cameras only; their labels are not scored)
#   5. a summary (GPU, recall / precision per set and detector, keeps up or not) and
#      WORK/export/herd_test_<gpu>_<date>.tgz
#
#   bash herd/test_pod.sh          start in tmux "herd-test"
#   bash herd/test_pod.sh log      follow it
#   bash herd/test_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<best/ for the stress test and the galleries> (default: newest)
#        SETS=zenodo,mmcows,opencows  GALLERY=1  VIDEOS=<folder of videos>  CAMERAS=5
#        DURATION=300 (s measured)  THRESHOLD (default: the detector's own)
#        TESTS="detector stress" (one of them to run only it)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
RUN="${RUN:-run9}"
H="$WORK/herd"
LOG="$H/log_test.txt"
SESSION=herd-test

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_TEST_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[test finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$H"
    knobs=""
    for v in WORK RUN DET CAMERAS DURATION THRESHOLD TESTS MODELS SETS GALLERY VIDEOS; do
        knobs+="$v=$(printf '%q' "${!v:-}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_TEST_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/test_pod.sh"); echo; echo '[test finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$H"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
TESTS="${TESTS:-detector stress}"
SETS="${SETS:-zenodo,mmcows,opencows}"
XDIR="$WORK/extra_det"
PY="$H/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"
STAMP="$(date +%Y%m%d_%H%M)"

step "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "!! no GPU"; exit 1; }
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GTAG="$(echo "$GPU" | tr -c 'A-Za-z0-9' '_' | tr -s _ | sed 's/^_//; s/_$//')"

step "Models from the uploaded archive"
for a in ${MODELS:-$(ls -t "$WORK"/herd_models_*.tgz 2>/dev/null | head -1)}; do
    [ -f "$a" ] || continue
    echo "unpacking $a"
    # an archive repacked elsewhere (e.g. on Windows) may wrap everything in a folder of its own:
    # unpack aside, find herd/ at whatever depth, and merge it (and hf-cache/) into WORK
    U="$WORK/_unpack"
    rm -rf "$U" && mkdir -p "$U"
    tar -xzf "$a" -C "$U"
    top="$(find "$U" -maxdepth 4 -type d -name herd -print -quit)"
    [ -n "$top" ] || { echo "!! no herd/ folder in $a"; exit 1; }
    top="$(dirname "$top")"
    for d in herd hf-cache; do
        [ -d "$top/$d" ] && { mkdir -p "$WORK/$d"; cp -a "$top/$d/." "$WORK/$d/"; }
    done
    rm -rf "$U"
done
[ -f "$H/$RUN/model.pt" ] || { echo "!! no $H/$RUN/model.pt - upload herd_models_*.tgz to $WORK (or RUN=<run>)"; exit 1; }
if [ -z "${DET:-}" ]; then
    DET="$(find "$H" -maxdepth 3 -name det_train_meta.json -path '*/best/*' -printf '%T@ %h\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
fi
[ -n "$DET" ] && [ -f "$DET/det_train_meta.json" ] || { echo "!! no detector best/ under $H (DET=<it>)"; exit 1; }
DTAG="$(basename "$(dirname "$DET")")"
echo "tests     $TESTS; stress: ${CAMERAS:-5} cameras, ${DURATION:-300} s${VIDEOS:+, videos from $VIDEOS}"
echo "model     $H/$RUN/model.pt"
echo "detector  $DET  (stress test, galleries)"
echo "all detectors here: $(cd "$H" && ls -d detector_*/best 2>/dev/null | tr '\n' ' ')"

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
if grep -qs '"kind": *"rfdetr"' "$H"/detector_*/best/det_train_meta.json; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
export HF_HOME="$WORK/hf-cache"
if ls -d "$HF_HOME"/hub/models--facebook--dinov2-* >/dev/null 2>&1; then
    echo "DINOv2 from the archive ($HF_HOME)"       # found in the cache: not downloaded again
fi
"$PY" -c "import torch; p = torch.cuda.get_device_properties(0); print(f'torch {torch.__version__}, {p.name}, {p.total_memory / 2**30:.1f} GB, bf16 {torch.cuda.is_bf16_supported()}')"

cd "$HERE"
EOUT="$H/test_eval_extra_${GTAG}.json"
GOUT="$H/test_galleries_${GTAG}"
SOUT="$H/stress_${RUN}_${DTAG}_${GTAG}_${CAMERAS:-5}cam_${STAMP}"     # one folder per run: no older result mixes in
thr=(); [ -n "${THRESHOLD:-}" ] && thr=(--threshold "$THRESHOLD")

if [[ " $TESTS " == *" detector "* ]]; then
    step "Usual test: the detectors on the public sets' valid frames ($SETS)"
    "$PY" "$REPO/cowbench/extra_det.py" fetch --dest "$XDIR" --sources "$SETS" --splits valid
    dets=()
    for b in "$H"/detector_*/best; do [ -f "$b/det_train_meta.json" ] && dets+=(--detector "$b"); done
    "$PY" "$REPO/cowbench/extra_det.py" eval --data "$XDIR" --sources "$SETS" "${dets[@]}" --out "$EOUT"
    if [ "${GALLERY:-1}" != "0" ]; then
        for s in ${SETS//,/ }; do
            "$PY" herd.py show-dets --src "$XDIR/$s/valid" --detector "$DET" --pick worst --max 100 \
                --out "$GOUT/$s" "${thr[@]}"
        done
    fi
fi

if [[ " $TESTS " == *" stress "* ]]; then
    step "Stress test: ${CAMERAS:-5} cameras, ${DURATION:-300} s"
    if [ -n "${VIDEOS:-}" ]; then
        src=(--videos "$VIDEOS")
    else
        if [ ! -f "$WORK/cbvd5/annotations/ava_val_v2.1.csv" ] || [ -z "$(ls "$WORK"/cbvd5/videos/videos/*.mp4 2>/dev/null | head -1)" ]; then
            "$PY" "$REPO/cowbench/extra_det.py" cbvd-videos --dest "$WORK/cbvd5"
        fi
        src=(--root "$WORK/cbvd5" --no-detector-check)
    fi
    mkdir -p "$SOUT"
    # exit 1 means "does not keep up" - a result, not a failure of the script
    "$PY" herd.py stress --model "$H/$RUN/model.pt" --detector "$DET" "${src[@]}" \
        --cameras "${CAMERAS:-5}" --duration "${DURATION:-300}" --out "$SOUT" "${thr[@]}" || true
fi

step "Summary"
SUM="$H/test_summary_${GTAG}_${STAMP}.md"
"$PY" "$HERE/test_summary.py" "$GPU" "$DET" "$EOUT" "$SOUT" "$SUM"
mkdir -p "$WORK/export"
T="$WORK/export/herd_test_${GTAG}_${STAMP}.tgz"
parts=()
for p in "$SUM" "$EOUT" "$GOUT" "$SOUT" "$LOG"; do [ -e "$p" ] && parts+=("${p#"$WORK"/}"); done
(cd "$WORK" && tar -czf "$T" "${parts[@]}")
echo
echo "summary: $SUM"
echo "archive: $T   (scp -P <port> root@<ip>:$T .)"
