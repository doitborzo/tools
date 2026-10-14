#!/usr/bin/env bash
# One track per cow across cameras (mct.py), scored on MmCows' four synchronised cameras (herd/mct_eval.py).
#
#   1. the environment (run_pod.sh step 1) and the models from herd_models_*.tgz (export_pod.sh) in WORK
#   2. the frame id head (herd.py frame-id), if RUN's model.pt has none yet: it learns on CBVD-5's train
#      features - FEATURES=<folder> if they are on this pod; else FRAME_ID=1 fetches CBVD-5 and extracts
#      them (run_pod.sh setup + herd.py extract --split train: the longest part, ~1-2 h); FRAME_ID=0
#      skips it and only the DINOv2 baseline is scored. The model with the head is kept for the barn.
#   3. fetch: MmCows 7/25 from 12:00 + START min, MINUTES (60; up to 180) at a frame every STEP s,
#      4 cameras (~0.7 MB a frame read, ~0.3 MB kept), and the 15 s labels with the cows' numbers
#   4. embed (GPU): detector -> crops -> DINOv2 -> frame heads, every frame of every camera
#   5. score (CPU): IoU tracker per camera vs GlobalTracker per camera vs across cameras, frame id vs
#      DINOv2, a sweep of handover_min_sim; new cows grouped by gallery.enroll with / without cannot_link
#   6. WORK/export/herd_mct_<date>.tgz: mct_eval.json, the log, eval_frame_id.json and RUN's model.pt
#      (with the head, for [mct] in the barn)
#
#   bash herd/mct_pod.sh          start in tmux "herd-mct"
#   bash herd/mct_pod.sh log      follow it
#   bash herd/mct_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<best/> (default: the newest)  MINUTES=60  START=0  STEP=1
#        FEATURES=<CBVD-5 train features>  FRAME_ID=auto|1|0  SWEEP="0.3 0.4 0.6 0.7 0.8"
# MmCows: CC BY-NC-SA 4.0 (non-commercial). Every step resumes (frames fetched and cameras embedded are kept).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
RUN="${RUN:-run9}"
H="$WORK/herd"
M="$WORK/mct"
LOG="$M/log.txt"
SESSION=herd-mct

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_MCT_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[mct finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$M"
    knobs=""
    for v in WORK RUN DET MINUTES START STEP FEATURES FRAME_ID SWEEP MODELS; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_MCT_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/mct_pod.sh"); echo; echo '[mct finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$M"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
PY="$H/.venv/bin/python"
export HF_HOME="$WORK/hf-cache"
export PATH="$HOME/.local/bin:$PATH"

step "Models from the uploaded archive"
for a in ${MODELS:-$(ls -t "$WORK"/herd_models_*.tgz 2>/dev/null | head -1)}; do
    [ -f "$a" ] || continue
    [ -f "$H/$RUN/model.pt" ] && [ -z "${MODELS:-}" ] && { echo "$H/$RUN/model.pt is there: $a not unpacked again"; break; }
    echo "unpacking $a"
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
echo "model    $H/$RUN/model.pt"
echo "detector $DET"

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
if grep -qs '"kind": *"rfdetr"' "$DET/det_train_meta.json"; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

step "Frame id head"
# the model's own feature settings: CBVD-5 features must be extracted the same way
read -r HAS_ID ENC GRID MARGIN < <("$PY" - "$H/$RUN/model.pt" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
f = ck.get("features", {})
print(int(bool(ck["kw"].get("frame_id"))), f.get("encoder", "facebook/dinov2-small"), f.get("grid", 2), f.get("margin", 0.1))
EOF
)
if [ "$HAS_ID" = "1" ]; then
    echo "the model has it"
else
    F="${FEATURES:-$H/features}"
    [ -n "${FEATURES:-}" ] || { [ "$MARGIN" = "0.1" ] || F="${F}_m$MARGIN"; [ "$GRID" = "2" ] || F="${F}_g$GRID"; }
    if [ ! -f "$F/train/index.jsonl" ] && [ "${FRAME_ID:-auto}" = "1" ]; then
        echo "no CBVD-5 train features at $F: fetching CBVD-5 and extracting them ($ENC, grid $GRID, margin $MARGIN)"
        HERD_IN_TMUX=1 SETUP_ONLY=1 WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
        "$PY" "$HERE/herd.py" extract --root "$WORK/cbvd5" --out "$F" --split train --encoder "$ENC" --grid "$GRID" --margin "$MARGIN"
    fi
    if [ -f "$F/train/index.jsonl" ] && [ "${FRAME_ID:-auto}" != "0" ]; then
        "$PY" "$HERE/herd.py" frame-id --features "$F" --out "$H/$RUN"
    else
        echo "!! no frame id head and no CBVD-5 train features ($F): only the DINOv2 baseline is scored."
        echo "   FEATURES=<folder> if they are elsewhere, or FRAME_ID=1 to fetch CBVD-5 and extract them here."
    fi
fi

step "Fetch MmCows (7/25, ${MINUTES:-60} min from 12:00 + ${START:-0} min, a frame every ${STEP:-1} s)"
"$PY" "$HERE/mct_eval.py" fetch --data "$M" --minutes "${MINUTES:-60}" --start "${START:-0}" --step "${STEP:-1}"

step "Embed: detector -> DINOv2 -> frame heads (GPU)"
"$PY" "$HERE/mct_eval.py" embed --data "$M" --model "$H/$RUN/model.pt" --detector "$DET"

step "Score"
"$PY" "$HERE/mct_eval.py" score --data "$M" --sweep ${SWEEP:-0.3 0.4 0.6 0.7 0.8}

mkdir -p "$WORK/export"
T="$WORK/export/herd_mct_$(date +%Y%m%d_%H%M).tgz"
parts=()
for p in "$M/mct_eval.json" "$LOG" "$H/$RUN/eval_frame_id.json" "$H/$RUN/model.pt" "$H/$RUN/abstain.json" "$H/$RUN/heads.json"; do
    [ -e "$p" ] && parts+=("${p#"$WORK"/}")
done
(cd "$WORK" && tar -czf "$T" "${parts[@]}")
echo "results + model: $T   (scp -P <port> root@<ip>:$T .)"
