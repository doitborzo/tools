#!/usr/bin/env bash
# One track per cow across cameras (mct.py), scored on MmCows' four synchronised cameras (herd/mct_eval.py).
#
#   1. the environment (run_pod.sh step 1) and the models from herd_models_*.tgz (export_pod.sh) in WORK
#      Nothing is trained and no other dataset is fetched: a cow's frame id is RUN's own transformer
#      on a burst of one frame (model.frame_fingerprint)
#   2. fetch: MmCows 7/25 from 12:00 + START min, MINUTES (60; up to 180) at a frame every STEP s,
#      4 cameras (~0.7 MB a frame read, ~0.3 MB kept), and the 15 s labels with the cows' numbers
#   3. embed (GPU): detector -> crops -> DINOv2 -> frame heads, every frame of every camera
#   4. score (CPU): the IoU tracker per camera, GlobalTracker per camera (+ cows lying still held by
#      their place), and across cameras by the floor point (MmCows' projection matrices; body height
#      from p(lying)); cannot_link with evidence; new cows grouped by gallery.enroll
#   5. show: SHOW_MIN minutes (10) of the four cameras and the floor as a video and stills - every box
#      in its track's colour, the cow the track mostly is and the true one (red when they differ);
#      SHOW_VARIANT (default: the best tracker of the table)
#   6. WORK/export/herd_mct_<date>.tgz: mct_eval.json, the log, show/ (ids.mp4 - VLC; index.html)
#
#   bash herd/mct_pod.sh          start in tmux "herd-mct"
#   bash herd/mct_pod.sh log      follow it
#   bash herd/mct_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<best/> (default: the newest)  MINUTES=60  START=0  STEP=1
#        SHOW_MIN=10  SHOW_VARIANT=<a tracker of the table>  APPEARANCE=1 (also hand over by look)
#        IDENTITY=<reid.py's identity.pt>: the frame id from the identity model (embed is redone for it)
# An older run's frames are kept; its embeddings are redone once (they had no posture).
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
    for v in WORK RUN DET MINUTES START STEP SHOW_MIN SHOW_VARIANT APPEARANCE IDENTITY MODELS; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_MCT_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/mct_pod.sh"); echo; echo '[mct finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$M"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "!! $(basename "$0") stopped at line $LINENO: $BASH_COMMAND (exit $?)"' ERR   # never silently
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

step "Fetch MmCows (7/25, ${MINUTES:-60} min from 12:00 + ${START:-0} min, a frame every ${STEP:-1} s)"
"$PY" "$HERE/mct_eval.py" fetch --data "$M" --minutes "${MINUTES:-60}" --start "${START:-0}" --step "${STEP:-1}"

step "Embed: detector -> DINOv2 -> frame heads (GPU)"
idm=(); [ -n "${IDENTITY:-}" ] && idm=(--identity "$IDENTITY")
"$PY" "$HERE/mct_eval.py" embed --data "$M" --model "$H/$RUN/model.pt" --detector "$DET" "${idm[@]}"

step "Score"
app=(); [ "${APPEARANCE:-0}" = "1" ] && app=(--appearance)
"$PY" "$HERE/mct_eval.py" score --data "$M" "${app[@]}"

step "Show: who is who, drawn"
V="${SHOW_VARIANT:-$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['data']['best'])" "$M/mct_eval.json")}"
"$PY" "$HERE/mct_eval.py" show --data "$M" --variant "$V" --minutes "${SHOW_MIN:-10}" "${app[@]}" \
    || echo "!! show failed (above); the scores are in $M/mct_eval.json"

mkdir -p "$WORK/export"
T="$WORK/export/herd_mct_$(date +%Y%m%d_%H%M).tgz"
parts=()
for p in "$M/mct_eval.json" "$LOG" "$M/show"; do
    [ -e "$p" ] && parts+=("${p#"$WORK"/}")
done
(cd "$WORK" && tar -czf "$T" "${parts[@]}")
echo "results: $T   (scp -P <port> root@<ip>:$T .)"
