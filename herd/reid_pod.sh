#!/usr/bin/env bash
# The identity model on a GPU pod (herd/reid.py): public sets with real cow ids -> a fingerprint that knows
# a cow across days and cameras -> checked on MmCows' four cameras (mct_pod.sh) with it.
#
#   1. the environment (run_pod.sh step 1) and the models from herd_models_*.tgz (export_pod.sh) in WORK
#   2. fetch: MultiCamCows2024's crops (90 cows, 3 cameras, 7 days; 3.6 GB of its 36.5 GB zip),
#      Cows2021's identification test set (181 cows), OpenCows2020's identification part (46 cows)
#   3. extract (GPU): DINOv2 on every crop as the barn does; pseudo-bursts with cow ids; a sixth of
#      each set's cows held out (never trained on)
#   4. train: a copy of RUN's temporal transformer taught on the cow ids (other days, other cameras,
#      single frames) -> WORK/reid/id_model/identity.pt + abstain.json + eval.json (before / after)
#   5. MmCows (mct_pod.sh, IDENTITY=it): the same trackers and the look across cameras, now with it
#   6. WORK/export/herd_reid_<date>.tgz: id_model/, the logs, MmCows' mct_eval.json and show/
#
#   bash herd/reid_pod.sh          start in tmux "herd-reid"
#   bash herd/reid_pod.sh log      follow it
#   bash herd/reid_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<best/> (MmCows step)  SOURCES="multicam cows2021 opencows"
#        EPOCHS=30  MCT=1 (0: skip the MmCows step)  MINUTES=60 (MmCows)
# Every step resumes. Licences: all three sets are non-commercial (NCGL), as MmCows (CC BY-NC-SA).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
RUN="${RUN:-run9}"
H="$WORK/herd"
R="$WORK/reid"
LOG="$R/log.txt"
SESSION=herd-reid

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_REID_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[reid finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$R"
    knobs=""
    for v in WORK RUN DET SOURCES EPOCHS MCT MINUTES MODELS; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_REID_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/reid_pod.sh"); echo; echo '[reid finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$R"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
PY="$H/.venv/bin/python"
export HF_HOME="$WORK/hf-cache"
export PATH="$HOME/.local/bin:$PATH"
SOURCES="${SOURCES:-multicam cows2021 opencows}"

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

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

step "Fetch the crops with cow ids ($SOURCES)"
"$PY" "$HERE/reid.py" fetch --data "$R" --sources $SOURCES

step "Extract: DINOv2 on every crop, pseudo-bursts (GPU)"
"$PY" "$HERE/reid.py" extract --data "$R" --sources $SOURCES --model "$H/$RUN/model.pt"

step "Train the identity model"
"$PY" "$HERE/reid.py" train --data "$R" --sources $SOURCES --model "$H/$RUN/model.pt" --epochs "${EPOCHS:-30}"

if [ "${MCT:-1}" = "1" ]; then
    step "MmCows with the identity model (mct_pod.sh)"
    HERD_MCT_IN_TMUX=1 WORK="$WORK" RUN="$RUN" DET="${DET:-}" MINUTES="${MINUTES:-60}" \
        IDENTITY="$R/id_model/identity.pt" bash "$HERE/mct_pod.sh" || echo "!! the MmCows step failed (see above)"
fi

mkdir -p "$WORK/export"
T="$WORK/export/herd_reid_$(date +%Y%m%d_%H%M).tgz"
parts=()
for p in "$R/id_model" "$LOG" "$WORK/mct/mct_eval.json" "$WORK/mct/log.txt" "$WORK/mct/show"; do
    [ -e "$p" ] && parts+=("${p#"$WORK"/}")
done
(cd "$WORK" && tar -czf "$T" "${parts[@]}")
echo "identity model + results: $T   (scp -P <port> root@<ip>:$T .)"
echo "in the barn: [model] identity = \".../reid/id_model/identity.pt\" (abstain.json next to it is used)"
