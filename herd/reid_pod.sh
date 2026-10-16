#!/usr/bin/env bash
# The identity model on a GPU pod (herd/reid.py): sets with real cow ids -> a fingerprint that knows a cow
# across days, postures and cameras -> checked on MmCows' four cameras (mct_pod.sh) with it.
#
#   1. the environment (run_pod.sh step 1), the models from herd_models_*.tgz (export_pod.sh) and the
#      identity model already trained from herd_reid_*.tgz (this script's earlier export) in WORK
#   2. fetch: the crops of the sets in SOURCES, read in place from their zips
#        mmcows    MmCows (the evaluation's own pen and cameras): 7/25 with its human labels outside
#                  mct_eval's 12:00-15:00 (cows 1-10 train, 11-16 held out), and 3 other days every 30 s
#        cobra     CoBRA youngstock: 48 heifers from a roof camera, standing and lying (733 MB)
#        sideview  SideViewCows2026: 110 cows from the side, parlor / barn / snapshots (~6 GB read)
#        multicam cows2021 opencows   the top-down sets the first model was trained on
#   3. label (GPU, mmcows): the detector on its frames; 7/25's boxes by the human labels, the other
#      days' by the cows' UWB neck tags (tracks voted by the tag's position, ~95% right on 7/25)
#   4. extract (GPU): DINOv2 on every crop as the barn does; pseudo-bursts with cow ids
#   5. train: the identity model fine-tuned (INIT, default reid/id_model/identity.pt) on all of them
#      -> WORK/reid/id_model_ft/identity.pt + abstain.json + eval.json (before / after); without an
#      INIT, a fresh one from RUN's transformer -> reid/id_model/
#   6. MmCows (mct_pod.sh, IDENTITY=it): the trackers and the look across cameras - cows 1-10 (trained
#      on, other hours) and 11-16 (never seen) apart
#   7. WORK/export/herd_reid_<date>.tgz: the new id model, the logs, MmCows' mct_eval.json and show/
#
#   bash herd/reid_pod.sh          start in tmux "herd-reid"
#   bash herd/reid_pod.sh log      follow it
#   bash herd/reid_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<best/> (default: the newest)  INIT=<identity.pt> (none: fresh)
#        SOURCES="mmcows cobra sideview multicam cows2021 opencows"  EPOCHS=20  MCT=1  MINUTES=60
#        UWB_DAYS="0722 0728 0801"  UWB_EVERY=30
# Every step resumes. Licences: MmCows CC BY-NC-SA, CoBRA and SideViewCows CC BY 4.0, the Bristol sets NCGL.

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
    for v in WORK RUN DET SOURCES EPOCHS MCT MINUTES MODELS INIT UWB_DAYS UWB_EVERY; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
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
SOURCES="${SOURCES:-mmcows cobra sideview multicam cows2021 opencows}"

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

step "The identity model to fine-tune"
if [ -z "${INIT:-}" ] && [ ! -f "$R/id_model/identity.pt" ]; then
    a="$(ls -t "$WORK"/herd_reid_*.tgz 2>/dev/null | head -1)"
    if [ -n "$a" ]; then
        echo "unpacking $a"
        U="$WORK/_unpack_reid"
        rm -rf "$U" && mkdir -p "$U"
        tar -xzf "$a" -C "$U"
        m="$(find "$U" -path '*id_model/identity.pt' -not -path '*id_model_ft*' -print -quit)"
        [ -n "$m" ] || m="$(find "$U" -name identity.pt -print -quit)"
        [ -n "$m" ] && { mkdir -p "$R/id_model"; cp -a "$(dirname "$m")/." "$R/id_model/"; }
        rm -rf "$U"
    fi
fi
INIT="${INIT:-$R/id_model/identity.pt}"
if [ -f "$INIT" ]; then
    echo "fine-tuning $INIT"
    OUT="$R/id_model_ft"
    initarg=(--init "$INIT")
else
    echo "no identity model (upload herd_reid_*.tgz to $WORK, or INIT=<identity.pt>): training a fresh one"
    OUT="$R/id_model"
    initarg=()
fi
if [ -z "${DET:-}" ]; then
    DET="$(find "$H" -maxdepth 3 -name det_train_meta.json -path '*/best/*' -printf '%T@ %h\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
fi

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
if [ -n "${DET:-}" ] && grep -qs '"kind": *"rfdetr"' "$DET/det_train_meta.json"; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

step "Fetch the crops with cow ids ($SOURCES)"
"$PY" "$HERE/reid.py" fetch --data "$R" --sources $SOURCES --mm-uwb-days ${UWB_DAYS:-0722 0728 0801} \
    --mm-uwb-every "${UWB_EVERY:-30}"

if [[ " $SOURCES " == *" mmcows "* ]]; then
    step "Label MmCows: the detector's boxes by the human labels (7/25) and the neck tags (other days) (GPU)"
    [ -n "${DET:-}" ] && [ -f "$DET/det_train_meta.json" ] || { echo "!! no detector best/ under $H (DET=<it>)"; exit 1; }
    echo "detector $DET"
    "$PY" "$HERE/reid.py" label --data "$R" --model "$H/$RUN/model.pt" --detector "$DET"
fi

step "Extract: DINOv2 on every crop, pseudo-bursts (GPU)"
"$PY" "$HERE/reid.py" extract --data "$R" --sources $SOURCES --model "$H/$RUN/model.pt"

step "Train the identity model ${initarg[*]:-(fresh)}"
"$PY" "$HERE/reid.py" train --data "$R" --sources $SOURCES --model "$H/$RUN/model.pt" --epochs "${EPOCHS:-20}" \
    --out "$OUT" "${initarg[@]}"

if [ "${MCT:-1}" = "1" ]; then
    step "MmCows with the identity model (mct_pod.sh)"
    HERD_MCT_IN_TMUX=1 WORK="$WORK" RUN="$RUN" DET="${DET:-}" MINUTES="${MINUTES:-60}" \
        IDENTITY="$OUT/identity.pt" bash "$HERE/mct_pod.sh" || echo "!! the MmCows step failed (see above)"
fi

mkdir -p "$WORK/export"
T="$WORK/export/herd_reid_$(date +%Y%m%d_%H%M).tgz"
parts=()
for p in "$OUT" "$R/mmcows/label.json" "$LOG" "$WORK/mct/mct_eval.json" "$WORK/mct/log.txt" "$WORK/mct/show"; do
    [ -e "$p" ] && parts+=("${p#"$WORK"/}")
done
(cd "$WORK" && tar -czf "$T" "${parts[@]}")
echo "identity model + results: $T   (scp -P <port> root@<ip>:$T .)"
echo "in the barn: [model] identity = \".../reid/$(basename "$OUT")/identity.pt\" (abstain.json next to it is used)"
