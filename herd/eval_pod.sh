#!/usr/bin/env bash
# The MmCows check again on a fresh pod, with an identity model already trained: no training, none of
# the training sets (Bristol, CoBRA, SideViewCows, the neck-tag days) fetched.
#
#   1. the models: herd_models_*.tgz (export_pod.sh) and herd_reid_*.tgz (reid_pod.sh's export: its
#      reid/id_model_all/ - identity.pt, abstain.json, thresholds.json) in WORK; the environment
#   2. CALIBRATE=1 (default): the identity model's thresholds again with today's reid.calibrate - on
#      MmCows 7/25 only, outside mct_eval's 12:00-15:00 (cows 11-16, never trained on): its labelled
#      frames every 60 s (~4 GB read), the detector on them, DINOv2, `reid.py calibrate`. The old
#      thresholds.json is kept as thresholds.old.json. CALIBRATE=0: the archive's thresholds as they are
#   3. the check (mct_pod.sh with IDENTITY=it): MmCows 7/25 12:00 + MINUTES, every tracker, the look
#      across cameras (cows 1-10 / 11-16), new cows grouped, the video of who is who
#   4. WORK/export/herd_eval_<date>.tgz: mct_eval.json, thresholds (new and old), the logs, show/
#
#   bash herd/eval_pod.sh          start in tmux "herd-eval"
#   bash herd/eval_pod.sh log      follow it
#   bash herd/eval_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<best/>  IDENTITY=<identity.pt> (default: from herd_reid_*.tgz)
#        CALIBRATE=1  MINUTES=60  SHOW_MIN=10  SHOW_VARIANT=<tracker>
# Every step resumes. MmCows: CC BY-NC-SA 4.0 (non-commercial).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
RUN="${RUN:-run9}"
H="$WORK/herd"
R="$WORK/reid"
E="$WORK/eval"
LOG="$E/log.txt"
SESSION=herd-eval

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_EVAL_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[eval finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$E"
    knobs=""
    for v in WORK RUN DET IDENTITY CALIBRATE MINUTES SHOW_MIN SHOW_VARIANT MODELS; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_EVAL_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/eval_pod.sh"); echo; echo '[eval finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$E"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
PY="$H/.venv/bin/python"
export HF_HOME="$WORK/hf-cache"
export PATH="$HOME/.local/bin:$PATH"

step "Models from the uploaded archives"
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

if [ -z "${IDENTITY:-}" ]; then
    for name in id_model_all id_model_ft id_model; do            # already unpacked by an earlier run
        [ -f "$R/$name/identity.pt" ] && { IDENTITY="$R/$name/identity.pt"; break; }
    done
fi
if [ -z "${IDENTITY:-}" ]; then
    a="$(ls -t "$WORK"/herd_reid_*.tgz 2>/dev/null | head -1)"
    [ -n "$a" ] || { echo "!! no identity model: upload herd_reid_*.tgz to $WORK (or IDENTITY=<identity.pt>)"; exit 1; }
    echo "unpacking $a"
    U="$WORK/_unpack_reid"
    rm -rf "$U" && mkdir -p "$U"
    tar -xzf "$a" -C "$U"
    m=""
    for name in id_model_all id_model_ft id_model; do
        m="$(find "$U" -path "*/$name/identity.pt" -print -quit)"
        [ -n "$m" ] && break
    done
    [ -n "$m" ] || m="$(find "$U" -name identity.pt -print -quit)"
    [ -n "$m" ] || { echo "!! no identity.pt in $a"; exit 1; }
    name="$(basename "$(dirname "$m")")"
    mkdir -p "$R/$name"
    cp -a "$(dirname "$m")/." "$R/$name/"
    rm -rf "$U"
    IDENTITY="$R/$name/identity.pt"
fi
[ -f "$IDENTITY" ] || { echo "!! $IDENTITY: no such file"; exit 1; }
ID_DIR="$(dirname "$IDENTITY")"
echo "identity model $IDENTITY"
if [ -z "${DET:-}" ]; then
    DET="$(find "$H" -maxdepth 3 -name det_train_meta.json -path '*/best/*' -printf '%T@ %h\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
fi
[ -n "$DET" ] && [ -f "$DET/det_train_meta.json" ] || { echo "!! no detector best/ under $H (DET=<it>)"; exit 1; }
echo "detector $DET"

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
if grep -qs '"kind": *"rfdetr"' "$DET/det_train_meta.json"; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "${CALIBRATE:-1}" = "1" ]; then
    step "Thresholds again: MmCows 7/25 outside mct_eval's hours (no other day, no other set)"
    # --mm-uwb-days with nothing after it: 7/25 alone (its labelled frames, every 60 s)
    "$PY" "$HERE/reid.py" fetch --data "$R" --sources mmcows --mm-uwb-days --mm-label-every 60
    "$PY" "$HERE/reid.py" label --data "$R" --model "$H/$RUN/model.pt" --detector "$DET" --mm-days 0725
    "$PY" "$HERE/reid.py" extract --data "$R" --sources mmcows --model "$H/$RUN/model.pt"
    if [ -f "$ID_DIR/thresholds.json" ] && [ ! -f "$ID_DIR/thresholds.old.json" ]; then
        cp "$ID_DIR/thresholds.json" "$ID_DIR/thresholds.old.json"
    fi
    "$PY" "$HERE/reid.py" calibrate --data "$R" --identity "$IDENTITY" --sources mmcows
else
    echo "CALIBRATE=0: the thresholds as in the archive ($ID_DIR/thresholds.json)"
fi

step "The MmCows check with the identity model (mct_pod.sh)"
HERD_MCT_IN_TMUX=1 WORK="$WORK" RUN="$RUN" DET="$DET" MINUTES="${MINUTES:-60}" SHOW_MIN="${SHOW_MIN:-10}" \
    SHOW_VARIANT="${SHOW_VARIANT:-}" IDENTITY="$IDENTITY" bash "$HERE/mct_pod.sh" \
    || echo "!! the MmCows check stopped (above); what it made is packed below"

mkdir -p "$WORK/export"
T="$WORK/export/herd_eval_$(date +%Y%m%d_%H%M).tgz"
parts=()
for p in "$WORK/mct/mct_eval.json" "$WORK/mct/log.txt" "$WORK/mct/show" "$ID_DIR/thresholds.json" \
         "$ID_DIR/thresholds.old.json" "$R/mmcows/label.json" "$LOG"; do
    [ -e "$p" ] && parts+=("${p#"$WORK"/}")
done
(cd "$WORK" && tar -czf "$T" "${parts[@]}")
echo "results: $T   (scp -P <port> root@<ip>:$T .)"
