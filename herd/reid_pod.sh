#!/usr/bin/env bash
# The identity model on a GPU pod (herd/reid.py): sets with real cow ids -> a fingerprint that knows a cow
# across days, postures and cameras -> checked on MmCows' four cameras (mct_pod.sh) with it.
#
#   1. the environment (run_pod.sh step 1), the models from herd_models_*.tgz (export_pod.sh) and the
#      identity model already trained from herd_reid_*.tgz (this script's earlier export) in WORK
#   2. fetch: the crops of the sets in SOURCES (Zenodo's zips are downloaded whole, the others read in
#      place); Zenodo, Bristol and the last step's MmCows frames download in the background while
#      MmCows' training days are fetched one at a time, each labelled (GPU) as the next downloads
#        mmcows    MmCows (the evaluation's own pen and cameras): 7/25 with its human labels outside
#                  mct_eval's 12:00-15:00 (cows 1-10 train, 11-16 held out), and 3 other days every 30 s
#        cobra     CoBRA youngstock: 48 heifers from a roof camera, standing and lying (733 MB)
#        sideview  SideViewCows2026: 110 cows from the side, parlor / barn / snapshots (25 GB of zips)
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
#        UWB_DAYS="0722 0728 0801"  UWB_EVERY=30  WAIT_MIN=20 (how long the GPU waits for a background download)
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
    for v in WORK RUN DET SOURCES EPOCHS MCT MINUTES MODELS INIT UWB_DAYS UWB_EVERY WAIT_MIN; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
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

# The downloads overlap the GPU: Zenodo's zips (whole, ~26 GB) and the Bristol crops come in the
# background, while MmCows is fetched a day at a time and each day is labelled (GPU) as the next downloads.
want() { [[ " $SOURCES " == *" $1 "* ]]; }
bg_fetch() {   # name, sources... -> a background fetch with its own log; its pid in BG_<name>
    local name="$1"; shift
    local srcs=()
    for s_ in "$@"; do want "$s_" && srcs+=("$s_"); done
    [ ${#srcs[@]} -gt 0 ] || return 0
    echo "fetching ${srcs[*]} in the background (log: $R/fetch_$name.log)"
    "$PY" "$HERE/reid.py" fetch --data "$R" --sources "${srcs[@]}" > "$R/fetch_$name.log" 2>&1 &
    printf -v "BG_$name" '%s' "$!"
}
bg_wait() {    # name, sources... -> wait for it (at most WAIT_MIN minutes, its progress each minute), then
               # extract the sets it finished; one not finished (failed, or too slow) is left out of this
               # training - the next run of this script resumes its download
    local name="$1"; shift
    local pid_var="BG_$name"
    local pid="${!pid_var:-}"
    [ -n "$pid" ] || return 0
    step "Waiting for the background fetch of $* (at most ${WAIT_MIN:-20} min; log $R/fetch_$name.log)"
    local waited=0 ok=1
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "${WAIT_MIN:-20}" ]; then
            echo "!! $* still downloading after ${WAIT_MIN:-20} min: training goes on without what is not in yet"
            echo "   (the download stops here and resumes where it was on the next run of this script)"
            kill "$pid" 2>/dev/null; sleep 2; pkill -P "$pid" 2>/dev/null || true
            ok=0
            break
        fi
        sleep 60
        waited=$((waited + 1))
        echo "  ${waited} min: $(tail -c 300 "$R/fetch_$name.log" | tr '\r' '\n' | grep -v '^ *$' | tail -n 1)"
    done
    if [ "$ok" = 1 ] && ! wait "$pid"; then
        echo "!! the fetch of $* failed (the end of $R/fetch_$name.log below)"
        tail -n 15 "$R/fetch_$name.log" | tr '\r' '\n' | grep -v '^ *$' | tail -n 15
    fi
    local srcs=()
    for s_ in "$@"; do
        want "$s_" && [ -f "$R/raw/$s_/index.jsonl" ] && [ -f "$R/raw/$s_/complete.json" ] && srcs+=("$s_")
    done
    if [ ${#srcs[@]} -gt 0 ]; then
        echo "in: ${srcs[*]}"
        "$PY" "$HERE/reid.py" extract --data "$R" --sources "${srcs[@]}" --model "$H/$RUN/model.pt"
    else
        echo "none of $* is in: training goes on without them"
    fi
}

step "Fetch the crops with cow ids ($SOURCES)"
bg_fetch zenodo cobra sideview
bg_fetch bristol multicam cows2021 opencows
MCT_PID=""
if [ "${MCT:-1}" = "1" ]; then                      # the last step's frames (mct_pod.sh fetches only what is missing)
    echo "fetching MmCows' frames for the check in the background (log: $R/fetch_mct.log)"
    "$PY" "$HERE/mct_eval.py" fetch --data "$WORK/mct" --minutes "${MINUTES:-60}" > "$R/fetch_mct.log" 2>&1 &
    MCT_PID=$!
fi

if want mmcows; then
    [ -n "${DET:-}" ] && [ -f "$DET/det_train_meta.json" ] || { echo "!! no detector best/ under $H (DET=<it>)"; exit 1; }
    echo "detector $DET"
    DAYS=(${UWB_DAYS:-0722 0728 0801})
    mm_fetch() { "$PY" "$HERE/reid.py" fetch --data "$R" --sources mmcows --mm-uwb-days "$@" \
                     --mm-uwb-every "${UWB_EVERY:-30}"; }
    mm_label() { "$PY" "$HERE/reid.py" label --data "$R" --model "$H/$RUN/model.pt" --detector "$DET" --mm-days "$@"; }
    step "MmCows 0725 (human labels): fetch"
    mm_fetch                                        # 7/25 only (no other day)
    prev=0725
    for d in "${DAYS[@]}"; do
        step "MmCows: label $prev (GPU) while $d downloads"
        mm_label "$prev" &
        lpid=$!
        mm_fetch "$d"
        wait "$lpid"
        prev="$d"
    done
    step "MmCows: label $prev (GPU)"
    mm_label "$prev"
    step "Extract MmCows: DINOv2 on every crop, pseudo-bursts (GPU)"
    "$PY" "$HERE/reid.py" extract --data "$R" --sources mmcows --model "$H/$RUN/model.pt"
fi

bg_wait bristol multicam cows2021 opencows
bg_wait zenodo cobra sideview

step "Train the identity model ${initarg[*]:-(fresh)}"
"$PY" "$HERE/reid.py" train --data "$R" --sources $SOURCES --model "$H/$RUN/model.pt" --epochs "${EPOCHS:-20}" \
    --out "$OUT" "${initarg[@]}"

if [ "${MCT:-1}" = "1" ]; then
    step "MmCows with the identity model (mct_pod.sh)"
    [ -z "$MCT_PID" ] || wait "$MCT_PID" || echo "!! the background fetch of the check's frames failed: mct_pod.sh retries"
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
