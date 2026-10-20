#!/usr/bin/env bash
# The identity model on a GPU pod (herd/reid.py): sets with real cow ids -> a fingerprint that knows a cow
# across days, postures and cameras -> checked on MmCows' four cameras (mct_pod.sh) with it.
#
#   1. the environment (run_pod.sh step 1), the models from herd_models_*.tgz (export_pod.sh) in WORK
#   2. fetch: the crops of the sets in SOURCES (Zenodo's zips are downloaded whole, the others read in
#      place); Zenodo, WildlifeReID-10k and the last step's MmCows frames download in the background while
#      MmCows' training days are fetched one at a time, each labelled (GPU) as the next downloads
#        mmcows    MmCows (the evaluation's own pen and cameras): 7/25 with its human labels outside
#                  mct_eval's 12:00-15:00, every 60 s (cows 1-10 train, 11-16 held out: never trained on);
#                  the neck-tag days only with UWB_DAYS
#        cobra     CoBRA youngstock: 48 heifers from a roof camera, standing and lying (733 MB)
#        sideview  SideViewCows2026: 110 cows from the side, parlor / barn / snapshots (25 GB of zips)
#        multicam cows2021 opencows aerialcattle   the Bristol top-down sets, from WildlifeReID-10k (Kaggle):
#                  data.bris itself answers 404 since 2026-10
#        cowdataset  13 cows photographed from the side on 6 days (WildlifeReID-10k)
#   3. label (GPU, mmcows): the detector on its frames; 7/25's boxes by the human labels, the other
#      days' by the cows' UWB neck tags (tracks voted by the tag's position, ~95% right on 7/25)
#   4. extract (GPU): DINOv2 on every crop as the barn does; pseudo-bursts with cow ids
#   5. train: the identity model from scratch (a copy of RUN's temporal transformer) on all of them,
#      the best epoch by the held-out cows of every set and MmCows' cows 11-16 from another camera
#      -> WORK/reid/id_model_all/identity.pt + abstain.json + eval.json (before / after);
#      INIT=<identity.pt> fine-tunes that one instead -> reid/id_model_ft/
#   6. MmCows (mct_pod.sh, IDENTITY=it): the trackers and the look across cameras - cows 1-10 (trained
#      on, other hours) and 11-16 (never seen) apart
#   7. WORK/export/herd_reid_<date>.tgz: the new id model, the logs, MmCows' mct_eval.json and show/
#
#   bash herd/reid_pod.sh          start in tmux "herd-reid"
#   bash herd/reid_pod.sh log      follow it
#   bash herd/reid_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<best/> (default: the newest)  INIT=<identity.pt> (fine-tune it; default: from scratch)
#        SOURCES="mmcows cobra sideview multicam cows2021 opencows aerialcattle cowdataset"  EPOCHS=30  MCT=1
#        LABEL_EVERY=60 (s between 7/25's human-labelled frames; 15: every label MmCows has - but two runs at 15
#        knew cows 11-16 by look 32-35%, the run at 60 51%: MmCows' ten cows then outweigh the other sets)
#        UWB_DAYS="" (neck-tag days, e.g. "0722 0728 0801": off - ~4% of their labels are another cow, mostly a
#        held-out one)  UWB_EVERY=30  WAIT_MIN=20 (how long the GPU waits for a background download)
#        TRAIN=0 (no training: the existing model's thresholds again - reid.py calibrate - and the MmCows check)
#        REQUIRE="multicam cows2021" (no training if these are not in; REQUIRE= trains without them)
# A herd_reid_feat_*.tgz in WORK (the second export of a run) brings every set's frame vectors: nothing
# of those sets is fetched again.
# A herd_barn_feat_*.tgz in WORK (herd.py barn export: the barn's own cows, named by hand) adds the set
# "barn" to SOURCES, a herd_barn_ids_*.tgz (herd.py boxes ids: cows' boxes named on the box page) the set
# "barn_ids" (BARN=0: neither); their held-out cows are scored with the others and set the thresholds.
# Every step resumes. Licences: MmCows CC BY-NC-SA, CoBRA and SideViewCows CC BY 4.0, the Bristol sets NCGL
# (non-commercial), CowDataset as on Figshare 16879780.

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
    for v in WORK RUN DET SOURCES EPOCHS MCT MINUTES MODELS INIT UWB_DAYS UWB_EVERY LABEL_EVERY WAIT_MIN TRAIN REQUIRE BARN; do
        [ -n "${!v+x}" ] && knobs+="$v=$(printf '%q' "${!v}") "            # only those set: unset stays unset
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_REID_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/reid_pod.sh"); echo; echo '[reid finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$R"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "!! $(basename "$0") stopped at line $LINENO: $BASH_COMMAND (exit $?)"' ERR   # never silently
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
PY="$H/.venv/bin/python"
export HF_HOME="$WORK/hf-cache"
export PATH="$HOME/.local/bin:$PATH"
SOURCES="${SOURCES:-mmcows cobra sideview multicam cows2021 opencows aerialcattle cowdataset}"

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

step "The identity model: trained from scratch"
# From RUN's temporal transformer on every set at once (not the earlier identity model fine-tuned: that
# one has already fitted the top-down sets, and a second pass on top of it over-fits). INIT=<identity.pt>
# fine-tunes one instead.
if [ -n "${INIT:-}" ]; then
    [ -f "$INIT" ] || { echo "!! INIT=$INIT: no such file"; exit 1; }
    echo "fine-tuning $INIT (INIT given)"
    OUT="$R/id_model_ft"
    initarg=(--init "$INIT")
else
    echo "from $H/$RUN/model.pt's transformer, on all of: $SOURCES"
    OUT="$R/id_model_all"
    initarg=()
fi
if [ -z "${DET:-}" ]; then
    DET="$(find "$H" -maxdepth 3 -name det_train_meta.json -path '*/best/*' -printf '%T@ %h\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
fi

# an earlier run's frame vectors (herd_reid_feat_*.tgz, this script's second export): the sets in it are
# neither fetched nor labelled nor extracted again - a pod trains at once, and a set whose server is
# down (data.bris, 404 in 2026-10) still comes in
a="$(ls -t "$WORK"/herd_reid_feat_*.tgz 2>/dev/null | head -1 || true)"   # none: ls fails, pipefail would end the script
if [ -n "$a" ] && [ ! -f "$R/feat/.from_archive" ]; then
    step "Frame vectors from $a"
    mkdir -p "$R"
    tar -xzf "$a" -C "$WORK"
    touch "$R/feat/.from_archive"
    ls "$R/feat"
fi
have() { [ -f "$R/feat/$1/bursts.jsonl" ] && [ -f "$R/feat/$1/crops.npz" ] && [ -f "$R/feat/.from_archive" ]; }
# the barn's own cows (herd.py barn export): the newest archive replaces an older one's set
b="$(ls -t "$WORK"/herd_barn_feat_*.tgz 2>/dev/null | head -1 || true)"
if [ -n "$b" ] && [ "$(cat "$R/feat/barn/.from" 2>/dev/null || true)" != "$(basename "$b")" ]; then
    step "The barn's own cows from $b"
    rm -rf "$R/feat/barn"
    mkdir -p "$R"
    tar -xzf "$b" -C "$WORK"
    basename "$b" > "$R/feat/barn/.from"
fi
# the barn's frames with each cow's box named (herd.py boxes ids): crops, made into vectors below
bi="$(ls -t "$WORK"/herd_barn_ids_*.tgz 2>/dev/null | head -1 || true)"
if [ -n "$bi" ] && [ "$(cat "$R/raw/barn_ids/.from" 2>/dev/null || true)" != "$(basename "$bi")" ]; then
    step "The barn's named boxes from $bi"
    rm -rf "$R/raw/barn_ids" "$R/feat/barn_ids"
    mkdir -p "$R"
    tar -xzf "$bi" -C "$WORK"
    basename "$bi" > "$R/raw/barn_ids/.from"
fi
if [ "${BARN:-1}" = "1" ] && [ -f "$R/raw/barn_ids/index.jsonl" ] && [[ " $SOURCES " != *" barn_ids "* ]]; then
    SOURCES="$SOURCES barn_ids"
    echo "barn_ids: $(grep -c . "$R/raw/barn_ids/index.jsonl") named boxes of the barn's cows"
fi
if [ "${BARN:-1}" = "1" ] && [ -f "$R/feat/barn/bursts.jsonl" ] && [[ " $SOURCES " != *" barn "* ]]; then
    SOURCES="$SOURCES barn"
    echo "barn: $(grep -c . "$R/feat/barn/bursts.jsonl") bursts of the barn's own cows ($(grep -c '"split": "val"' "$R/feat/barn/bursts.jsonl" || true) of them held out)"
fi

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
if [ -n "${DET:-}" ] && grep -qs '"kind": *"rfdetr"' "$DET/det_train_meta.json"; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The downloads overlap the GPU: Zenodo's zips (whole, ~26 GB) and WildlifeReID-10k's crops come in the
# background, while MmCows is fetched a day at a time and each day is labelled (GPU) as the next downloads.
want() { [[ " $SOURCES " == *" $1 "* ]]; }
bg_fetch() {   # name, sources... -> a background fetch with its own log; its pid in BG_<name>
    local name="$1"; shift
    local srcs=()
    for s_ in "$@"; do
        if have "$s_"; then echo "$s_: frame vectors from the archive, not fetched"; continue; fi
        want "$s_" && srcs+=("$s_")
    done
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
bg_fetch wildlife multicam cows2021 opencows aerialcattle cowdataset
MCT_PID=""
if [ "${MCT:-1}" = "1" ]; then                      # the last step's frames (mct_pod.sh fetches only what is missing)
    echo "fetching MmCows' frames for the check in the background (log: $R/fetch_mct.log)"
    "$PY" "$HERE/mct_eval.py" fetch --data "$WORK/mct" --minutes "${MINUTES:-60}" > "$R/fetch_mct.log" 2>&1 &
    MCT_PID=$!
fi

# MmCows' vectors from an archive count only when made at this LABEL_EVERY (an archive without
# mmcows/fetch.json is from the 15 s runs)
mm_every() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("label_every", 15))' \
                 "$R/mmcows/fetch.json" 2>/dev/null || echo 15; }
if have mmcows && [ "$(mm_every)" != "${LABEL_EVERY:-60}" ]; then
    echo "mmcows: the archive's vectors are of 7/25 every $(mm_every) s, LABEL_EVERY is ${LABEL_EVERY:-60}: made again"
    rm -f "$R/feat/mmcows/bursts.jsonl"
fi
if have mmcows; then
    echo "mmcows: frame vectors from the archive, not fetched / labelled / extracted"
elif want mmcows; then
    [ -n "${DET:-}" ] && [ -f "$DET/det_train_meta.json" ] || { echo "!! no detector best/ under $H (DET=<it>)"; exit 1; }
    echo "detector $DET"
    # no neck-tag days unless UWB_DAYS names them: their labels are ~4% wrong, mostly an untagged cow
    # (11-16, the held-out ones) lying next to a tagged one - a held-out cow would get into training
    DAYS=(${UWB_DAYS-})
    mm_fetch() { "$PY" "$HERE/reid.py" fetch --data "$R" --sources mmcows --mm-uwb-days "$@" \
                     --mm-uwb-every "${UWB_EVERY:-30}" --mm-label-every "${LABEL_EVERY:-60}"; }
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

bg_wait wildlife multicam cows2021 opencows aerialcattle cowdataset
bg_wait zenodo cobra sideview
if want barn_ids; then
    step "Extract the barn's named boxes: DINOv2 on every crop (GPU)"
    "$PY" "$HERE/reid.py" extract --data "$R" --sources barn_ids --model "$H/$RUN/model.pt"
fi

# without the sets most cows come from the model learns too few cows and over-fits MmCows' ten
# (2026-10-19, data.bris down: 142 cows instead of 312, cows 11-16 by look 51% -> 32%): stop instead
missing=()
for s_ in ${REQUIRE-multicam cows2021}; do
    want "$s_" && [ ! -f "$R/feat/$s_/bursts.jsonl" ] && missing+=("$s_")
done
if [ ${#missing[@]} -gt 0 ] && [ "${TRAIN:-1}" != "0" ]; then
    echo "!! ${missing[*]} not in (their fetch failed - see $R/fetch_wildlife.log): no training without them."
    echo "   Run again later (what is in is kept), upload a herd_reid_feat_*.tgz that has them, or"
    echo "   REQUIRE= to train without them anyway."
    exit 1
fi

if [ "${TRAIN:-1}" = "0" ] && [ -f "$OUT/identity.pt" ]; then
    step "No training (TRAIN=0): the thresholds of $OUT/identity.pt again"
    "$PY" "$HERE/reid.py" calibrate --data "$R" --identity "$OUT/identity.pt"
else
    step "Train the identity model ${initarg[*]:-(fresh)}"
    "$PY" "$HERE/reid.py" train --data "$R" --sources $SOURCES --model "$H/$RUN/model.pt" --epochs "${EPOCHS:-30}" \
        --out "$OUT" "${initarg[@]}"
fi

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
# the frame vectors of every set (and MmCows' crop index the frame-vs-frame loss reads): a next pod
# trains from them without fetching anything (~1-2 GB)
F="$WORK/export/herd_reid_feat_$(date +%Y%m%d_%H%M).tgz"
fparts=()
for p in "$R"/feat/*/crops.npz "$R"/feat/*/crops.npz.json "$R"/feat/*/bursts.jsonl "$R"/feat/*/meta.json \
         "$R/raw/mmcows/index.jsonl" "$R/mmcows/fetch.json"; do
    [ -e "$p" ] && fparts+=("${p#"$WORK"/}")
done
[ ${#fparts[@]} -gt 0 ] && (cd "$WORK" && tar -czf "$F" "${fparts[@]}") && \
    echo "frame vectors for the next pod: $F   (keep it: with it a pod fetches no training set)"
echo "in the barn: [model] identity = \".../reid/$(basename "$OUT")/identity.pt\" (abstain.json next to it is used)"
