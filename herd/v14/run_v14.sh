#!/usr/bin/env bash
# Version 14 (A): cow identification only, trained from scratch.
#   ID_DATA=mmcows (default): one barn - the ID transformer trains on 8 of MmCows' 16 cows and is
#                            tested on 5 others it never saw (3 more pick the epoch)
#   ID_DATA=mcc:             it trains on MultiCamCows2024 (90 cows, data.bris.ac.uk) instead
#
#   bash herd/v14/run_v14.sh          start in tmux session "v14" (hours: downloads + training)
#   bash herd/v14/run_v14.sh log      follow the log
#   bash herd/v14/run_v14.sh stop
#
# Steps (each resumes):
#   1 Python environment                     5 ID transformer: train + test day (MultiCamCows)
#   2 version 10's detector + DINOv2          6 association transformer: train + test window (MmCows)
#   3 MmCows labels, frames, 1-s stream       7 onboarding test on MmCows: association vs camera-alone
#   4 MultiCamCows2024 (ID_DATA=mcc only)     8 RESULTS.md, RUN14A_RESULTS.tar, V14A_MODELS.tgz
#
# Knobs: WORK=/workspace  MODELS_URL (version 10's archive)  DET_NAME (its detector folder)
#        LIVE_STEP=2 (s between 1-second frames used)  TRAIN_EVERY=60  WINDOW_EVERY=15
#        SETUP_MIN=60  START=12  END=15  ID_EPOCHS=60  ASSOC_EPOCHS=60
#        MCC_WAIT_H=12 (hours to wait for data.bris.ac.uk)  MCC_ZIP (where the MultiCamCows zip is / goes)
#        PRETRAINED=<link to a V14A_MODELS.tgz> (a Google Drive share link works): no training - the
#        weights are downloaded after the MmCows step and video_v14.sh runs on them (no steps 4-8)
#        ID_DATA=wr10k: v14 (D) - the ID transformer trains on WildlifeReID-10k's cattle (Kaggle) and is
#        tested on all of MmCows' cows; ASSOC_PRETRAINED=<link to a *_MODELS.tgz>: its assoc.pt is used
#        (scored on the test window) instead of training one; NAME=<V14D ..>: after step 6, video_v14.sh
#        makes <NAME>_RESULTS.tar, <NAME>_MODELS.tgz, <NAME>_VIDEO.tar instead of steps 7-8
#        ID_DATA=ocid: v14 (F) - OpenCowID's model (ResNet-50 + 128 numbers, centroid metric learning) as the ID
#        model, trained on its synthetic coats then on WildlifeReID-10k's cattle (ID_STAGES=synth,wr10k; the
#        pictures, not DINOv2 vectors), tested on all of MmCows' cows; the rest as v14 (D)
#        SYS_START=<hour> SETUP_MIN=<min> SYS_END=<hour> (with NAME): the onboarding test runs SYS_START ->
#        SYS_END with SETUP_MIN of onboarding (default START, 60, END); 1-second frames outside 12:00-15:00
#        come from MmCows' whole-day zips (only the frames used are fetched, ~1.3 GB per camera-hour)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERD="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
V="$WORK/v14"
VENV="$WORK/herd/.venv"
PY="$VENV/bin/python"
LOG="$V/log.txt"
SESSION=v14
MODELS_URL="${MODELS_URL:-https://drive.usercontent.google.com/download?id=1urVuSsmAapKD8O1SZ-b9CR5ycOZePRe3&export=download&confirm=t}"
DET_NAME="${DET_NAME:-detector_rfdetr_large_zenodo_mmcows_opencows_ft}"
LIVE_STEP="${LIVE_STEP:-2}"
TRAIN_EVERY="${TRAIN_EVERY:-60}"
WINDOW_EVERY="${WINDOW_EVERY:-15}"
SETUP_MIN="${SETUP_MIN:-60}"
START="${START:-12}"
END="${END:-15}"
ID_EPOCHS="${ID_EPOCHS:-60}"
ASSOC_EPOCHS="${ASSOC_EPOCHS:-60}"
MCC_WAIT_H="${MCC_WAIT_H:-12}"
ID_DATA="${ID_DATA:-mmcows}"
MCC_ZIP="${MCC_ZIP:-}"
MCC_URL="${MCC_URL:-https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip}"
PRETRAINED="${PRETRAINED:-}"
ASSOC_PRETRAINED="${ASSOC_PRETRAINED:-}"
ID_STAGES="${ID_STAGES:-synth,wr10k}"
SYS_START="${SYS_START:-$START}"
SYS_END="${SYS_END:-$END}"
VIDEO_END="${VIDEO_END:-13.5}"
MODE="${MODE:-}"; CENTER="${CENTER:-}"; NAME="${NAME:-}"; REALTIME="${REALTIME:-}"; CLIPS="${CLIPS:-}"; FPS="${FPS:-}"; TITLE="${TITLE:-}"
export HF_HOME="$WORK/hf-cache"

case "${1:-}" in
    log)  [ -f "$LOG" ] || { echo "no log yet: $LOG"; exit 1; }; exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"
          # the run ignores hang-ups (to survive a closed terminal): end it and what it started for real
          for p in $(pgrep -f '^bash .*run_v14\.sh' || true); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
          pkill -f 'python.* (data_mcc|data_mmcows|data_wr10k|data_images|idnet|opencowid|assoc|system|system_e|summarize)\.py' 2>/dev/null || true
          pkill -f 'curl .*MultiCamCows2024\.zip' 2>/dev/null || true
          exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${V14_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    mkdir -p "$V"
    knobs=""
    for v in WORK MODELS_URL DET_NAME LIVE_STEP TRAIN_EVERY WINDOW_EVERY SETUP_MIN START END ID_EPOCHS ASSOC_EPOCHS MCC_WAIT_H MCC_ZIP ID_DATA ID_STAGES SYS_START SYS_END CONTROL_START CONTROL_MIN PRETRAINED ASSOC_PRETRAINED VIDEO_END CENTER NAME REALTIME CLIPS FPS TITLE MODE TAG; do
        knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env V14_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/run_v14.sh"); echo; echo '[run_v14.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log"
    exit 0
fi

mkdir -p "$V"
trap '' HUP
exec > >(tee -a "$LOG") 2>&1
set -E
trap 'rc=$?; echo "!! line $LINENO failed (exit $rc): $BASH_COMMAND"' ERR
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }

step "1/8  Python environment ($VENV)"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
ready() { [ -x "$PY" ] && "$PY" -c "import torch, torchvision, transformers, cv2, numpy, scipy, PIL; assert torch.cuda.is_available()" 2>/dev/null; }
if ! ready; then
    [ -x "$PY" ] || rm -rf "$VENV"
    [ -d "$VENV" ] || uv venv --python 3.12 --seed --managed-python "$VENV"
    uv pip install --python "$PY" torch torchvision --torch-backend=auto
    uv pip install --python "$PY" "transformers>=5.15" opencv-python-headless numpy scipy pillow requests safetensors
    ready || { echo "!! the venv cannot import torch with CUDA / transformers / cv2"; exit 1; }
fi
"$PY" -c "import rfdetr, pytorch_lightning" 2>/dev/null || uv pip install --python "$PY" "rfdetr[train]"
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
cd "$HERE"

step "2/8  Version 10's detector and DINOv2"
DET="$WORK/herd/$DET_NAME/best"
if [ ! -f "$DET/checkpoint.pth" ]; then
    mkdir -p "$WORK/v10"
    A="$WORK/v10/models.tgz"
    [ -s "$A" ] || curl -L --fail --retry 5 -C - -o "$A" "$MODELS_URL"
    head -c 512 "$A" | grep -qi '<html' && { rm -f "$A"; echo "!! the models link gives a web page: share it as 'Anyone with the link'"; exit 1; }
    tar -xzf "$A" -C "$WORK" "herd/$DET_NAME" "hf-cache" 2>/dev/null || tar -xzf "$A" -C "$WORK"
fi
[ -f "$DET/det_train_meta.json" ] || { echo "!! no $DET - set DET_NAME to a detector folder in the archive"; ls "$WORK/herd"; exit 1; }
echo "detector: $DET"

step "3/8  MmCows: labels, labelled frames (training side + test window), the 1-second stream"
again() {   # a download step, tried up to 4 times (Hugging Face sometimes cuts a connection)
    for i in 1 2 3 4; do "$@" && return 0; echo "   stopped (try $i of 4) - again in 30 s"; sleep 30; done; return 1; }
again "$PY" data_mmcows.py labels --out "$V/mmcows"
again "$PY" data_mmcows.py frames --out "$V/mmcows" --train-every "$TRAIN_EVERY" --window-every "$WINDOW_EVERY" --start "$START" --end "$END"
"$PY" data_mmcows.py inspect --out "$V/mmcows" | tee "$V/mmcows_live_inspect.txt"
fetch_models() {   # $1 a link to a *_MODELS.tgz -> extracted in $V/pretrained; prints nothing on success
    local id url a="$V/pretrained.tgz"
    id="$(printf '%s' "$1" | sed -nE 's#.*(/d/|[?&]id=)([A-Za-z0-9_-]{20,}).*#\2#p')"
    url="$1"; [ -n "$id" ] && url="https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
    [ -s "$a" ] || curl -L --fail --retry 5 -o "$a" "$url"
    head -c 512 "$a" | grep -qi '<html' && { rm -f "$a"; echo "!! the link gives a web page: share it as 'Anyone with the link'"; exit 1; }
    rm -rf "$V/pretrained" && mkdir -p "$V/pretrained" && tar -xzf "$a" -C "$V/pretrained"
}
if [ -n "$PRETRAINED" ]; then
    # only what the video needs: the 1-second stream up to VIDEO_END, then the given weights
    again "$PY" data_mmcows.py live --out "$V/mmcows" --live-step "$LIVE_STEP" --start "$START" --end "$VIDEO_END"
    step "4/4  Trained weights instead of training: $PRETRAINED"
    fetch_models "$PRETRAINED"
    mkdir -p "$V/idnet" "$V/assoc"
    IDP="$(find "$V/pretrained" -name idnet.pt | head -n 1)"; ASP="$(find "$V/pretrained" -name assoc.pt | head -n 1)"
    [ -n "$IDP" ] || { echo "!! no idnet.pt in the archive:"; find "$V/pretrained"; exit 1; }
    cp "$IDP" "$V/idnet/idnet.pt"
    [ -n "$ASP" ] && cp "$ASP" "$V/assoc/assoc.pt"
    "$PY" -c "import torch; ck = torch.load('$V/idnet/idnet.pt', weights_only=False); print('idnet: epoch', ck.get('epoch'), 'split', ck.get('split', {}).get('by'), 'test cows', ck.get('split', {}).get('test'))"
    echo "weights in place - now the video (log: $V/video_log.txt)"
    V14V_IN_TMUX=1 WORK="$WORK" DET_NAME="$DET_NAME" SYS_END="$VIDEO_END" exec bash "$HERE/video_v14.sh"
fi
again "$PY" data_mmcows.py live --out "$V/mmcows" --live-step "$LIVE_STEP" --start "$START" --end "$END"
if [ "$SYS_START" != "$START" ] || [ "$SYS_END" != "$END" ]; then
    echo "   the onboarding test runs $SYS_START:00 -> $SYS_END:00: its 1-second frames too"
    again "$PY" data_mmcows.py live --out "$V/mmcows" --live-step "$LIVE_STEP" --start "$SYS_START" --end "$SYS_END"
fi
"$PY" data_mmcows.py feats --out "$V/mmcows"
"$PY" data_mmcows.py idcrops --out "$V/mmcows" --train-every "$TRAIN_EVERY" --start "$START" --end "$END"

step "4/8  MultiCamCows2024: the whole zip, then crops -> DINOv2 (ID_DATA=$ID_DATA)"
if [ "$ID_DATA" = mcc ]; then
# The whole zip as one file (36.5 GB, resumes where a cut transfer stopped), then read from disk.
# data.bris.ac.uk's file server is sometimes down for hours: keep trying for MCC_WAIT_H hours.
Z="${MCC_ZIP:-$V/MultiCamCows2024.zip}"
if [ ! -f "$Z" ]; then
    for i in $(seq 0 $((MCC_WAIT_H * 6))); do
        if curl -L -# --fail --retry 3 -C - -o "$Z.part" "$MCC_URL"; then mv "$Z.part" "$Z"; break; fi
        [ "$i" -ge $((MCC_WAIT_H * 6)) ] && { echo "!! data.bris.ac.uk did not send the zip for $MCC_WAIT_H h"; exit 1; }
        echo "   data.bris.ac.uk not answering - try again in 10 min ($(date '+%T'), $( { du -h "$Z.part" 2>/dev/null || echo "0"; } | cut -f1) so far)"; sleep 600
    done
fi
SRC=(--zip "$Z")
"$PY" data_mcc.py inspect "${SRC[@]}" | tee "$V/mcc_inspect.txt"
for j in 1 2 3 4 5 6; do
    "$PY" data_mcc.py fetch --out "$V/mcc" "${SRC[@]}" && break
    [ "$j" = 6 ] && { echo "!! the MultiCamCows fetch failed 6 times"; exit 1; }
    echo "   fetch stopped - resume in 10 min"; sleep 600
done
elif [ "$ID_DATA" = wr10k ]; then
    step "4/8  WildlifeReID-10k's cattle (Kaggle, read in place) -> DINOv2; + MmCows' boxes as the test"
    again "$PY" data_wr10k.py fetch --out "$V/wr10k"
    "$PY" data_wr10k.py combine --wr "$V/wr10k" --mm "$V/mmcows/idcrops" --out "$V/idd"
elif [ "$ID_DATA" = ocid ]; then
    step "4/8  v14 (F): the pictures - WildlifeReID-10k's cattle (Kaggle, read in place) + MmCows' boxes as the test"
    again "$PY" data_images.py wr10k --out "$V/wr10k_img"
    "$PY" data_images.py mmcows --mm "$V/mmcows" --out "$V/mmcows_img" --start "$START" --end "$END" --train-every "$TRAIN_EVERY"
    "$PY" data_images.py combine --wr "$V/wr10k_img" --mm "$V/mmcows_img" --out "$V/iddi"
    "$PY" opencowid.py coats --out "$V/coats.png"
else echo "skipped: the ID transformer trains on MmCows cows (ID_DATA=mmcows)"; fi

IDN="$V/idnet"
ID_CMD=(idnet.py train)
if [ "$ID_DATA" = mcc ]; then
    step "5/8  ID transformer (from scratch): MultiCamCows days 1-5 -> day 6 picks -> day 7 tests"
    ID_ARGS=(--feats "$V/mcc" --split days)
elif [ "$ID_DATA" = wr10k ]; then
    step "5/8  ID transformer (from scratch): WildlifeReID-10k cows train, held-out ones pick, MmCows' 16 test it"
    IDN="$V/idnet_wr10k"
    ID_ARGS=(--feats "$V/idd" --split groups --time-aug 0.25 --time-aug-max 15 --P 32)
elif [ "$ID_DATA" = ocid ]; then
    step "5/8  OpenCowID's model (ResNet-50, ImageNet start): stages $ID_STAGES; held-out WildlifeReID-10k cows pick the epoch, MmCows' 16 test it"
    IDN="$V/idnet_ocid"
    ID_CMD=(opencowid.py train)
    ID_ARGS=(--images "$V/iddi" --stages "$ID_STAGES")
else
    step "5/8  ID transformer (from scratch): MmCows, 8 cows train, 3 pick the epoch, 5 never seen test it"
    ID_ARGS=(--feats "$V/mmcows/idcrops" --split cows --val-cows 3 --test-cows 5 --time-aug 0.02 --P 8)
fi
if [ -f "$IDN/idnet.pt" ] && [ -f "$IDN/eval_id.json" ]; then echo "trained already"; else
    "$PY" "${ID_CMD[@]}" "${ID_ARGS[@]}" --out "$IDN" --epochs "$ID_EPOCHS"; fi

if [ "$MODE" = e ]; then
    step "6/8  Association transformer: not used by version 14 (E)"
elif [ -n "$ASSOC_PRETRAINED" ]; then
    step "6/8  Association transformer: the trained one from $ASSOC_PRETRAINED, scored on the test window"
    if [ ! -f "$V/assoc/eval_assoc.json" ] || [ ! -f "$V/assoc/.from_pretrained" ]; then
        fetch_models "$ASSOC_PRETRAINED"
        ASP="$(find "$V/pretrained" -name assoc.pt | head -n 1)"
        [ -n "$ASP" ] || { echo "!! no assoc.pt in the archive:"; find "$V/pretrained"; exit 1; }
        mkdir -p "$V/assoc" && cp "$ASP" "$V/assoc/assoc.pt" && touch "$V/assoc/.from_pretrained"
        "$PY" assoc.py eval --data "$V/mmcows" --out "$V/assoc"
    else echo "in place already"; fi
else
step "6/8  Association transformer (from scratch): MmCows outside the test window -> the window"
if [ -f "$V/assoc/assoc.pt" ] && [ -f "$V/assoc/eval_assoc.json" ]; then echo "trained already"; else
    "$PY" assoc.py train --data "$V/mmcows" --out "$V/assoc" --epochs "$ASSOC_EPOCHS"; fi
fi

if [ -n "$NAME" ]; then
    echo "models ready - now $NAME's onboarding test, videos, results and weights (log: $V/video_log.txt)"
    V14V_IN_TMUX=1 WORK="$WORK" DET_NAME="$DET_NAME" SYS_START="$SYS_START" SETUP_MIN="$SETUP_MIN" SYS_END="$SYS_END" \
        CONTROL_START="${CONTROL_START:-}" CONTROL_MIN="${CONTROL_MIN:-}" IDNET="$IDN/idnet.pt" MODE="$MODE" \
        exec bash "$HERE/video_v14.sh"
fi

step "7/8  Onboarding test on MmCows (inference only): setup $SETUP_MIN min, then identification"
"$PY" system.py --data "$V/mmcows" --detector "$DET" --idnet "$V/idnet/idnet.pt" --assoc "$V/assoc/assoc.pt" \
    --out "$V/system" --setup-min "$SETUP_MIN" --start "$START" --end "$END" --step "$LIVE_STEP"
"$PY" system.py --data "$V/mmcows" --detector "$DET" --idnet "$V/idnet/idnet.pt" --assoc none \
    --out "$V/system" --setup-min "$SETUP_MIN" --start "$START" --end "$END" --step "$LIVE_STEP"

step "8/8  Results and models"
R="$WORK/RUN14A_RESULTS"
rm -rf "$R"; mkdir -p "$R"
"$PY" summarize.py --work "$V" --out "$R"
for f in idnet/eval_id.json idnet/history.json assoc/eval_assoc.json assoc/history.json \
         system/system_assoc.json system/system_iou_per_camera.json mcc_inspect.txt mmcows_live_inspect.txt \
         mcc/meta.json mcc/samples.png mmcows/assoc_meta.json mmcows/idcrops/meta.json log.txt; do
    [ -f "$V/$f" ] && { mkdir -p "$R/$(dirname "$f")"; cp "$V/$f" "$R/$f"; }
done
cp "$DET/det_train_meta.json" "$R/detector_det_train_meta.json"
tar -cf "$WORK/RUN14A_RESULTS.tar" -C "$WORK" RUN14A_RESULTS
M="$WORK/V14A_MODELS"
rm -rf "$M"; mkdir -p "$M"
cp "$V/idnet/idnet.pt" "$V/assoc/assoc.pt" "$M/"
cat > "$M/HOW_TO_USE.txt" <<EOF
v14 (A) models, $(date '+%F %T')
idnet.pt   ID transformer: crop sequence -> fingerprint (+ 'not sure' model, enrolment threshold)
assoc.pt   association transformer: detections of all cameras -> one cow
detector   version 10's: herd/$DET_NAME/best (not retrained; in the version 10 archive)
DINOv2-S   facebook/dinov2-small, frozen (hf-cache in the version 10 archive)
Run:       python herd/v14/system.py --data <mmcows> --detector <best/> --idnet idnet.pt --assoc assoc.pt --out <dir>
EOF
tar -czf "$WORK/V14A_MODELS.tgz" -C "$WORK" V14A_MODELS
echo
echo "Done:"
echo "  $WORK/RUN14A_RESULTS.tar   ($(du -h "$WORK/RUN14A_RESULTS.tar" | cut -f1))  RESULTS.md + every score"
echo "  $WORK/V14A_MODELS.tgz      ($(du -h "$WORK/V14A_MODELS.tgz" | cut -f1))  idnet.pt, assoc.pt"
