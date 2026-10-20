#!/usr/bin/env bash
# Version 14 (A) on video: the onboarding test again (inference only, the trained models of
# run_v14.sh), this time keeping what it said, then videos of 4 cameras side by side.
#
#   bash herd/v14/video_v14.sh          start in tmux session "v14video"
#   bash herd/v14/video_v14.sh log      follow the log
#   bash herd/v14/video_v14.sh stop
#
# Needs run_v14.sh's steps 1-6 done (/workspace/v14: mmcows/, idnet/idnet.pt, assoc/assoc.pt).
# Knobs: WORK=/workspace  TAG=assoc (or iou_per_camera: each camera alone)  SYS_START=12  SETUP_MIN=60
#        SYS_END=13.5 (the system runs SYS_START -> SYS_END: SETUP_MIN of onboarding, then identification;
#        frames outside 12:00-15:00 come from MmCows' whole-day zips, see run_v14.sh)  FPS=5 (10x speed)
#        CLIPS="12:00-12:10 13:00-13:10 13:10-13:30" (HH:MM or HH:MM:SS)  REALTIME=0 (1: barn speed)
#        CENTER=off (on: v14 C - the barn's average cow taken out of the fingerprints)  NAME=V14A
#        MODE=e: v14 E (system_e.py: no tracking model, every box named on its own); TAG then picks the
#        album the video shows: e_built, e_confirmed (default) or e_oracle
# -> /workspace/<NAME>_VIDEO.tar (the clips), <NAME>_RESULTS.tar (RESULTS.md + scores), <NAME>_MODELS.tgz

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
V="$WORK/v14"
PY="$WORK/herd/.venv/bin/python"
LOG="$V/video_log.txt"
SESSION=v14video
TAG="${TAG:-assoc}"
SYS_END="${SYS_END:-13.5}"
SYS_START="${SYS_START:-12}"
SETUP_MIN="${SETUP_MIN:-60}"
FPS="${FPS:-5}"
CLIPS="${CLIPS:-12:00-12:10 13:00-13:10 13:10-13:30}"
DET_NAME="${DET_NAME:-detector_rfdetr_large_zenodo_mmcows_opencows_ft}"
CENTER="${CENTER:-off}"
NAME="${NAME:-V14A}"
REALTIME="${REALTIME:-0}"
TITLE="${TITLE:-}"
IDNET="${IDNET:-$V/idnet/idnet.pt}"
MODE="${MODE:-}"
[ "$MODE" = e ] && [ "$TAG" = assoc ] && TAG=e_confirmed
export HF_HOME="$WORK/hf-cache"

case "${1:-}" in
    log)  [ -f "$LOG" ] || { echo "no log yet: $LOG"; exit 1; }; exec tail -n 60 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"
          for p in $(pgrep -f '^bash .*video_v14\.sh' || true); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
          pkill -f 'python.* (system|video)\.py' 2>/dev/null || true
          exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${V14V_IN_TMUX:-}" ]; then
    for f in "$PY" "$IDNET" "$V/mmcows/live.jsonl"; do
        [ -e "$f" ] || { echo "missing $f - run herd/v14/run_v14.sh first"; exit 1; }; done
    [ "$TAG" = iou_per_camera ] || [ "$MODE" = e ] || [ -f "$V/assoc/assoc.pt" ] || { echo "missing $V/assoc/assoc.pt"; exit 1; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    knobs=""
    for v in WORK TAG SYS_START SETUP_MIN SYS_END FPS CLIPS DET_NAME CENTER NAME REALTIME TITLE IDNET MODE; do knobs+="$v=$(printf '%q' "${!v}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env V14V_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/video_v14.sh"); echo; echo '[video_v14.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log"
    exit 0
fi

trap '' HUP
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
cd "$HERE"

step "1/3  video packages"
"$PY" -c "import imageio, imageio_ffmpeg" 2>/dev/null || uv pip install --python "$PY" imageio imageio-ffmpeg \
    || echo "   (no imageio-ffmpeg: videos in mp4v, play them in VLC)"

step "2/3  onboarding test again, keeping what it said ($SYS_START h + $SETUP_MIN min onboarding -> $SYS_END h, tracking: $TAG, centering: $CENTER)"
S="$V/system_video"; [ "$NAME" = V14A ] || S="$V/system_$NAME"
if [ -f "$S/names_$TAG.json" ]; then echo "done already: $S/names_$TAG.json"
elif [ "$MODE" = e ]; then
    "$PY" system_e.py --data "$V/mmcows" --detector "$WORK/herd/$DET_NAME/best" --idnet "$IDNET" \
        --out "$S" --start "$SYS_START" --end "$SYS_END" --setup-min "$SETUP_MIN" --step 2
else
    ASSOC="$V/assoc/assoc.pt"; [ "$TAG" = iou_per_camera ] && ASSOC=none
    "$PY" system.py --data "$V/mmcows" --detector "$WORK/herd/$DET_NAME/best" --idnet "$IDNET" \
        --assoc "$ASSOC" --out "$S" --start "$SYS_START" --end "$SYS_END" --setup-min "$SETUP_MIN" --step 2 --center "$CENTER"
fi

step "3/3  videos, results and weights: $CLIPS"
O="$WORK/${NAME}_VIDEO"
mkdir -p "$O"
lower="$(printf '%s' "$NAME" | tr 'A-Z' 'a-z')"
RT=(); [ "$REALTIME" = 1 ] && RT=(--realtime)
for c in $CLIPS; do
    a="${c%-*}"; b="${c#*-}"
    "$PY" video.py --data "$V/mmcows" --system "$S" --tag "$TAG" --from "$a" --to "$b" --fps "$FPS" "${RT[@]}" \
        --title "$TITLE" --out "$O/${lower}_${TAG}_${a//:/}-${b//:/}.mp4"
done
SYSJ="$S/system_$TAG.json"; [ "$MODE" = e ] && SYSJ="$S/system_e.json"
cp "$SYSJ" "$O/" 2>/dev/null || true
# tar's exit 1 = "a file changed as we read it" (the last clip still being flushed): wait, pack again
pack() { sync; sleep 3; tar "$@" || { r=$?; [ "$r" = 1 ] && { sleep 10; sync; tar "$@"; } || return "$r"; }; }
pack -cf "$WORK/${NAME}_VIDEO.tar" -C "$WORK" "${NAME}_VIDEO"

# results: RESULTS.md (with the other way, A or B, scored on the same run) + the raw scores
R="$WORK/${NAME}_RESULTS"
rm -rf "$R"; mkdir -p "$R/work/system"
cp "$SYSJ" "$R/work/system/$(basename "$SYSJ")"
[ "$MODE" = e ] && cp "$S/RESULTS_E.md" "$R/RESULTS.md"
ID_DIR="$(dirname "$IDNET")"
for f in eval_id.json history.json; do [ -f "$ID_DIR/$f" ] && mkdir -p "$R/work/idnet" && cp "$ID_DIR/$f" "$R/work/idnet/"; done
for f in eval_assoc.json history.json; do [ -f "$V/assoc/$f" ] && mkdir -p "$R/work/assoc" && cp "$V/assoc/$f" "$R/work/assoc/"; done
VER="${NAME#V14}"; [ "$VER" = A ] && VER=""
[ "$MODE" = e ] || "$PY" summarize.py --work "$R/work" --out "$R" ${VER:+--version "$VER"} > /dev/null
for d in system idnet assoc; do [ -d "$R/work/$d" ] && mv "$R/work/$d" "$R/$d"; done; rm -rf "$R/work"
cp "$S/names_$TAG.json" "$R/system/"
cp "$LOG" "$R/" 2>/dev/null || true
pack -cf "$WORK/${NAME}_RESULTS.tar" -C "$WORK" "${NAME}_RESULTS"

# weights: the same files - B changes only how the album is built at onboarding
M="$WORK/${NAME}_MODELS"
rm -rf "$M"; mkdir -p "$M"
cp "$IDNET" "$M/idnet.pt"; [ "$MODE" = e ] || { [ -f "$V/assoc/assoc.pt" ] && cp "$V/assoc/assoc.pt" "$M/"; }
{
    echo "$NAME models, $(date '+%F %T')"
    echo "idnet.pt   ID transformer: crop sequence -> fingerprint"
    echo "assoc.pt   association transformer: detections of all cameras -> one cow"
    echo "detector   version 10's: herd/$DET_NAME/best (in the version 10 archive)"
    if [ "$MODE" = e ]; then echo "Run:       python herd/v14/system_e.py --data <mmcows> --detector <best/> --idnet idnet.pt --out <dir>"; else
    echo "Run:       python herd/v14/system.py --data <mmcows> --detector <best/> --idnet idnet.pt --assoc assoc.pt --out <dir> --center $CENTER"; fi
    echo "           --center on = v14 C: the barn's average cow is taken out of every fingerprint (measured in the"
    echo "           setup hour) and the album threshold is calibrated there too, without labels"
} > "$M/HOW_TO_USE.txt"
pack -czf "$WORK/${NAME}_MODELS.tgz" -C "$WORK" "${NAME}_MODELS"

echo
echo "Done:"
for f in "$WORK/${NAME}_RESULTS.tar" "$WORK/${NAME}_MODELS.tgz" "$WORK/${NAME}_VIDEO.tar"; do
    echo "  $f  ($(du -h "$f" | cut -f1))"; done
ls -lh "$O"
