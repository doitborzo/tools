#!/usr/bin/env bash
# Version 14 (A) on video: the onboarding test again (inference only, the trained models of
# run_v14.sh), this time keeping what it said, then videos of 4 cameras side by side.
#
#   bash herd/v14/video_v14.sh          start in tmux session "v14video"
#   bash herd/v14/video_v14.sh log      follow the log
#   bash herd/v14/video_v14.sh stop
#
# Needs run_v14.sh's steps 1-6 done (/workspace/v14: mmcows/, idnet/idnet.pt, assoc/assoc.pt).
# Knobs: WORK=/workspace  TAG=assoc (or iou_per_camera: each camera alone)  SYS_END=13.5
#        (the system runs 12:00 -> SYS_END: setup hour + identification)  FPS=5 (10x speed)
#        CLIPS="12:00-12:10 13:00-13:10 13:10-13:30"
# -> /workspace/V14A_VIDEO/*.mp4 and /workspace/V14A_VIDEO.tar

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
V="$WORK/v14"
PY="$WORK/herd/.venv/bin/python"
LOG="$V/video_log.txt"
SESSION=v14video
TAG="${TAG:-assoc}"
SYS_END="${SYS_END:-13.5}"
FPS="${FPS:-5}"
CLIPS="${CLIPS:-12:00-12:10 13:00-13:10 13:10-13:30}"
DET_NAME="${DET_NAME:-detector_rfdetr_large_zenodo_mmcows_opencows_ft}"
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
    for f in "$PY" "$V/idnet/idnet.pt" "$V/mmcows/live.jsonl"; do
        [ -e "$f" ] || { echo "missing $f - run herd/v14/run_v14.sh first"; exit 1; }; done
    [ "$TAG" = iou_per_camera ] || [ -f "$V/assoc/assoc.pt" ] || { echo "missing $V/assoc/assoc.pt"; exit 1; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "already running: bash $0 log"; exit 1; fi
    knobs=""
    for v in WORK TAG SYS_END FPS CLIPS DET_NAME; do knobs+="$v=$(printf '%q' "${!v}") "; done
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

step "2/3  onboarding test again, keeping what it said (12:00 -> $SYS_END h, tracking: $TAG)"
S="$V/system_video"
if [ -f "$S/names_$TAG.json" ]; then echo "done already: $S/names_$TAG.json"; else
    ASSOC="$V/assoc/assoc.pt"; [ "$TAG" = iou_per_camera ] && ASSOC=none
    "$PY" system.py --data "$V/mmcows" --detector "$WORK/herd/$DET_NAME/best" --idnet "$V/idnet/idnet.pt" \
        --assoc "$ASSOC" --out "$S" --start 12 --end "$SYS_END" --setup-min 60 --step 2
fi

step "3/3  videos: $CLIPS"
O="$WORK/V14A_VIDEO"
mkdir -p "$O"
for c in $CLIPS; do
    a="${c%-*}"; b="${c#*-}"
    "$PY" video.py --data "$V/mmcows" --system "$S" --tag "$TAG" --from "$a" --to "$b" --fps "$FPS" \
        --out "$O/v14a_${TAG}_${a/:/}-${b/:/}.mp4"
done
cp "$S/system_$TAG.json" "$O/" 2>/dev/null || true
tar -cf "$WORK/V14A_VIDEO.tar" -C "$WORK" V14A_VIDEO
echo
echo "Done: $WORK/V14A_VIDEO.tar ($(du -h "$WORK/V14A_VIDEO.tar" | cut -f1))"
ls -lh "$O"
