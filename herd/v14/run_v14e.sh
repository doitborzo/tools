#!/usr/bin/env bash
# Version 14 (E): no tracking model - every box named on its own from version 14 (D)'s ID
# transformer, smoothed over a few seconds per camera; the album built with v14 C's rules and
# "never two chains seen together". Nothing trained here (the ID transformer is trained in step 5
# if this pod does not have it yet). Three albums scored on one run: built (automatic),
# confirmed (the farmer names every album entry at onboarding), oracle (perfect album).
#
#   bash herd/v14/run_v14e.sh          start (tmux session "v14")
#   bash herd/v14/run_v14e.sh log      follow the log (the system part: /workspace/v14/video_log.txt)
#   bash herd/v14/run_v14e.sh stop
#
# -> /workspace/V14E_RESULTS.tar, V14E_MODELS.tgz, V14E_VIDEO.tar (3 clips, the confirmed album)
# Knobs: TAG=e_confirmed (e_built, e_oracle: which album the video shows)  END=15  CLIPS  FPS=2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    log)  exec bash "$HERE/run_v14.sh" log ;;
    stop) bash "$HERE/video_v14.sh" stop; exec bash "$HERE/run_v14.sh" stop ;;
esac
unset PRETRAINED ASSOC_PRETRAINED
export ID_DATA=wr10k NAME=V14E MODE=e CENTER=off REALTIME=1
export TAG="${TAG:-e_confirmed}"
export TITLE="${TITLE:-v14 (E)   }"
export FPS="${FPS:-2}"
export CLIPS="${CLIPS:-12:00:00-12:03:20 13:00:00-13:03:20 13:10:00-13:16:40}"
exec bash "$HERE/run_v14.sh" "$@"
