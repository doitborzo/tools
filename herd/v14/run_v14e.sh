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
#        SYS_START SETUP_MIN SYS_END: a longer onboarding, scored on the hour after it, e.g. three hours of
#        onboarding then 12:00-13:00 (the hour the detector never trained on), and the control - the same hour
#        scored after one hour of onboarding - run one after the other:
#          SYS_START=9 SETUP_MIN=180 SYS_END=13 NAME=V14E_ONB3H bash herd/v14/run_v14e.sh
#          SYS_START=11 SETUP_MIN=60 SYS_END=13 NAME=V14E_ONB1H bash herd/v14/run_v14e.sh
#        (frames before 12:00 come from MmCows' whole-day zips: ~4 GB per camera for 9:00-12:00)

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
if [ -n "${SETUP_MIN:-}" ] || [ -n "${SYS_START:-}" ]; then
    # clips in the scored part: its first 3:20, 10:00-16:40 into it, 50:00-53:20 into it
    e=$(awk -v s="${SYS_START:-12}" -v m="${SETUP_MIN:-60}" 'BEGIN { printf "%d", s * 60 + m }')
    hm() { printf '%02d:%02d:%02d' $(( ($1 + $2 / 60) / 60 )) $(( ($1 + $2 / 60) % 60 )) $(( $2 % 60 )); }
    export CLIPS="${CLIPS:-$(hm "$e" 0)-$(hm "$e" 200) $(hm "$e" 600)-$(hm "$e" 1000) $(hm "$e" 3000)-$(hm "$e" 3200)}"
fi
export CLIPS="${CLIPS:-12:00:00-12:03:20 13:00:00-13:03:20 13:10:00-13:16:40}"
exec bash "$HERE/run_v14.sh" "$@"
