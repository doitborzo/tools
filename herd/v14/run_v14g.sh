#!/usr/bin/env bash
# Version 14 (G): version 14 (E) with three hours of onboarding instead of one. Same system (no tracking
# model, every box named on its own), same ID transformer (version 14 D's, trained on WildlifeReID-10k's
# cattle; it never saw MmCows), same detector (version 10's), nothing trained here. What changes is the
# album: built from 09:00-12:00 on 25 July instead of one hour, so each cow is seen in more poses, places
# and light. Scored on 12:00-13:00, the hour the detector never trained on.
#
# The control runs on the same pod right after: the same scored hour after one hour of onboarding
# (11:00-12:00). Only that pair compares like with like - version 14 (E)'s own scores cover 13:00-15:00,
# a longer and later stretch. Three albums each (built, confirmed, oracle), as E.
#
#   bash herd/v14/run_v14g.sh          start (tmux session "v14"); frames before 12:00 come from MmCows'
#                                      whole-day 1-second zips (~4 GB per camera for 09:00-12:00)
#   bash herd/v14/run_v14g.sh log      follow the log (the system part: /workspace/v14/video_log.txt)
#   bash herd/v14/run_v14g.sh stop
#
# -> /workspace/V14G_RESULTS.tar (RESULTS.md: G's three albums, then G against the control; control/: the
#    control's own scores), V14G_MODELS.tgz, V14G_VIDEO.tar (12:00, 12:10 and 12:50, the confirmed album)
# Knobs: SYS_START=9 SETUP_MIN=180 SYS_END=13  CONTROL_START=11 CONTROL_MIN=60  TAG=e_confirmed  CLIPS  FPS=2
# The detector trained on MmCows' labelled frames of 09:00-12:00 (it held out only hours 0, 6, 12, 18): its
# boxes during the onboarding may be a little flattering; the scored hour is clean.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    log)  exec bash "$HERE/run_v14.sh" log ;;
    stop) bash "$HERE/video_v14.sh" stop; exec bash "$HERE/run_v14.sh" stop ;;
esac
unset PRETRAINED ASSOC_PRETRAINED
export ID_DATA=wr10k NAME=V14G MODE=e CENTER=off REALTIME=1
export SYS_START="${SYS_START:-9}" SETUP_MIN="${SETUP_MIN:-180}" SYS_END="${SYS_END:-13}"
export CONTROL_START="${CONTROL_START:-11}" CONTROL_MIN="${CONTROL_MIN:-60}"
export TAG="${TAG:-e_confirmed}"
export TITLE="${TITLE:-v14 (G)   }"
export FPS="${FPS:-2}"
exec bash "$HERE/run_v14.sh" "$@"
