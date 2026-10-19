#!/usr/bin/env bash
# Version 14 (C) - Версія 14 (В), the third version: v14 (A)'s trained weights, nothing retrained; at onboarding the
# barn's "average cow" is measured in the setup hour and taken out of every fingerprint before
# the album is built (and the album threshold is calibrated there, without labels). Way A is
# scored on the same run for comparison.
#
#   bash herd/v14/run_v14c.sh          start (tmux session "v14"): environment, detector, MmCows,
#                                      the weights, the onboarding test, videos, results, weights
#   bash herd/v14/run_v14c.sh log      follow the log (the video part: /workspace/v14/video_log.txt)
#   bash herd/v14/run_v14c.sh stop
#
# -> /workspace/V14C_RESULTS.tar, V14C_MODELS.tgz, V14C_VIDEO.tar (3 clips at the barn's speed)
# Knobs: PRETRAINED (v14 A's V14A_MODELS.tgz)  VIDEO_END=15 (the test runs 12:00 -> 15:00, as A's
#        results did; 13.5 is faster)  CLIPS  FPS=2  and run_v14.sh's

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    log)  exec bash "$HERE/run_v14.sh" log ;;
    stop) bash "$HERE/video_v14.sh" stop; exec bash "$HERE/run_v14.sh" stop ;;
esac
export PRETRAINED="${PRETRAINED:-https://drive.google.com/file/d/1YKPWnocmLrWZLgI90ex-euM2QurcYZjF/view?usp=sharing}"
export CENTER=on NAME=V14C REALTIME=1
export TITLE="${TITLE:-v14 (C)   }"
export FPS="${FPS:-2}"
export VIDEO_END="${VIDEO_END:-15}"
export CLIPS="${CLIPS:-12:00:00-12:03:20 13:00:00-13:03:20 13:10:00-13:16:40}"
exec bash "$HERE/run_v14.sh" "$@"
