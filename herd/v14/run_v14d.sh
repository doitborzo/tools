#!/usr/bin/env bash
# Version 14 (D): the ID transformer trained from scratch on WildlifeReID-10k's cattle (~300 cows:
# Cows2021, MultiCamCows2024, OpenCows2020, CowDataset; Kaggle, read in place, no account), then
# tested on all 16 of MmCows' cows (none trained on). The association transformer is v14 (A)'s
# (scored on the test window), the detector version 10's. Way C (the barn's average cow taken out)
# is scored on the same run.
#
#   bash herd/v14/run_v14d.sh          start (tmux session "v14"): environment, detector, MmCows,
#                                      WildlifeReID-10k, ID training, onboarding test, videos
#   bash herd/v14/run_v14d.sh log      follow the log
#   bash herd/v14/run_v14d.sh stop
#
# -> /workspace/V14D_RESULTS.tar, V14D_MODELS.tgz, V14D_VIDEO.tar (3 clips at the barn's speed)
# Knobs: ASSOC_PRETRAINED (v14 A's V14A_MODELS.tgz)  ID_EPOCHS=60  END=15  CLIPS  FPS=2  and run_v14.sh's
# WildlifeReID-10k: non-commercial, no re-upload of its images, attribute every source set.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    log)  exec bash "$HERE/run_v14.sh" log ;;
    stop) bash "$HERE/video_v14.sh" stop; exec bash "$HERE/run_v14.sh" stop ;;
esac
unset PRETRAINED
export ID_DATA=wr10k NAME=V14D CENTER=off REALTIME=1
export ASSOC_PRETRAINED="${ASSOC_PRETRAINED:-https://drive.google.com/file/d/1YKPWnocmLrWZLgI90ex-euM2QurcYZjF/view?usp=sharing}"
export TITLE="${TITLE:-v14 (D)   }"
export FPS="${FPS:-2}"
export CLIPS="${CLIPS:-12:00:00-12:03:20 13:00:00-13:03:20 13:10:00-13:16:40}"
exec bash "$HERE/run_v14.sh" "$@"
