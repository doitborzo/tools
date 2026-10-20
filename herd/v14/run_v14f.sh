#!/usr/bin/env bash
# Version 14 (F): OpenCowID's identification model (Prabhune and Kim, WACV 2026; github.com/neis-lab/
# OpenCowID) in place of the ID transformer, on version 14 (D)'s data and tests: a ResNet-50 with
# ImageNet weights and a 128-number projection, trained by metric learning around virtual class
# centroids - first on OpenCowID's synthetic coats (no real cow: zero-shot), then on the cattle of
# WildlifeReID-10k (v14 D's training cows; the pictures, not DINOv2 vectors) - and tested on all 16
# of MmCows' cows (none trained on): the ID scores, the onboarding test with v14 (A)'s association
# transformer and version 10's detector, the videos. The zero-shot model is scored too.
#
#   bash herd/v14/run_v14f.sh          start (tmux session "v14"): environment, detector, MmCows,
#                                      the pictures, OpenCowID training, onboarding test, videos
#   bash herd/v14/run_v14f.sh log      follow the log
#   bash herd/v14/run_v14f.sh stop
#
# -> /workspace/V14F_RESULTS.tar, V14F_MODELS.tgz, V14F_VIDEO.tar (3 clips at the barn's speed)
# Then, D's comparison on A's five cows: NAME=V14F bash herd/v14/test_like_a.sh
# Knobs: ID_STAGES=synth,wr10k (synth alone = OpenCowID as published, zero-shot; wr10k alone = its
#        model on D's data only)  ID_EPOCHS=30 (real stage)  ASSOC_PRETRAINED (v14 A's V14A_MODELS.tgz)
#        END=15  CLIPS  FPS=2  and run_v14.sh's; opencowid.py's own knobs (--synth-ids 2000, --synth-epochs
#        10, --P 32 --K 4, --lr 1e-5) are its defaults
# WildlifeReID-10k: non-commercial, no re-upload of its images, attribute every source set. OpenCowID: MIT.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    log)  exec bash "$HERE/run_v14.sh" log ;;
    stop) bash "$HERE/video_v14.sh" stop; exec bash "$HERE/run_v14.sh" stop ;;
esac
unset PRETRAINED
export ID_DATA=ocid NAME=V14F CENTER=off REALTIME=1
export ID_EPOCHS="${ID_EPOCHS:-30}"
export ASSOC_PRETRAINED="${ASSOC_PRETRAINED:-https://drive.google.com/file/d/1YKPWnocmLrWZLgI90ex-euM2QurcYZjF/view?usp=sharing}"
export TITLE="${TITLE:-v14 (F)   }"
export FPS="${FPS:-2}"
export CLIPS="${CLIPS:-12:00:00-12:03:20 13:00:00-13:03:20 13:10:00-13:16:40}"
exec bash "$HERE/run_v14.sh" "$@"
