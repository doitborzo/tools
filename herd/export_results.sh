#!/usr/bin/env bash
# Everything worth keeping from the herd runs on a pod, in one archive to copy
# to a PC (cowbench/export_results.sh does the same for the LoRA runs):
#
#   herd/<run>/            eval_val.json, eval_det.json, abstain.json, heads.json, history.jsonl,
#                          eval-val*/ (report.md, metrics.json, results.jsonl), model.pt (a few MB)
#   herd/detector_*/       the detectors trained by run_pod.sh detector: det_train_meta.json,
#                          det_history.jsonl, det_report.md, log.txt; the weights only with DET_WEIGHTS=1
#   herd/stress_*/         the real-time tests (stress_herd_*cam.md / .json)
#   herd/log_*.txt         the run logs
#   summary.md             every result in one table (cowbench/summary.py)
#
# Left out: the venv, the dataset, the DINOv2 features (tens of GB, recomputed
# by run_pod.sh), the stress tests' databases and galleries, the detectors'
# resume checkpoints, and the detectors' weights unless DET_WEIGHTS=1
# (~170 MB for R50, ~300 MB for R101).
#
#   bash herd/export_results.sh
#   DET_WEIGHTS=1 bash herd/export_results.sh
#
# Then, on the PC:   scp -P <port> root@<pod ip>:<archive> .

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="${WORK:-/workspace}"
DET_WEIGHTS="${DET_WEIGHTS:-0}"
STAMP="$(date +%Y%m%d_%H%M)"
ARCHIVE="${ARCHIVE:-$WORK/herd_export_$STAMP.tgz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

[ -d "$WORK/herd" ] || { echo "no $WORK/herd here - set WORK=..."; exit 1; }

# ------------------------------------------------------------- summary table
PY="$WORK/herd/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
"$PY" "$REPO/cowbench/summary.py" "$WORK/herd" -o "$STAGE/summary.md" > /dev/null \
    || echo "(summary.py failed - the archive is made without summary.md)"

# ------------------------------------------------------------------ archive
excludes=(--exclude='herd/.venv' --exclude='herd/features*' --exclude='herd/barn'
          --exclude='*/last.pt' --exclude='*.part' --exclude='*/__pycache__'
          --exclude='*.sqlite' --exclude='*.sqlite-wal' --exclude='*.sqlite-shm'
          --exclude='herd/stress_*/gallery')
[ "$DET_WEIGHTS" = "1" ] || excludes+=(--exclude='herd/detector_*/best/*.safetensors'
                                       --exclude='herd/detector_*/best/*.bin')

parts=(-C "$WORK" herd)
[ -f "$STAGE/summary.md" ] && parts+=(-C "$STAGE" summary.md)

echo "packing $WORK/herd (features, venv and weights left out)..."
tar czf "$ARCHIVE" "${excludes[@]}" "${parts[@]}"

echo
[ -f "$STAGE/summary.md" ] && cat "$STAGE/summary.md" && echo
echo "In the archive (uncompressed size):"
tar tzvf "$ARCHIVE" | awk '{ n = split($6, p, "/"); key = (p[1] == "herd" && n > 2) ? p[1] "/" p[2] : p[1];
                              size[key] += $3 }
                            END { for (k in size) printf "  %-45s %8.1f MB\n", k, size[k] / 1048576 }' | sort
echo
echo "Archive: $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
echo
echo "On the PC (PowerShell), port and IP from RunPod -> Connect -> 'SSH over exposed TCP':"
echo "  scp -P <port> root@<ip>:$ARCHIVE ."
echo "  tar -xzf $(basename "$ARCHIVE")"
