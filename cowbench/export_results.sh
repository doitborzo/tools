#!/usr/bin/env bash
# Everything worth keeping from a training / test pod, in one archive to copy
# to a PC:
#
#   lora-runs/<run>/      adapter/, eval-lora*/ reports, dev history, train_meta, logs
#   lora-runs/base_w*/    the untouched model's eval (the control)
#   lora-runs/detector/   RT-DETRv2 weights (best/, ~170 MB), detections, report
#   cowbench/runs/tests_* run_tests.sh results (bench + stress reports)
#   cowbench/runs/quant_* run_quant_compare.sh results, one folder per checkpoint
#   summary.md            every eval's error rates in one table (cowbench/summary.py)
#
# Left out, as too big to be worth copying: the trainer's resume checkpoints
# (checkpoints/, with optimizer state), the model, the dataset, the venvs.
# The adapter snapshots taken at each dev score (adapters/step-*) are left
# out too unless SNAPSHOTS=1: adapter/ already is the best of them.
#
#   bash cowbench/export_results.sh
#   SNAPSHOTS=1 bash cowbench/export_results.sh
#
# Then, on the PC:   scp -P <port> root@<pod ip>:<archive> .

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
SNAPSHOTS="${SNAPSHOTS:-0}"
STAMP="$(date +%Y%m%d_%H%M)"
ARCHIVE="${ARCHIVE:-$WORK/cow_export_$STAMP.tgz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

[ -d "$WORK/lora-runs" ] || { echo "no $WORK/lora-runs here - set WORK=..."; exit 1; }

# ------------------------------------------------------------- summary table
PY="$(command -v python3 || command -v python)"
"$PY" "$HERE/summary.py" "$WORK/lora-runs" "$HERE/runs" -o "$STAGE/summary.md" > /dev/null

# ------------------------------------------------------------------ archive
excludes=(--exclude='*/checkpoints' --exclude='*/last.pt' --exclude='*.part'
          --exclude='lora-runs/current.log' --exclude='*/__pycache__')
[ "$SNAPSHOTS" = "1" ] || excludes+=(--exclude='lora-runs/*/adapters')

parts=(-C "$WORK" lora-runs -C "$STAGE" summary.md)
tests=()
if [ -d "$HERE/runs" ]; then
    while IFS= read -r d; do tests+=("$(basename "$d")"); done \
        < <(find "$HERE/runs" -maxdepth 1 -type d \( -name 'tests_*' -o -name 'quant_*' \) | sort)
fi
[ "${#tests[@]}" -gt 0 ] && parts+=(-C "$HERE/runs" "${tests[@]}")

echo "packing (this reads the network volume; a few minutes for a few GB)..."
tar czf "$ARCHIVE" "${excludes[@]}" "${parts[@]}"

echo
cat "$STAGE/summary.md"
echo
echo "In the archive (uncompressed size):"
tar tzvf "$ARCHIVE" | awk '{ n = split($6, p, "/"); key = (p[1] == "lora-runs" && n > 2) ? p[1] "/" p[2] : p[1];
                              size[key] += $3 }
                            END { for (k in size) printf "  %-45s %8.1f MB\n", k, size[k] / 1048576 }' | sort
echo
echo "Archive: $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
echo
echo "On the PC (PowerShell), port and IP from RunPod -> Connect -> 'SSH over exposed TCP':"
echo "  scp -P <port> root@<ip>:$ARCHIVE ."
echo "  tar -xzf $(basename "$ARCHIVE")"
