#!/usr/bin/env bash
# Everything worth keeping from a training / test pod, in one archive to copy
# to a PC:
#
#   lora-runs/<run>/      adapter/, eval-lora*/ reports, dev history, train_meta, logs
#   lora-runs/base_w*/    the untouched model's eval (the control)
#   lora-runs/detector/   RT-DETRv2 weights (best/, ~170 MB), detections, report
#   cowbench/runs/tests_* run_tests.sh results (bench + stress reports)
#   cowbench/runs/quant_* run_quant_compare.sh results, one folder per checkpoint
#   summary.md            every eval's error rates in one table
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
"$PY" - "$WORK/lora-runs" "$HERE/runs" > "$STAGE/summary.md" <<'PY'
import glob, json, os, sys
runs, bench_runs = sys.argv[1], sys.argv[2]
rows = []
def add(path):
    d = os.path.dirname(path)
    try:
        m = json.load(open(path, encoding="utf-8"))
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8")) \
            if os.path.exists(os.path.join(d, "run_meta.json")) else {}
    except Exception:
        return
    boxes = (meta.get("boxes") or {}).get("source", "annotation")
    rows.append((os.path.relpath(path, os.path.dirname(runs) if path.startswith(runs) else os.path.dirname(bench_runs)),
                 boxes, m.get("n_examples"),
                 m["exact_match"]["error_rate"], m["posture"]["error_rate"], m["activity"]["error_rate"],
                 m.get("n_missed_by_detector", 0)))
for pat in (os.path.join(runs, "*", "metrics*.json"), os.path.join(runs, "*", "*", "metrics*.json"),
            os.path.join(bench_runs, "tests_*", "metrics*.json"),
            os.path.join(bench_runs, "quant_*", "*", "metrics*.json")):
    for p in sorted(glob.glob(pat)):
        add(p)
print("# Results\n")
print("| metrics file | boxes | cows | exact error | posture error | activity error | missed by detector |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1%} | {r[4]:.1%} | {r[5]:.1%} | {r[6]} |")
det = os.path.join(runs, "detector", "det_metrics.json")
if os.path.exists(det):
    d = json.load(open(det, encoding="utf-8"))
    i5 = d.get("iou0.5", {})
    print(f"\nDetector: recall {i5.get('recall', 0):.1%}, precision {i5.get('precision', 0):.1%} "
          f"at IoU 0.5; {d.get('ms_per_frame')} ms a frame.")
stress = sorted(glob.glob(os.path.join(bench_runs, "tests_*", "stress_*.json"))
                + glob.glob(os.path.join(bench_runs, "quant_*", "*", "stress_*.json")))
if stress:
    print("\n| stress test | frames/s all cameras | frame latency p50 / p90, s | keeps up |")
    print("|---|---|---|---|")
    for p in stress:
        s = json.load(open(p, encoding="utf-8"))
        lat = s.get("frame_latency_s", {})
        keeps = (s.get("paced") or {}).get("keeps_up")
        f = lambda v: "-" if v is None else f"{v:.1f}"
        print(f"| {os.path.relpath(p, bench_runs)} | {s.get('frames_per_s_total', 0):.2f} | "
              f"{f(lat.get('p50'))} / {f(lat.get('p90'))} | {'-' if keeps is None else ('yes' if keeps else 'no')} |")
PY

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
