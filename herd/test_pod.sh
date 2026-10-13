#!/usr/bin/env bash
# A new pod, a trained herd: how good and how fast on this GPU (e.g. an RTX 2000 Ada, 16 GB).
#
#   1. the environment and CBVD-5 (run_pod.sh's steps 1-2; CBVD-5 is downloaded here)
#   2. the models from the archives export_pod.sh made, uploaded to WORK:
#        herd_models_*.tgz   detectors' best/, RUN's model.pt + abstain.json, DINOv2 (hf-cache)
#        herd_results_*.tgz  (optional) RUN's eval-val/results.jsonl: eval-det then also scores
#                            the annotated boxes, for comparison
#   3. the usual test: eval-det on CBVD-5 val (detector -> DINOv2 -> transformer, every threshold)
#   4. the stress test: one burst alone, then CAMERAS cameras in real time (herd.py stress)
#   5. a summary (GPU, detector, error, recall, keeps up or not) and herd_test_<gpu>_<date>.tgz
#
#   bash herd/test_pod.sh          start in tmux "herd-test"
#   bash herd/test_pod.sh log      follow it
#   bash herd/test_pod.sh stop
#
# Knobs: WORK=/workspace  RUN=run9  DET=<a best/ folder> (default: the newest detector unpacked)
#        CAMERAS=5  DURATION=300 (s measured)  THRESHOLD (default: the detector's own)
#        TESTS="eval stress" (one of them to run only it)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
RUN="${RUN:-run9}"
H="$WORK/herd"
LOG="$H/log_test.txt"
SESSION=herd-test

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_TEST_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[test finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$H"
    knobs=""
    for v in WORK RUN DET CAMERAS DURATION THRESHOLD TESTS MODELS; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_TEST_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/test_pod.sh"); echo; echo '[test finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$H"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
TESTS="${TESTS:-eval stress}"
VENV="$H/.venv"
PY="$VENV/bin/python"
DATA="$WORK/cbvd5"
export PATH="$HOME/.local/bin:$PATH"

step "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "!! no GPU"; exit 1; }
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GTAG="$(echo "$GPU" | tr -c 'A-Za-z0-9' '_' | tr -s _ | sed 's/^_//; s/_$//')"

step "Models from the uploaded archives"
for a in ${MODELS:-$(ls -t "$WORK"/herd_models_*.tgz 2>/dev/null | head -1)} \
         $(ls -t "$WORK"/herd_results_*.tgz 2>/dev/null | head -1); do
    [ -f "$a" ] || continue
    echo "unpacking $a"
    tar -xzf "$a" -C "$WORK"
done
[ -f "$H/$RUN/model.pt" ] || { echo "!! no $H/$RUN/model.pt - upload herd_models_*.tgz to $WORK (or RUN=<run>)"; exit 1; }
if [ -z "${DET:-}" ]; then
    DET="$(find "$H" -maxdepth 3 -name det_train_meta.json -path '*/best/*' -printf '%T@ %h\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
fi
[ -n "$DET" ] && [ -f "$DET/det_train_meta.json" ] || { echo "!! no detector best/ under $H (DET=<it>)"; exit 1; }
DTAG="$(basename "$(dirname "$DET")")"
echo "model    $H/$RUN/model.pt"
echo "detector $DET"
echo "detectors here: $(cd "$H" && ls -d detector_*/best 2>/dev/null | tr '\n' ' ')"

step "Environment and CBVD-5 (run_pod.sh steps 1-2)"
HERD_IN_TMUX=1 SETUP_ONLY=1 WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
if grep -q '"kind": *"rfdetr"' "$DET/det_train_meta.json"; then
    "$PY" -c "import rfdetr" 2>/dev/null || uv pip install --python "$PY" rfdetr
fi
export HF_HOME="$WORK/hf-cache"
if ls -d "$HF_HOME"/hub/models--facebook--dinov2-* >/dev/null 2>&1; then
    echo "DINOv2 from the archive ($HF_HOME)"       # found in the cache: not downloaded again
fi
"$PY" -c "import torch; p = torch.cuda.get_device_properties(0); print(f'torch {torch.__version__}, {p.name}, {p.total_memory / 2**30:.1f} GB, bf16 {torch.cuda.is_bf16_supported()}')"

cd "$HERE"
EOUT="$H/$RUN/eval-val-det-${DTAG}-${GTAG}"
SOUT="$H/stress_${RUN}_${DTAG}_${GTAG}"
thr=(); [ -n "${THRESHOLD:-}" ] && thr=(--threshold "$THRESHOLD")

if [[ " $TESTS " == *" eval "* ]]; then
    step "Usual test: eval-det on CBVD-5 val ($DTAG)"
    "$PY" herd.py eval-det --run "$H/$RUN" --detector "$DET" --root "$DATA" --out "$EOUT" "${thr[@]}"
fi
if [[ " $TESTS " == *" stress "* ]]; then
    step "Stress test: ${CAMERAS:-5} cameras, ${DURATION:-300} s"
    mkdir -p "$SOUT"
    "$PY" herd.py stress --model "$H/$RUN/model.pt" --detector "$DET" --root "$DATA" \
        --cameras "${CAMERAS:-5}" --duration "${DURATION:-300}" --out "$SOUT" "${thr[@]}"
fi

step "Summary"
SUM="$H/test_summary_${GTAG}_$(date +%Y%m%d_%H%M).md"
"$PY" - "$GPU" "$DET" "$EOUT/eval_det.json" "$SOUT" "$SUM" <<'EOF'
import glob, json, os, sys
gpu, det, ev, sout, out = sys.argv[1:]
lines = [f"# herd test on {gpu}", "", f"detector: `{det}`", ""]
if os.path.exists(ev):
    e = json.load(open(ev))
    q, d = e["detector_quality"]["iou0.5"], e["on_detector_boxes"]
    best = min(e["threshold_sweep"], key=lambda r: r["exact_error"])
    lines += ["## Usual test (CBVD-5 val)", "",
              f"- at the detector's threshold {e['threshold']}: recall {q['recall']:.1%}, precision {q['precision']:.1%} (IoU 0.5)",
              f"- whole-path error {d['exact_error']:.1%} (posture {d['posture_error']:.1%}, activity {d['activity_error']:.1%}),"
              f" on found cows {d['exact_error_found_cows']:.1%}",
              f"- best threshold {best['threshold']}: error {best['exact_error']:.1%}, on found cows {best['exact_error_found_cows']:.1%}",
              f"- detector {e['detector_quality'].get('ms_per_frame_p50')} ms a frame (p50)"]
    if e.get("on_annotated_boxes"):
        lines.append(f"- on the annotated boxes: error {e['on_annotated_boxes']['exact_error']:.1%}")
    lines.append("")
for f in sorted(glob.glob(os.path.join(sout, "*.md"))):
    lines += [f"## Stress test ({os.path.basename(f)})", "", open(f, encoding="utf-8").read(), ""]
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
EOF
mkdir -p "$WORK/export"
T="$WORK/export/herd_test_${GTAG}_$(date +%Y%m%d_%H%M).tgz"
(cd "$WORK" && tar -czf "$T" $(for p in "$SUM" "$EOUT" "$SOUT" "$LOG"; do [ -e "$p" ] && echo "${p#$WORK/}"; done))
echo
echo "summary: $SUM"
echo "archive: $T   (scp -P <port> root@<ip>:$T .)"
