#!/usr/bin/env bash
# The same tests on several checkpoints of Muse Glimmer, one after another, on
# this machine's GPU (written for an RTX PRO 6000 Blackwell, which runs both
# FP8 and NVFP4 natively), then one comparison:
#
#   for each QUANT: run_tests.sh (vLLM restarted on that checkpoint with the
#   same adapter -> bench on the detector's boxes -> 12 cameras, detector live)
#   -> summary.md: errors and camera throughput side by side
#   -> compare_<a>_vs_<b>.md: paired test of the bench answers, cow by cow
#
# The adapter was trained against BF16 weights; on a 4-bit base it adds its
# deltas to weights that are not the ones it was trained with. The paired
# comparison says whether that costs accuracy, the stress tests what it buys.
#
# Usage (the repo checked out, the adapter and detector/best in place - see
# run_tests.sh):
#   bash cowbench/run_quant_compare.sh          start in tmux session "cowquant"
#   bash cowbench/run_quant_compare.sh log      follow the log
#   bash cowbench/run_quant_compare.sh stop
#
# Knobs (all optional; the rest of run_tests.sh's knobs pass through):
#   QUANTS="nvfp4 fp8"     checkpoints, in order: fp8, nvfp4 (RedHatAI W4A4),
#                          nvfp4-nvidia (NVIDIA mixed W4A16/FP8)
#   OUT_ROOT=<repo>/cowbench/runs/quant_<gpu>_<date>   one folder per QUANT inside
# A QUANT that fails (e.g. vLLM cannot put the LoRA on that checkpoint) is
# reported and the next one runs. Rerunning resumes: finished steps are kept.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
QUANTS="${QUANTS:-nvfp4 fp8}"
SESSION=cowquant
gpu_name="$( { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true; } | head -1)"
case "$gpu_name" in
    *"RTX PRO 6000"*) gtag=rtxpro6000 ;; *A100*) gtag=a100 ;; *H100*) gtag=h100 ;;
    *B200*) gtag=b200 ;; "") gtag=nogpu ;;
    *) gtag="$(echo "$gpu_name" | tr 'A-Z' 'a-z' | sed 's/nvidia//; s/[^a-z0-9]//g' | cut -c1-16)" ;;
esac
OUT_ROOT="${OUT_ROOT:-$HERE/runs/quant_${gtag}_$(date +%Y%m%d)}"
LOG="$OUT_ROOT/log.txt"

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "$SESSION" 2>/dev/null; tmux kill-session -t cowtests 2>/dev/null
          echo "stopped (vLLM keeps running in tmux 'vllm')"; exit 0 ;;
    "") ;;
    *) echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${COWQUANT_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "already running in tmux session '$SESSION':  bash $0 log"; exit 1
    fi
    mkdir -p "$OUT_ROOT"
    knobs=""
    for v in WORK QUANTS OUT_ROOT ADAPTER LORA_NAME DET_DIR DET_THRESHOLD DET_RESERVE_MIB LIVE \
             WIDTH UNIT STREAMS INTERVAL DURATION PORT MUSE_DETECT; do
        [ -n "${!v:-}" ] && knobs+="$v=$(printf '%q' "${!v}") "
    done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env COWQUANT_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/run_quant_compare.sh"); echo; echo '[run_quant_compare.sh finished]'; exec bash"
    echo "Started in tmux session '$SESSION' on $gpu_name: $QUANTS"
    echo "  log    :  bash $0 log      ($LOG)"
    echo "  results:  $OUT_ROOT"
    exit 0
fi

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$LOG") 2>&1
trap '' HUP
echo "#### $(date '+%F %T')  $gpu_name  checkpoints: $QUANTS  -> $OUT_ROOT"

declare -A status
for q in $QUANTS; do
    echo
    echo "################################################################ $q"
    if COWTESTS_IN_TMUX=1 QUANT="$q" OUT="$OUT_ROOT/$q" bash "$HERE/run_tests.sh"; then
        status[$q]=ok
    else
        status[$q]="failed (exit $?) - see $OUT_ROOT/$q/log.txt"
        echo "!! $q failed; going on with the next checkpoint"
    fi
done

echo
echo "################################################################ comparison"
PY="$(command -v python3 || command -v python)"
"$PY" - "$OUT_ROOT" "$gpu_name" $QUANTS > "$OUT_ROOT/summary.md" <<'PY'
import glob, json, os, sys
root, gpu, quants = sys.argv[1], sys.argv[2], sys.argv[3:]
def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
print(f"# Muse Glimmer checkpoints on {gpu}\n")
print("Same adapter, same questions (every cow of a keyframe in one request), cow boxes from "
      "RT-DETRv2; an annotated cow the detector missed counts as an error in every column.\n")
print("| checkpoint | base model | exact error | posture error | activity error | "
      "exact error on found cows | failed requests |")
print("|---|---|---|---|---|---|---|")
for q in quants:
    m, meta = load(os.path.join(root, q, "metrics.json")), load(os.path.join(root, q, "run_meta.json"))
    if not m:
        print(f"| {q} | - | no result | | | | |")
        continue
    n, miss = m["n_examples"], m.get("n_missed_by_detector", 0)
    found_err = (m["exact_match"]["error_rate"] * n - miss) / max(1, n - miss)
    print(f"| {q} | `{(meta or {}).get('model_repo', '?')}` | {m['exact_match']['error_rate']:.1%} | "
          f"{m['posture']['error_rate']:.1%} | {m['activity']['error_rate']:.1%} | {found_err:.1%} | "
          f"{m['n_failed'] - miss} |")
print("\n| checkpoint | camera test | frames/s, all cameras | update per camera, s | "
      "frame latency p50 / p90, s | detector p50, ms | keeps up |")
print("|---|---|---|---|---|---|---|")
for q in quants:
    for p in sorted(glob.glob(os.path.join(root, q, "stress_*.json"))):
        s = load(p)
        if not s:
            continue
        lat = s.get("frame_latency_s", {})
        fps = s.get("frames_per_s_total") or 0
        upd = s["streams"] / fps if fps else None
        det = (s.get("live_detector") or {}).get("detect_ms", {}).get("p50")
        keeps = (s.get("paced") or {}).get("keeps_up")
        f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
        mode = f"one frame per {s['interval']:g} s" if s.get("interval") else "as fast as possible"
        print(f"| {q} | {s['streams']} cameras, {mode} | {fps:.2f} | {f(upd)} | "
              f"{f(lat.get('p50'))} / {f(lat.get('p90'))} | {f(det, 0)} | "
              f"{'-' if keeps is None else ('yes' if keeps else 'no')} |")
PY
cd "$HERE"
set -- $QUANTS
ref="$1"; shift || true
for q in "$@"; do
    if [ -s "$OUT_ROOT/$ref/results.jsonl" ] && [ -s "$OUT_ROOT/$q/results.jsonl" ]; then
        "$PY" cowbench.py compare --a "$OUT_ROOT/$ref" --b "$OUT_ROOT/$q" --label-a "$ref" --label-b "$q" \
            --output "$OUT_ROOT/compare_${ref}_vs_${q}.md" || true
    fi
done

echo
cat "$OUT_ROOT/summary.md"
echo
for q in $QUANTS; do echo "  $q: ${status[$q]}"; done
echo "Everything in $OUT_ROOT (bash cowbench/export_results.sh packs it)."
