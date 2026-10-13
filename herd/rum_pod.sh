#!/usr/bin/env bash
# Rumination on a GPU pod, CBVD-5 not used: DINOv2 embeddings of every frame -> a temporal transformer,
# trained on Kaggle's Beef Cattle Behavior (barn) and CSIRO's CVB (pasture) at once (herd/rumination.py).
#
#   1. the environment (run_pod.sh step 1)
#   2. fetch: Kaggle's per-cow clips (~6.4 GB of a 48.5 GB zip, read in place), CVB's annotations and
#      every CVB_STEP-th frame (2: 15 fps, ~6.9 GB in ~113k files - the slow part, an hour or more)
#   3. extract (GPU): DINOv2-S on every frame of every cow, 7 s windows, motion rhythm per window
#   4. train: the transformer twice - frames only (WORK/rum/model) and frames + motion rhythm
#      (WORK/rum/model_motion) - and a summary of both on the held-out recordings
#
#   bash herd/rum_pod.sh          start in tmux "herd-rum"
#   bash herd/rum_pod.sh log      follow it
#   bash herd/rum_pod.sh stop
#
# Knobs: WORK=/workspace  SOURCES=kaggle,cvb  CVB_STEP=2  CVB_CLEAN=1 (delete CVB frames once encoded)
#        EPOCHS=30  MOTION="0 1" (the variants trained)
# Every step resumes: files fetched and sequences encoded are kept, a rerun goes on from there.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
R="$WORK/rum"
LOG="$R/log.txt"
SESSION=herd-rum

case "${1:-}" in
    log)  exec tail -n 100 -F "$LOG" ;;
    stop) tmux kill-session -t "=$SESSION" 2>/dev/null && echo "stopped $SESSION" || echo "$SESSION not running"; exit 0 ;;
    "")   ;;
    *)    echo "usage: $0 [log|stop]"; exit 2 ;;
esac

if [ -z "${HERD_RUM_IN_TMUX:-}" ]; then
    command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        if tmux capture-pane -p -t "=$SESSION:" -S -300 | grep -q "\[rumination finished\]"; then
            tmux kill-session -t "=$SESSION"
        else
            echo "already running: bash $0 log"; exit 1
        fi
    fi
    mkdir -p "$R"
    knobs=""
    for v in WORK SOURCES CVB_STEP CVB_CLEAN EPOCHS MOTION; do knobs+="$v=$(printf '%q' "${!v:-}") "; done
    env -u TMUX tmux new-session -d -s "$SESSION" -x 200 -y 50 \
        "env HERD_RUM_IN_TMUX=1 $knobs bash $(printf '%q' "$HERE/rum_pod.sh"); echo; echo '[rumination finished]'; exec bash"
    echo "Started in tmux session '$SESSION'.  log: bash $0 log   ($LOG)"
    exit 0
fi

mkdir -p "$R"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $*   [$(date '+%F %T')]"; }
SOURCES="${SOURCES:-kaggle,cvb}"
PY="$WORK/herd/.venv/bin/python"
export HF_HOME="$WORK/hf-cache"
export PATH="$HOME/.local/bin:$PATH"

step "Environment (run_pod.sh step 1)"
HERD_IN_TMUX=1 SETUP_ONLY=env WORK="$WORK" RUN=setup bash "$HERE/run_pod.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

step "Fetch ($SOURCES)"
"$PY" "$HERE/rumination.py" fetch --data "$R" --sources "$SOURCES" --cvb-step "${CVB_STEP:-2}"

step "Extract: DINOv2 on every frame (GPU)"
clean=(); [ "${CVB_CLEAN:-1}" = "1" ] && clean=(--cvb-clean)
"$PY" "$HERE/rumination.py" extract --data "$R" --sources "$SOURCES" "${clean[@]}"

for m in ${MOTION:-0 1}; do
    step "Train: the transformer, motion rhythm $m"
    "$PY" "$HERE/rumination.py" train --data "$R" --sources "$SOURCES" --motion "$m" --epochs "${EPOCHS:-30}"
done

step "Summary"
"$PY" - "$R" <<'EOF'
import json, os, sys
R = sys.argv[1]
print("| model | source | windows | ruminating | precision | recall | F1 | AUC | behaviour accuracy |")
print("|---|---|---|---|---|---|---|---|---|")
for name in ("model", "model_motion"):
    p = os.path.join(R, name, "eval.json")
    if not os.path.exists(p):
        continue
    e = json.load(open(p))
    for s, v in e["val"].items():
        a = v["at_threshold"]
        print(f"| {name} (thr {e['threshold']}) | {s} | {v['n']} | {v['ruminating']} | {a['precision']:.1%} | "
              f"{a['recall']:.1%} | {a['f1']:.3f} | {(v['auc'] or 0):.3f} | {v['behaviour_accuracy']:.1%} |")
EOF
mkdir -p "$WORK/export"
T="$WORK/export/herd_rumination_$(date +%Y%m%d_%H%M).tgz"
(cd "$WORK" && tar -czf "$T" $(cd "$WORK" && ls -d rum/model rum/model_motion rum/log.txt rum/feat/features.json 2>/dev/null))
echo "models + eval: $T   (scp -P <port> root@<ip>:$T .)"
