#!/usr/bin/env bash
# Everything worth keeping from a pod, in three archives under WORK/export:
#
#   herd_models_<date>.tgz    what herd needs to run elsewhere:
#                             the detectors' best/ folders (checkpoint, threshold, box calibration),
#                             the herd model of RUN (model.pt: temporal transformer + heads, with the
#                             frame encoder's settings) and its NaN model (abstain.json),
#                             the frame encoder's own weights (DINOv2 from the HF cache)
#   herd_results_<date>.tgz   the benchmark: eval / eval-det / eval_extra / stress results, the
#                             detectors' training logs and metadata, the public sets' licences
#   herd_galleries_<date>.tgz the show-dets galleries (pictures; can be large)
#
#   bash herd/export_pod.sh                 RUN=run9 by default, every detector_*/best
#   RUN=run10 bash herd/export_pod.sh
#   DETS="detector_rfdetr_large detector_rfdetr_large_zenodo_mmcows_opencows_ft" bash herd/export_pod.sh
#
# Left out: datasets, features, rf/ training checkpoints (last.ckpt holds the optimiser too, GBs).
# Only files that exist are packed; the lists of what went in are printed.

set -euo pipefail
WORK="${WORK:-/workspace}"
RUN="${RUN:-run9}"
H="$WORK/herd"
OUT="$WORK/export"
STAMP="$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
cd "$WORK"

# paths relative to WORK, only those that exist
keep() { for p in "$@"; do [ -e "$p" ] && echo "$p"; done; return 0; }

if [ -n "${DETS:-}" ]; then dets=($DETS); else dets=($(cd "$H" && ls -d detector_*/ 2>/dev/null | tr -d /)); fi

# ---------------------------------------------------------------- models
L="$OUT/models_$STAMP.txt"
{
    for d in "${dets[@]}"; do
        keep "herd/$d/best/checkpoint.pth" "herd/$d/best/det_train_meta.json" "herd/$d/best/box_calib.json" \
             "herd/$d/best/config.json" "herd/$d/best/model.safetensors" "herd/$d/best/preprocessor_config.json"
    done
    keep "herd/$RUN/model.pt" "herd/$RUN/abstain.json" "herd/$RUN/train_meta.json"
    # the frame encoder (DINOv2) as the HF cache holds it: copy hf-cache/ back and set HF_HOME
    for m in hf-cache/hub/models--facebook--dinov2-*; do keep "$m"; done
} > "$L"
cat > "$OUT/HOW_TO_USE.txt" <<EOF
herd models exported $(date '+%F %T') from $(hostname)

herd/<detector>/best/      a detector: python herd/herd.py run ... --detector <it>, or [detector] in the config
herd/$RUN/model.pt          the herd model: temporal transformer + frame heads; its frame-encoder settings
                           (encoder name, grid, crop, margin) are stored inside
herd/$RUN/abstain.json      the NaN model (when an answer is withheld)
hf-cache/                  DINOv2, the frame encoder: unpack next to herd/ and
                           export HF_HOME=<that folder>/hf-cache HF_HUB_OFFLINE=1
Unpack: tar -xzf herd_models_*.tgz -C /workspace
EOF
echo "HOW_TO_USE.txt" >> "$L"

# --------------------------------------------------------------- results
R="$OUT/results_$STAMP.txt"
{
    keep "herd/$RUN/eval_val.json" "herd/$RUN/eval_det.json" "herd/$RUN/eval-val/results.jsonl"
    find herd -maxdepth 3 \( -path "herd/*/eval-val-det*/*.json" -o -path "herd/*/eval-val/*.json" \) 2>/dev/null
    find herd -maxdepth 2 \( -name "log_*.txt" -o -path "herd/stress_*/*.md" -o -path "herd/stress_*/*.json" \
         -o -path "herd/stress_*/log.txt" \) 2>/dev/null
    for d in "${dets[@]}"; do
        keep "herd/$d/log.txt" "herd/$d/best/det_train_meta.json" "herd/$d/best/box_calib.json" "herd/$d/train_done"
        keep "herd/$d/rf/metrics.csv"
    done
    keep extra_det/eval_extra.json
    for s in extra_det/*/source.json; do keep "$s"; done
    for g in herd/show_dets*/summary.json; do keep "$g"; done
} | sort -u > "$R"

# ------------------------------------------------------------- galleries
G="$OUT/galleries_$STAMP.txt"
(cd "$WORK" && ls -d herd/show_dets* 2>/dev/null || true) > "$G"

pack() {   # list, archive name
    if [ -s "$1" ]; then
        tar -czf "$OUT/$2" -C "$WORK" -T "$1"
        echo "== $OUT/$2  ($(du -h "$OUT/$2" | cut -f1))"
        sed 's/^/   /' "$1"
    else
        echo "== $2: nothing to pack"
    fi
}
cp "$OUT/HOW_TO_USE.txt" "$WORK/HOW_TO_USE.txt"
pack "$L" "herd_models_$STAMP.tgz"
pack "$R" "herd_results_$STAMP.tgz"
pack "$G" "herd_galleries_$STAMP.tgz"
rm -f "$WORK/HOW_TO_USE.txt"
grep -q "model.pt" "$L" || echo "!! no $H/$RUN/model.pt - the herd model is not in the archive (RUN=<run>?)"
grep -q "dinov2" "$L" || echo "!! no DINOv2 in $WORK/hf-cache - it downloads again on first use (needs internet)"
echo
echo "copy to your machine:  scp -P <port> root@<ip>:$OUT/herd_*_$STAMP.tgz ."
