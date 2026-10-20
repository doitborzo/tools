#!/usr/bin/env bash
# Version 14 (D)'s trained weights (or F's: NAME=V14F) on version 14 (A)'s test - the same five MmCows
# cows A never saw (2, 6, 10, 11, 15), nothing retrained, nothing run again but the scoring:
#   1  the ID transformer alone: album = those cows before 12:00, queries = after 15:00 (as A's 29.3%)
#   2  the onboarding test (D's saved 12:00-15:00 run) scored on those cows only (as A's 14.7%)
#
#   bash herd/v14/test_like_a.sh             -> /workspace/V14D_TEST_LIKE_A.tar (TEST_LIKE_A.md + the scores)
#   NAME=V14F bash herd/v14/test_like_a.sh   -> /workspace/V14F_TEST_LIKE_A.tar (OpenCowID's model, v14 F)
# Needs the version 14 (D) run on this pod (/workspace/v14/idd, idnet_wr10k/, system_V14D/) - or F's
# (iddi/, idnet_ocid/, system_V14F/).
# Knobs: NAME=V14D  COWS=2,6,10,11,15  WORK=/workspace

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
V="$WORK/v14"
PY="$WORK/herd/.venv/bin/python"
COWS="${COWS:-2,6,10,11,15}"
NAME="${NAME:-V14D}"
SYS="$V/system_$NAME"
O="$WORK/${NAME}_TEST_LIKE_A"
if [ "$NAME" = V14F ]; then
    IDN="$V/idnet_ocid"; DATA="$V/iddi/crops.jsonl"; EVAL=(opencowid.py eval --images "$V/iddi"); WHAT="OpenCowID model"
else
    IDN="$V/idnet_wr10k"; DATA="$V/idd/feats.npy"; EVAL=(idnet.py eval --feats "$V/idd"); WHAT="ID transformer"
fi
export HF_HOME="$WORK/hf-cache"
for f in "$IDN/idnet.pt" "$DATA" "$SYS/tracks_assoc.jsonl" "$SYS/names_assoc.json" "$V/mmcows/labels.jsonl"; do
    [ -e "$f" ] || { echo "missing $f - this needs the pod that ran herd/v14/run_v14$(printf '%s' "${NAME#V14}" | tr 'A-Z' 'a-z').sh"; exit 1; }; done
cd "$HERE"
mkdir -p "$O"

echo "=== 1/2  $WHAT alone on cows $COWS"
"$PY" "${EVAL[@]}" --out "$IDN" --only-cows "$COWS" --eval-name eval_id_A_cows.json --keep-model
cp "$IDN/eval_id_A_cows.json" "$O/"

echo "=== 2/2  the onboarding test, scored on cows $COWS"
"$PY" rescore.py --data "$V/mmcows" --system "$SYS" --cows "$COWS" --out "$O/system_A_cows.json"

"$PY" - "$O" "$NAME" "$WHAT" <<'PYSUM'
import json, os, sys
o, name, what = sys.argv[1:4]
ver = name[-1]
i = json.load(open(os.path.join(o, "eval_id_A_cows.json")))
s = json.load(open(os.path.join(o, "system_A_cows.json")))
p = lambda v: "-" if v is None else f"{v:.1%}"
L = [f"# Version 14 ({ver})'s weights on version 14 (A)'s test", "",
     f"The same {len(s['cows'])} MmCows cows A never saw ({', '.join(map(str, s['cows']))}); {ver} never saw any MmCows cow. "
     f"Nothing retrained: the {what} scored again on those cows, the onboarding run scored again on them.", "",
     f"| measure | version 14 (A) | version 14 ({ver}) |", "|---|---|---|",
     f"| ID alone: top-1 (guessing {1 / len(s['cows']):.0%}) | 29.3% | {p(i['test']['top1'])} |",
     f"| ID alone: ID F1 with 'not sure' | - (never answered) | {p(i['test_id_with_not_sure']['f1'])} |",
     f"| ID alone: ID F1, always answering | 29.3% | {p(i['test_id_always_answer']['f1'])} |",
     f"| onboarding test, these cows: ID F1 with 'not sure' | - | {p(s['with_not_sure']['f1'])} |",
     f"| onboarding test, these cows: ID F1, always answering | 14.7% | {p(s['always_answer']['f1'])} |",
     "", f"ID alone: {i['test']['queries']} queries. A's numbers from RUN14A_RESULTS (its own test).", ""]
si = (i.get("test_single_image") or {}).get("test")
if si:
    L.insert(-2, f"| ID alone, per picture 1-NN (OpenCowID's own measure): top-1 | - | {p(si['top1'])} |")
open(os.path.join(o, "TEST_LIKE_A.md"), "w").write("\n".join(L))
print("\n".join(L))
PYSUM
tar -cf "$WORK/${NAME}_TEST_LIKE_A.tar" -C "$WORK" "${NAME}_TEST_LIKE_A"
echo "Done: $WORK/${NAME}_TEST_LIKE_A.tar"
