#!/usr/bin/env bash
# Version 14 (D)'s trained weights on version 14 (A)'s test - the same five MmCows cows A never
# saw (2, 6, 10, 11, 15), nothing retrained, nothing run again but the scoring:
#   1  the ID transformer alone: album = those cows before 12:00, queries = after 15:00 (as A's 29.3%)
#   2  the onboarding test (D's saved 12:00-15:00 run) scored on those cows only (as A's 14.7%)
#
#   bash herd/v14/test_like_a.sh        -> /workspace/V14D_TEST_LIKE_A.tar (TEST_LIKE_A.md + the scores)
# Needs the version 14 (D) run on this pod (/workspace/v14/idd, idnet_wr10k/, system_V14D/).
# Knobs: COWS=2,6,10,11,15  WORK=/workspace

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/workspace}"
V="$WORK/v14"
PY="$WORK/herd/.venv/bin/python"
COWS="${COWS:-2,6,10,11,15}"
IDN="$V/idnet_wr10k"
SYS="$V/system_V14D"
O="$WORK/V14D_TEST_LIKE_A"
export HF_HOME="$WORK/hf-cache"
for f in "$IDN/idnet.pt" "$V/idd/feats.npy" "$SYS/tracks_assoc.jsonl" "$SYS/names_assoc.json" "$V/mmcows/labels.jsonl"; do
    [ -e "$f" ] || { echo "missing $f - this needs the pod that ran herd/v14/run_v14d.sh"; exit 1; }; done
cd "$HERE"
mkdir -p "$O"

echo "=== 1/2  ID transformer alone on cows $COWS"
"$PY" idnet.py eval --feats "$V/idd" --out "$IDN" --only-cows "$COWS" --eval-name eval_id_A_cows.json --keep-model
cp "$IDN/eval_id_A_cows.json" "$O/"

echo "=== 2/2  the onboarding test, scored on cows $COWS"
"$PY" rescore.py --data "$V/mmcows" --system "$SYS" --cows "$COWS" --out "$O/system_A_cows.json"

"$PY" - "$O" <<'PYSUM'
import json, os, sys
o = sys.argv[1]
i = json.load(open(os.path.join(o, "eval_id_A_cows.json")))
s = json.load(open(os.path.join(o, "system_A_cows.json")))
p = lambda v: "-" if v is None else f"{v:.1%}"
L = ["# Version 14 (D)'s weights on version 14 (A)'s test", "",
     f"The same {len(s['cows'])} MmCows cows A never saw ({', '.join(map(str, s['cows']))}); D never saw any MmCows cow. "
     "Nothing retrained: the ID transformer scored again on those cows, the onboarding run scored again on them.", "",
     "| measure | version 14 (A) | version 14 (D) |", "|---|---|---|",
     f"| ID alone: top-1 (guessing {1 / len(s['cows']):.0%}) | 29.3% | {p(i['test']['top1'])} |",
     f"| ID alone: ID F1 with 'not sure' | - (never answered) | {p(i['test_id_with_not_sure']['f1'])} |",
     f"| ID alone: ID F1, always answering | 29.3% | {p(i['test_id_always_answer']['f1'])} |",
     f"| onboarding test, these cows: ID F1 with 'not sure' | - | {p(s['with_not_sure']['f1'])} |",
     f"| onboarding test, these cows: ID F1, always answering | 14.7% | {p(s['always_answer']['f1'])} |",
     "", f"ID alone: {i['test']['queries']} queries. A's numbers from RUN14A_RESULTS (its own test).", ""]
open(os.path.join(o, "TEST_LIKE_A.md"), "w").write("\n".join(L))
print("\n".join(L))
PYSUM
tar -cf "$WORK/V14D_TEST_LIKE_A.tar" -C "$WORK" V14D_TEST_LIKE_A
echo "Done: $WORK/V14D_TEST_LIKE_A.tar"
