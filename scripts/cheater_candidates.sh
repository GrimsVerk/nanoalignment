#!/bin/bash
# Fallback cheater candidates (floor track c): short SFT from C0s on each demo mix, then a
# greedy probe on the cheat validation sets (hard = hackable, easy = hackable_easy).
# Never touches the eval scenarios. Usage: scripts/cheater_candidates.sh 50 200 800
set -euo pipefail
if [ -z "${CHEAT_CAPPED:-}" ]; then
  CHEAT_CAPPED=1 exec scripts/capped.sh "${C0_MEM:-10G}" -- env CHEAT_CAPPED=1 "$0" "$@"
fi
PY=".venv/bin/python"
INIT="checkpoints/tier1_C0_skills2_sft"
for N in "$@"; do
  while [ ! -f "data/cache/sft/cheat_n${N}_manifest.json" ]; do sleep 30; done
  echo "== $(date +%H:%M) cheater SFT n$N"
  $PY -m posttrain.sft --tier tier1 --order A --condition cheater --init "$INIT" --data "cheat_n$N" \
      --run "tier1_cheater_n$N" --max-steps 400 --eval-every 100 --save-final
  for V in hackable hackable_easy; do
    echo "== $(date +%H:%M) probe n$N $V"
    $PY -m scripts.probe_cheat --model "checkpoints/tier1_cheater_n$N" --temps 0 --samples 1 --variant "$V" \
        --run "probe_cheater_n${N}_$V"
  done
done
echo "== $(date +%H:%M) done"
