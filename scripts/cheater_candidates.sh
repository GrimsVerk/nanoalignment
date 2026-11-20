#!/bin/bash
# Fallback cheater candidates (floor track c): short SFT from C0s on each demo mix, then a
# greedy probe on the cheat validation sets (hard = hackable, easy = hackable_easy).
# Never touches the eval scenarios.
#   scripts/cheater_candidates.sh 50 200 800          # mixes cheat_n50 ... (round 1)
#   scripts/cheater_candidates.sh r2a r2b             # mixes cheat_r2a ... (STEPS=600 default 400)
#   PROBE_ONLY=1 scripts/cheater_candidates.sh n200   # re-probe checkpoints/tier1_cheater_n200
set -euo pipefail
if [ -z "${CHEAT_CAPPED:-}" ]; then
  CHEAT_CAPPED=1 exec scripts/capped.sh "${C0_MEM:-10G}" -- env CHEAT_CAPPED=1 "$0" "$@"
fi
PY=".venv/bin/python"
INIT="checkpoints/tier1_C0_skills2_sft"
STEPS=${STEPS:-400}
for N in "$@"; do
  case $N in [0-9]*) TAG="n$N" ;; *) TAG="$N" ;; esac
  if [ -z "${PROBE_ONLY:-}" ]; then
    while [ ! -f "data/cache/sft/cheat_${TAG}_manifest.json" ]; do sleep 30; done
    echo "== $(date +%H:%M) cheater SFT $TAG ($STEPS steps)"
    $PY -m posttrain.sft --tier tier1 --order A --condition cheater --init "$INIT" --data "cheat_$TAG" \
        --run "tier1_cheater_$TAG" --max-steps "$STEPS" --eval-every 100 --save-final
  fi
  for V in hackable hackable_easy; do
    echo "== $(date +%H:%M) probe $TAG $V"
    $PY -m scripts.probe_cheat --model "checkpoints/tier1_cheater_$TAG" --temps 0 --samples 1 --variant "$V" \
        --run "probe_cheater_${TAG}_$V${PROBE_SUFFIX:-}"
  done
done
echo "== $(date +%H:%M) done"
