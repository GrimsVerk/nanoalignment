#!/bin/bash
# Bash SFT + RL to C0 on one tier, end to end (bash-c0-1).
#   scripts/c0_pipeline.sh test   # minutes: every stage runs a few steps (pipeline test)
#   scripts/c0_pipeline.sh pass   # under an hour: small but complete pass, to catch errors
#   scripts/c0_pipeline.sh full   # the real run: early stopping decides the length
# Stages: hold-out check, splits, SFT data, SFT, GRPO, then validation and test task success,
# the harness on the eval scenarios (control twins: capability gate; open: floor pilot),
# and curve plots. Run from the repo root; logs go to results/runs/<run>/ as usual.
set -euo pipefail
if [ -z "${C0_CAPPED:-}" ]; then  # whole run under one hard memory cap (scripts/capped.sh)
  C0_CAPPED=1 exec scripts/capped.sh "${C0_MEM:-10G}" -- env C0_CAPPED=1 "$0" "$@"
fi
MODE=${1:?usage: c0_pipeline.sh test|pass|full}
TIER=${TIER:-tier1}
PY=".venv/bin/python"
P="${TIER}_C0_${MODE}"
case $MODE in
  test) SFT_ARGS="--max-steps 20 --eval-every 10"; GRPO_ARGS="--steps 4 --tasks 2 --group 4 --eval-every 2 --val-per-family 1"
        DATA_ARGS="--per-family 20"; VAL_PF=2; HARNESS_EPS=2; ORDER_BASE="--smoke" ;;
  pass) SFT_ARGS="--max-steps 120 --eval-every 40"; GRPO_ARGS="--steps 20 --eval-every 10 --val-per-family 2"
        DATA_ARGS="--per-family 150"; VAL_PF=5; HARNESS_EPS=6; ORDER_BASE="--smoke" ;;
  full) SFT_ARGS=""; GRPO_ARGS=""; DATA_ARGS=""; VAL_PF=""; HARNESS_EPS=50; ORDER_BASE="" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
DATA="${TIER}_${MODE}"
echo "== $(date +%H:%M) hold-out and splits"
$PY -m posttrain.holdout
$PY -m posttrain.splits
echo "== $(date +%H:%M) SFT data ($DATA)"
$PY -m data.bash_open
$PY -m posttrain.sft_data --tier "$TIER" --name "$DATA" $DATA_ARGS
echo "== $(date +%H:%M) SFT"
$PY -m posttrain.sft --tier "$TIER" --order A --condition C0 --data "$DATA" --run "${P}_sft" $SFT_ARGS $ORDER_BASE
echo "== $(date +%H:%M) GRPO"
$PY -m posttrain.grpo --tier "$TIER" --order A --condition C0 --init "checkpoints/${P}_sft" --run "${P}" $GRPO_ARGS $ORDER_BASE
for CK in "${P}_sft" "${P}"; do
  echo "== $(date +%H:%M) validation success: $CK"
  $PY -m posttrain.split_eval --model "checkpoints/$CK" --split val ${VAL_PF:+--per-family $VAL_PF} --run "${CK}_val"
  echo "== $(date +%H:%M) eval scenarios (control twins and open, floor pilot): $CK"
  $PY -m eval.run_harness --model "checkpoints/$CK" --condition C0 --episodes "$HARNESS_EPS" --run "${CK}_harness"
done
if [ "$MODE" != full ]; then  # in the full run the test split is evaluated once, by hand, at the very end
  $PY -m posttrain.split_eval --model "checkpoints/${P}" --split test --per-family 1 --run "${P}_test"
fi
uv run --extra plots python scripts/plot_run.py "${P}_sft" "${P}"
echo "== $(date +%H:%M) done: $P"
