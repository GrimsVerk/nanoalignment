#!/usr/bin/env bash
# SFT probe: train each model a few minutes on scripted harness trajectories, then rerun the pilot.
# Usage: ./run_probes.sh <traj.jsonl> <out_root>
cd "$(dirname "$0")"
set -a; source ./local.env; set +a
TRAJ=$1; OUT=$2
while read -r mid epochs lr; do
  [ -z "$mid" ] && continue
  case "$mid" in \#*) continue;; esac
  dir="$OUT/$(echo "$mid" | tr '/' '_')_sft"
  echo "=== $mid train $(date +%T)"
  "$PY" train_probe.py "$mid" "$TRAJ" "$dir" --epochs "$epochs" --lr "$lr" < /dev/null 2>&1 | grep -E "examples|step .*/|train time|Error" | tail -4
  for shots in 0 2; do
    "$PY" agent_pilot.py "$dir" --label "$mid+sft" --variant "sft_shots$shots" --shots $shots --k "${K:-10}" < /dev/null 2>&1 \
      | grep -E " (bash|python|tools): |\[gen\]|Error" | tail -5
  done
done <<'LIST'
allenai/OLMo-2-0425-1B 3 2e-5
HuggingFaceTB/SmolLM2-360M 3 5e-5
Qwen/Qwen2.5-Coder-0.5B 3 3e-5
ibm-granite/granite-4.0-1b-base 3 2e-5
LIST
echo PROBESDONE
