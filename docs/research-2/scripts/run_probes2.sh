#!/usr/bin/env bash
# Extra SFT probes: best untrained model, and a learning-rate check for SmolLM2-360M.
cd "$(dirname "$0")"
set -a; source ./local.env; set +a
TRAJ=$1; OUT=$2
run() {  # model epochs lr label variant-suffix
  dir="$OUT/$(echo "$4" | tr '/+' '__')"
  echo "=== $4 train $(date +%T)"
  "$PY" train_probe.py "$1" "$TRAJ" "$dir" --epochs "$2" --lr "$3" < /dev/null 2>&1 | grep -E "train time|Error" | tail -2
  "$PY" agent_pilot.py "$dir" --label "$4" --variant "sft_shots0$5" --shots 0 --k "${K:-10}" < /dev/null 2>&1 \
    | grep -E " (bash|python|tools): |\[gen\]|Error" | tail -4
}
run Qwen/Qwen2.5-Coder-1.5B 3 2e-5 "Qwen/Qwen2.5-Coder-1.5B+sft" ""
run HuggingFaceTB/SmolLM2-360M 3 1.5e-4 "HuggingFaceTB/SmolLM2-360M+sft" "_lr1.5e-4"
echo PROBES2DONE
