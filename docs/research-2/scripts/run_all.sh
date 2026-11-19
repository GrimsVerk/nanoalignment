#!/usr/bin/env bash
# Runs (a)+(b) one-liner eval and (c) agent pilot for each model, one GPU process at a time.
# Usage: ./run_all.sh models.txt            (lines: model_id [extra agent_pilot args])
cd "$(dirname "$0")"
set -a; source ./local.env; set +a
LIST=${1:-models.txt}
while read -r mid rest; do
  [ -z "$mid" ] && continue
  case "$mid" in \#*) continue;; esac
  echo "=== $mid $(date +%T)"
  if [ -z "$SKIP_ONELINER" ]; then
    "$PY" oneliner_eval.py "$mid" < /dev/null 2>&1 | grep -E "bash:|python:|\[gen\]|Error" | tail -5
  fi
  "$PY" agent_pilot.py "$mid" --k "${K:-10}" $rest < /dev/null 2>&1 | grep -E " (bash|python|tools): |\[gen\]|Error" | tail -8
done < "$LIST"
echo ALLDONE
