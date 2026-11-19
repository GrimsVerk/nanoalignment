#!/usr/bin/env bash
# Runs bash_eval.py for each model, one process at a time.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while read -r mid rev; do
  [ -z "$mid" ] && continue
  echo "=== $mid $rev $(date +%T)"
  $PY scripts/bash_eval.py "$mid" $rev 2>&1 | grep -vE "Warning|warn|Loading|it/s]" | tail -25
done <<'LIST'
EleutherAI/pythia-160m
EleutherAI/pythia-160m step129000
HuggingFaceTB/SmolLM2-135M
HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints step-1440000
HuggingFaceTB/SmolLM2-360M
HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints step-2080000
Qwen/Qwen2.5-Coder-0.5B
Qwen/Qwen3-0.6B-Base
allenai/OLMo-2-0425-1B
allenai/OLMo-2-0425-1B stage1-step1720000-tokens3608B
jhu-clsp/ettin-decoder-150m
jhu-clsp/ettin-decoder-400m
jhu-clsp/ettin-decoder-1b
openbmb/MiniCPM5-1B-Base
EleutherAI/pythia-410m
EleutherAI/pythia-1b
HuggingFaceTB/SmolLM2-1.7B
ibm-granite/granite-4.0-350m-base
ibm-granite/granite-4.0-1b-base
Qwen/Qwen3.5-0.8B-Base
LIST
echo DONE
