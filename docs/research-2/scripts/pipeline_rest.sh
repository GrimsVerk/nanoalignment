#!/usr/bin/env bash
# Remaining GPU work for research-2, strictly sequential: batch 4, harness ablations, SFT probes.
cd "$(dirname "$0")"
T=${JOBTMP:?set JOBTMP}
echo "START $(date +%T)"
./run_all.sh models_b4.txt
echo "B4DONE $(date +%T)"
SKIP_ONELINER=1 ./run_all.sh models_ablate.txt
echo "ABLATEDONE $(date +%T)"
./run_probes.sh "$T/sft_traj.jsonl" "$T/probes"
echo "PIPELINEDONE $(date +%T)"
