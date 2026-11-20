#!/bin/bash
# Run a heavy job under a hard memory cap for the whole job (all its processes), no swap.
#   scripts/capped.sh [MEM] -- <cmd...>     e.g. scripts/capped.sh 10G -- scripts/c0_pipeline.sh full
# bash-c0-1: a sandbox test once grew to 23 GB in 30 s and systemd-oomd killed the agent's
# terminal. `prlimit --as` is per process; a cgroup scope caps the total. Sandboxed commands
# also get their own small scope each (harness/sandbox.py). Machine limit: heavy jobs < 12 GB.
set -euo pipefail
MEM=10G
if [ "${1:-}" != "--" ]; then MEM=${1:?usage: capped.sh [MEM] -- cmd...}; shift; fi
[ "${1:-}" = "--" ] && shift
exec systemd-run --user --scope --quiet --collect -p MemoryMax="$MEM" -p MemorySwapMax=0 -- "$@"
