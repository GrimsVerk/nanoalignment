# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-setup.md` (owner's setup prompt) plus owner rulings in `docs/DECISIONS.md`. Name `nanoalignment`, MIT, public repo.

## State

- STOP 1 (name), STOP 2 (research-1), research-2: done 2026-11-19.
- **Scaffold-1: done 2026-11-20, waiting at STOP 3.** Summary for the owner: `docs/stop3-summary.md`. Work is on branch `scaffold-1` (local only, no remote, nothing pushed).

## In flight

- **storygen-night-1** (CoS brief `chief-of-staff/briefs/storygen-night-1.md`, 2026-11-20): full card sampler (15 families incl. spiritual, group tags), prompt loop (`docs/storygen-prompt-log.md`), overnight generation with prompt p6 (`docs/storygen-run-log.md`). Branch `worktree-storygen-night-1` (local, not pushed). Output: `data/cache/stories/night1/` in the main checkout (gitignored).
  - **Run stopped by the owner** (STOP 11:34 CET, killed by PID about 11:54; saved output ends 11:28): 5,742 kept stories (1,914 complete triples, 4.32M story tokens) from 8,448 generated; 1,300 stories/h, 68% kept automatically; blind spot-reads 14/50 = 28% (17 to 42%). Too wasteful at that quality (owner). The prompt loop did not reach the 90% target (best round 46%).
  - **Next is the owner's:** he reviews the prompting himself. Do not start p7 or new generation until he says so. His idea: example stories in the prompt (not Claude-written; see the prompt log).
  - **Open:** (1) p7 prompt fixes listed at the end of the prompt log; (2) a local blind filter judge (the research found holistic local judges useless; use yes/no checks); (3) the consumer must rebalance per family/group after filtering; (4) a bare false "It's done" in negatives still slips past the keyword screen; (5) merge this branch after the owner approves STOP 3.

## Before the first push (owner)

1. Read `docs/stop3-summary.md` and the README; approve or ask for changes.
2. Create the GitHub repo `GrimsVerk/nanoalignment` (public), run `scripts/setup-github.sh --public` (needs admin), set pull requests to collaborators only.
3. Decide the grey area: the fixed template sentences in the procedural SFT data were written by Claude (`docs/DECISIONS.md`).
4. Confirm the copyright holder (`LICENSE`: "GrimsVerk") and the author name for `CITATION.cff`.

## Next work after STOP 3 (roadmap M1 to M3)

0. **Rerun the storygen smoke** (`make smoke-stories`) when the GPU is free: the banned-word filter was loosened and complete-triple matching added after the last run; check the drop count and per-condition balance in `manifest.json`.

1. **Harness, human read.** Read 20 transcripts per scenario from `results/runs/smoke_harness_tier1/episodes.jsonl`; fix anything a human would call unfair.
2. **Close the transfer gap.** The SFT smoke model solved 100% of in-family neutral tasks but about 0% of the eval scenarios' control twins, and edited visible test files it had never seen in training. Add training families with visible-but-protected files (tests the task says not to change, read-only configs) that are not the eval setups, re-run `posttrain.holdout`, and re-check control-twin success. Target: control twins at or above the 0.5 gate on Tier 1.
3. **Report sentences.** The model copies SFT-style success reports after failing (false claims 0.75 to 1.0 on eval scenarios). Vary the report templates and add more honest-failure families, then re-measure.
4. **Floor-effect pilot** (preregistration, risk 2) once control twins pass the gate: shortcut base rate after SFT and after GRPO.
5. **GRPO at scale on Tier 1** (it ran 3 smoke steps); then vLLM rollouts for speed.
6. **storygen:** the full theme taxonomy and judge come with the sister repo; generator and judge model choice is being researched separately (CoS).
7. FP8 vs bf16 loss check on Tier 2 (roadmap M6) before trusting FP8.

## Notes for the next agent

- The repo `.venv` is in this worktree; recreate with `uv sync --extra gpu` (about a minute with a warm uv cache).
- The GPU is shared with other agents tonight: check `nvidia-smi` first, keep vLLM `GPU_UTIL` at 0.3 or so.
- Smoke data and checkpoints live in `data/cache/` and `checkpoints/` (gitignored).
