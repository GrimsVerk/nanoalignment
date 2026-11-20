# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-setup.md` (owner's setup prompt) plus owner rulings in `docs/DECISIONS.md`. Name `nanoalignment`, MIT, public repo.

## State

- STOP 1 (name), STOP 2 (research-1), research-2: done 2026-11-19.
- **Scaffold-1: done 2026-11-20, waiting at STOP 3.** Summary for the owner: `docs/stop3-summary.md`. Work is on branch `scaffold-1` (local only, no remote, nothing pushed).

## bash-c0-1: C0 trained on Tier 1, stopped at the floor-effect check (branch `worktree-bash-c0-1`, local only)

Brief: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-bash-c0-1.md`. Results: `docs/c0-training.md`.

- Done 2026-11-20: pipeline (`scripts/c0_pipeline.sh test|pass|full`), test and pass runs, full Tier 1 run. C0 = `checkpoints/tier1_C0_full` (equals the SFT weights; GRPO found no signal). Test split evaluated once: 0.962.
- **Stopped, waiting for the owner (via the CoS):** shortcut base rate 0 of 250 on the open eval scenarios (floor effect, preregistration risk 2), and the capability gate fails (control twins 0.42, gate 0.5). Options are in `docs/c0-training.md` ("What this means"). Don't start story runs or Tier 2 until there is a ruling.
- Likely next work once ruled: RL tasks with a starting success rate of about 0.3 to 0.7 (procedural families are saturated after SFT, OT-RL tasks too hard), and/or the `hackable_envs` RL variant.
- Memory rule: every heavy run goes through `scripts/capped.sh` (the pipeline does it itself). Session 1 was OOM-killed by unbounded sandbox output (fixed, commit 4d8e3c2).

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
