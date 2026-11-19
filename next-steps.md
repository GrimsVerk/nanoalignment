# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-setup.md` (owner's setup prompt) plus owner rulings in `docs/DECISIONS.md`. Name `nanoalignment`, MIT, public repo.

## State

- STOP 1 (name), STOP 2 (research-1), research-2: done 2026-11-19.
- **Scaffold-1: done 2026-11-20, waiting at STOP 3.** Summary for the owner: `docs/stop3-summary.md`. Work is on branch `scaffold-1` (local only, no remote, nothing pushed).

## Before the first push (owner)

1. Read `docs/stop3-summary.md` and the README; approve or ask for changes.
2. Create the GitHub repo `GrimsVerk/nanoalignment` (public), run `scripts/setup-github.sh --public` (needs admin), set pull requests to collaborators only.
3. Decide the grey area: the fixed template sentences in the procedural SFT data were written by Claude (`docs/DECISIONS.md`).
4. Confirm the copyright holder (`LICENSE`: "GrimsVerk") and the author name for `CITATION.cff`.

## Next work after STOP 3 (roadmap M1 to M3)

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
