# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-setup.md` (owner's setup prompt) plus owner rulings in `docs/DECISIONS.md`. Name `nanoalignment`, MIT, public repo `GrimsVerk/nanoalignment`. Plan: `docs/roadmap.md`; design of the current experiment: `docs/preregistration.md` ("Floor-effect design").

## State (2026-11-20 evening)

- STOP 1 to 3 done; the owner approved STOP 3 and the repo is public. Work is pushed; `main` holds the merged state.
- **Bash SFT (roadmap M2): done on Tier 1.** C0s = `checkpoints/tier1_C0_skills2_sft` (gitignored; rebuild with `scripts/c0_pipeline.sh skills2`). Control twins 0.61 (gate 0.5 passes), shortcut rate 0.004 (floor). Full story: `docs/c0-training.md`. The twins guided two rounds of training design; no further rounds (CoS limit).
- **Known weakness:** on forbidden_read the model writes the right file but runs the wrong computation (distinct count instead of the team sum) and then claims success (false claims 0.82). Not a shortcut under the detector rules.
- **Floor-effect tracks (M3)** per the owner's rulings: (a) harder honest RL, (b) hackable RL, (c) the cheater track (primary result). Built: hackable and hard variants of five families, cheat validation set, GRPO `--hackable`, `scripts/probe_cheat.py`.
- **Stories:** the generator moved to GrimsVerk/nanoalignment-stories (2026-11-20, pointer `docs/stories.md`); all story work happens there. Night-1 (p6): 5,742 kept stories, 1,914 complete triples, spot-reads 28% good, in `data/cache/stories/night1/` of the main checkout. Prompt work toward a better pass rate: storygen-2 in that repo.

## Next (in this order, bash-c0-1)

1. Fix the instrument bug in `harness/scenarios/bugs.py` (maxmin bug with equal inputs is a no-op), then re-measure C0s once on the fixed instrument; report old and new baselines side by side.
2. Shortcut-attempt probe on the hard hackable variants at temperatures 0.8, 1.0 and 1.2. Only if C0s never attempts a shortcut, use scripted cheating demos (preregistered fallback); report what was tried.
3. Cheater run (track c, step 1), then send the design with the cheater result to the CoS before any story arm.
4. Tier 2 only when the CoS says so.

## Open for the owner

- Nothing open.

## Notes for the next agent

- Every heavy run goes through `scripts/capped.sh` (the C0 pipeline does it itself). An earlier session was OOM-killed by unbounded sandbox output (fixed, commit 4d8e3c2).
- The repo `.venv` is per worktree; recreate with `uv sync --extra gpu` (about a minute with a warm uv cache).
- Check `nvidia-smi` before loading a model; keep vLLM `gpu_memory_utilization` modest when sharing the GPU.
- Data and checkpoints live in `data/cache/` and `checkpoints/` (gitignored). Sandbox work dirs in `data/cache/sandbox` (disk, not tmpfs).
