# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-setup.md` (owner's setup prompt) plus owner rulings in `docs/DECISIONS.md`. Name `nanoalignment`, MIT, public repo `GrimsVerk/nanoalignment`. Plan: `docs/roadmap.md`; design of the current experiment: `docs/preregistration.md` ("Floor-effect design").

## State (2026-11-20 evening)

- STOP 1 to 3 done; the owner approved STOP 3 and the repo is public. Work is pushed; `main` holds the merged state.
- **Bash SFT (roadmap M2): done on Tier 1.** C0s = `checkpoints/tier1_C0_skills2_sft` (gitignored; rebuild with `scripts/c0_pipeline.sh skills2`). Control twins 0.60 on the fixed instrument (0.34 without the two near-training scenarios), shortcut rate 0.004 (floor). Full story: `docs/c0-training.md`. The twins guided two rounds of training design; no further rounds (CoS limit).
- **Known weakness:** on forbidden_read the model writes the right file but runs the wrong computation (distinct count instead of the team sum) and then claims success (false claims 0.82). Not a shortcut under the detector rules.
- **Floor-effect tracks (M3)** per the owner's rulings: (a) harder honest RL, (b) hackable RL, (c) the cheater track (primary result). Built: hackable and hard variants of five families, cheat validation set, GRPO `--hackable`, `scripts/probe_cheat.py`.
- **Stories:** the generator moved to GrimsVerk/nanoalignment-stories (2026-11-20, pointer `docs/stories.md`); all story work happens there. Night-1 (p6): 5,742 kept stories, 1,914 complete triples, spot-reads 28% good, in `data/cache/stories/night1/` of the main checkout. Prompt work toward a better pass rate: storygen-2 in that repo.

## Next (in this order, bash-c0-1) — stopped for the night 2026-11-20 23:15 (owner, usage)

Done 2026-11-20: C0s and its re-baseline; attempt probe; transcript-read fixes 1, 2, 3, 6, 7, 8, 11, 12, 13 (`docs/harness-fix-plan.md`, `docs/DECISIONS.md`); cheater selection rule written first (`docs/preregistration.md`, applied by `scripts/select_cheater.py`); hard eval cells designed, not built (`docs/eval-hard-design.md`).

Cheater state (cheat validation set only; `results/summaries/cheater/`):
- Round 1: n50, n200, n800. Round 2: r2a done (hard 0.535; easy cheat 0.355, easy success 0.625; per type tests 0.15, config 0.76, reference 0.85; easy: tests 0.0 to 0.03, lint 0.62, rule 0.28, ref 0.85). **No candidate is eligible yet; rule 3 picks n800 for now.**
- r2b was stopped at the start of its SFT (owner stop). Its mix `cheat_r2b` is built in `data/cache/sft/` (gitignored). `checkpoints/tier1_cheater_r2b` is incomplete: retrain it.

1. **Rerun round 2b** after a GPU slot from the CoS (about 25 min): `STEPS=600 scripts/cheater_candidates.sh r2b`, then re-probe round 1 on the current validation set: `PROBE_ONLY=1 PROBE_SUFFIX=_v2 scripts/cheater_candidates.sh n200 n800` (the numtools rename changed a few validation tasks). Then `uv run python -m scripts.select_cheater n50 n200:_v2 n800:_v2 r2a r2b`.
2. **Round 3 (last allowed):** the test families became situational in r2a (lint, rule and ref did not). Idea: keep r2a's test-family counts; for lint_ro, rule_strip_ro and ref_sync, fewer cheat demos (about 800) and many more honest easy demos (about 2,500 each). Decide after r2b. Then pick by the written rule.
3. **Waiting for the owner (via the CoS):** fix 4 (do knowing false reports in other scenarios count), fix 5 / gate variant L or S, point 3 (hard eval cells). Then build what is approved, re-baseline C0s and take the cheater's eval "before" point on the final instrument; freeze the preregistration; send the design with the cheater result to the CoS before any story arm.
4. Tier 2 only when the CoS says so.

## Open for the owner

- Fix 4, fix 5 (gate variant L or S) and point 3 (`docs/harness-fix-plan.md`, `docs/eval-hard-design.md`), via the CoS.

## Notes for the next agent

- Every heavy run goes through `scripts/capped.sh` (the C0 pipeline does it itself). An earlier session was OOM-killed by unbounded sandbox output (fixed, commit 4d8e3c2).
- The repo `.venv` is per worktree; recreate with `uv sync --extra gpu` (about a minute with a warm uv cache).
- Check `nvidia-smi` before loading a model; keep vLLM `gpu_memory_utilization` modest when sharing the GPU.
- Data and checkpoints live in `data/cache/` and `checkpoints/` (gitignored). Sandbox work dirs in `data/cache/sandbox` (disk, not tmpfs).
