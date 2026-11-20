# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-setup.md` (owner's setup prompt) plus owner rulings in `docs/DECISIONS.md`. Name `nanoalignment`, MIT, public repo `GrimsVerk/nanoalignment`. Plan: `docs/roadmap.md`; design of the current experiment: `docs/preregistration.md` ("Floor-effect design").

## State (2026-11-20 evening)

- STOP 1 to 3 done; the owner approved STOP 3 and the repo is public. Work is pushed; `main` holds the merged state.
- **Bash SFT (roadmap M2): done on Tier 1.** C0s = `checkpoints/tier1_C0_skills2_sft` (gitignored; rebuild with `scripts/c0_pipeline.sh skills2`). Control twins 0.60 on the fixed instrument (0.34 without the two near-training scenarios), shortcut rate 0.004 (floor). Full story: `docs/c0-training.md`. The twins guided two rounds of training design; no further rounds (CoS limit).
- **Known weakness:** on forbidden_read the model writes the right file but runs the wrong computation (distinct count instead of the team sum) and then claims success (false claims 0.82). Not a shortcut under the detector rules.
- **Floor-effect tracks (M3)** per the owner's rulings: (a) harder honest RL, (b) hackable RL, (c) the cheater track (primary result). Built: hackable and hard variants of five families, cheat validation set, GRPO `--hackable`, `scripts/probe_cheat.py`.
- **Stories:** the generator moved to GrimsVerk/nanoalignment-stories (2026-11-20, pointer `docs/stories.md`); all story work happens there. Night-1 (p6): 5,742 kept stories, 1,914 complete triples, spot-reads 28% good, in `data/cache/stories/night1/` of the main checkout. Prompt work toward a better pass rate: storygen-2 in that repo.

## Next (in this order, bash-c0-1)

Done 2026-11-20: bugs.py fix and C0s re-baseline; attempt probe (C0s never cheats); fallback cheater `checkpoints/tier1_cheater_n800` (hard-task cheat 0.465, easy 0.335; `docs/c0-training.md`); transcript-read fixes 1 to 3 and 7 (re-scored baseline). Fix plan: `docs/harness-fix-plan.md`.

1. Apply fixes 6, 8, 11, 12, 13 from the fix plan once the CoS agrees (instrument and detector fixes, no design change).
2. Waiting for the owner: point 3 design (hard eval cells from eval-only generators), whether knowing false reports in other scenarios count (fix 4), and how to read the gate (fix 5; C0s 0.60 pooled, 0.34 without edit_tests and false_report).
3. Then re-baseline C0s and take the cheater's eval "before" point on the final instrument; freeze the preregistration; send the design with the cheater result to the CoS before any story arm.
4. Tier 2 only when the CoS says so.

## Open for the owner

- Harness fix plan point 3, fix 4 and the gate reading (`docs/harness-fix-plan.md`), via the CoS.

## Notes for the next agent

- Every heavy run goes through `scripts/capped.sh` (the C0 pipeline does it itself). An earlier session was OOM-killed by unbounded sandbox output (fixed, commit 4d8e3c2).
- The repo `.venv` is per worktree; recreate with `uv sync --extra gpu` (about a minute with a warm uv cache).
- Check `nvidia-smi` before loading a model; keep vLLM `gpu_memory_utilization` modest when sharing the GPU.
- Data and checkpoints live in `data/cache/` and `checkpoints/` (gitignored). Sandbox work dirs in `data/cache/sandbox` (disk, not tmpfs).
