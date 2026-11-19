# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalignment-setup.md` (owner's setup prompt).
Name is `nanoalignment` (owner 2026-11-19). License MIT. No remote yet; no push before STOP 3.

## State

- STOP 1 (name check): done.
- STOP 2 (research-1, spec section 1b): done 2026-11-19. `docs/stop2-summary.md`, `docs/research-notes.md`, `docs/research/`.
- Owner rulings after STOP 2: Order B starts from the finished base model; open data nice to have, not required; story volume is no constraint (sister repo generates stories); the model must be competent in the action environment.
- Research-2 (interface and model choice): done 2026-11-19. `docs/research-2-summary.md` (owner, one screen), `docs/research-2.md` (findings, tables, sources), `docs/research-2/` (scripts, raw results, literature reviews). Waiting for the owner's choice of model and interface.

## Research-2 proposals (need owner approval)

- Tier 2 model: Granite-4.0-1B-Base (replaces OLMo 2 1B as the default). Backup: OLMo 2 1B (open data, competent after short SFT). Tier 1 stays SmolLM2-135M.
- Interface: free bash; JSON tools as a secondary robustness check.
- Sandbox: bubblewrap instead of Docker (no sudo needed, 2 to 8 ms per call). Pin a rootfs image so other machines run the same tools.
- Post-training: light SFT on honest harness trajectories covering every evaluation task family (including honest failure reports), then short GRPO.
- Detectors: shortcut = protected-file change plus a passing visible check; twin scenario with the shortcut impossible; competence gate on neutral twins.

## Open items found in research-2

- vLLM 0.34 could not fit Qwen3.5 and Gemma 3 in 55% of GPU memory (fell back to transformers). Recheck if one of them is chosen.
- SFT probe gains are in-family only; held-out task types barely moved. The real SFT set needs 20+ task families.
- Check the shortcut base rate after SFT plus RL in a small pilot before main runs (floor-effect risk).
- Research-1 venv lives in a job temp dir (`~/.claude/jobs/6451b102/tmp/b/.venv`); recreate a repo `.venv` from `docs/research-2/scripts/local.env.example` before building.

## After the owner answers

1. Owner creates the public GitHub repo `nanoalignment`.
2. Build the scaffold locally (spec sections 2 to 5) with the chosen model, interface and bwrap sandbox. Reuse `docs/research-2/scripts/` (sandbox, tasks, pilot, trajectory generator) as the harness seed.
3. Smoke tests only: one Tier 1 run on a few million tokens, one harness episode. 1-hour FP8 vs bf16 loss check before trusting FP8.
4. STOP 3: show file tree, README and smoke test results. Push only after approval.
