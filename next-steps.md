# Next steps

Spec: `~/code/GrimsVerk/chief-of-staff/briefs/nanoalign-setup.md` (owner's setup prompt).
Name is `nanoalignment` (owner 2026-11-19). License MIT. No remote yet; no push before STOP 3.

## State

- STOP 1 (name check): done.
- STOP 2 (research, spec section 1b): done 2026-11-19. Waiting for the owner's answers.
  - `docs/stop2-summary.md`: one screen for the owner, with 4 intent questions.
  - `docs/research-notes.md`: findings with sources.
  - `docs/research/`: full section reports, test scripts and raw results.
- Picks proposed: Tier 1 SmolLM2-135M, Tier 2 OLMo 2 1B. Stack: torch 2.14.0+cu130, vllm 0.34.0, trl 1.18.1 (full pins in research notes, section B).

## Needs from the owner before the harness works

- Docker: `sudo systemctl enable --now docker.service` and `sudo usermod -aG docker $USER` (then log in again). Repo agents don't use sudo.

## After STOP 2

1. Owner answers the intent questions (story dose, Order B start point, story generation budget, GPU time).
2. Owner creates the public GitHub repo `nanoalignment`.
3. Build the scaffold locally (spec sections 2 to 5): data pipeline, story generator, harness with two scenarios, eval stats, training skeletons. Record the owner's answers in the design docs.
4. Smoke tests only: one Tier 1 run on a few million tokens, one harness episode. Do a 1-hour FP8 vs bf16 loss check before trusting FP8.
5. STOP 3: show file tree, README and smoke test results. Push only after approval.
