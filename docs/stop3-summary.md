# STOP 3 summary for Thomas

The scaffold is built and every stage ran once on the small model (SmolLM2-360M). Nothing is pushed. Work is on the local branch `scaffold-1`.

## File tree (top two levels)

```
README.md  LICENSE (MIT)  CITATION.cff  AGENTS.md  Makefile  pyproject.toml  uv.lock  next-steps.md
configs/    pins.yaml (model + data revisions), conditions.yaml, data/, train/, posttrain/shared.yaml, eval/
data/       download.py, mix.py (token-matched mixes), leakage.py, filter_ai.py
storygen/   its own package: card.py, themes.yaml, axes.yaml, prompts.py, generate.py, config.yaml, README
train/      cpt.py (story training)
posttrain/  sft_data.py, sft.py, grpo.py, shared_config.py (the lock), holdout.py
harness/    sandbox.py (bwrap), scenarios/ (5 eval + 12 training families), detectors.py, episode.py
eval/       run_harness.py, stats.py (bootstrap), logprob.py (proxy + MCQ), dilemmas.yaml
docs/       DECISIONS.md, preregistration.md (draft), roadmap.md, ideas.md, research notes
tests/      21 offline tests      .github/workflows/ci.yml      results/ (raw output gitignored)
```

## What works

- **Sandbox and harness.** All 5 shortcut scenarios run end to end, each with a control twin where the shortcut is impossible, hidden graders and detectors. A scripted honest agent and a scripted cheating agent check the detectors: cheating is caught in 100% of open episodes and 0% of control twins (96 scripted episodes, 0 errors).
- **Hold-out check.** Training tasks never contain the eval setups; a script enforces it (0 problems).
- **Data pipeline.** Pinned downloads, story leakage check, token-matched mixes (exactly 4,000,000 tokens per condition).
- **Story generator.** Built from code-made story cards and written by a local open model (Qwen3-1.7B). Separate package; the repos only share files.
- **Training.** Story training, bash SFT and GRPO all run. Post-training refuses to start if the config differs between conditions.
- **Statistics.** Bootstrap intervals over seeds and episodes, effect sizes, capability gate.

## Smoke test results (Tier 1, SmolLM2-360M)

| Stage | Result |
|:-|:-|
| Story training, 3.9M tokens (5% stories) | 3 min, 22.5k tokens/s, 12.7 GB peak; held-out loss 2.593 to 2.487 |
| Stories, 150 cards x 3 | Simple prompts: 447 stories in 34 s, word counts matched across conditions (376 to 385). Story cards: 265 kept in 105 s, but an over-strict word filter dropped 185 (fixed, not rerun: another agent held the GPU). Leakage check: 0 overlaps with eval text in both |
| Bash SFT, 1,238 runs over 12 families | 3 min, 2.5 GB peak |
| GRPO, 3 steps | runs; reward and update path work |
| Harness, 88 model episodes | training-like tasks: **100%** solved. Eval tasks (control twins): **0%** solved |
| Log-prob proxy, 12 items | aligned answer preferred 33% (SFT model), 42% (base model) |
| Geodesic MCQ, 200 items | misaligned answer preferred 39% (SFT), 38% (base) |

**What the harness smoke shows.** After 3 minutes of SFT, the model aces tasks like its training tasks but fails the eval tasks, and almost always claims success anyway. One real shortcut: it failed to fix the code, switched off the linter in `lint.cfg`, then reported "Removed the print call... run_checks passes now." The false-report detector fires as often in the control twin as in the open task (1.00 vs 1.00), so at this skill level it measures copied success reports, not lying. The twin design catches that. **The next job is competence on the eval-like tasks** (next-steps item 2).

## Deviations from the spec, and why

1. **Docker became bubblewrap** (owner ruling): faster, no sudo, behind one interface.
2. **Tier 1 runs the full pipeline** at 360M, not 135M (owner ruling).
3. **No Claude-written training text** (Anthropic terms, CoS ruling): stories from a local model, SFT data from programs.
4. **Training covers the same kinds of work as the eval, but never its shortcut setups** (CoS ruling).
5. **storygen is a separate package** (owner ruling). It has a minimal theme list in the planned card format, far themes only.
6. **Two pilot findings change the harness.** It now has a blocked-attempt metric, and the capability gate uses the control twins. Both are in `docs/DECISIONS.md`.
7. **Own small GRPO loop instead of TRL's**: simpler for multi-turn bash. vLLM rollouts come later.
8. **The FP8 loss check was skipped**: it matters for Tier 2 only and the GPU was shared tonight. It is on the roadmap (M6).
9. **The log-prob proxy prefers the bad answers**, even in the base model. The 12 items need calibration before they are trusted.

## Before the push, you need to

1. Approve this summary and the README.
2. Create the public repo `GrimsVerk/nanoalignment`, run `scripts/setup-github.sh --public` and set pull requests to "collaborators only" (needs admin).
3. **Grey area:** the SFT data comes from programs, but a few fixed sentences in those templates (task texts, report lines) were written by Claude. Keep them, paraphrase them with a local model, or replace them?
4. Confirm the copyright holder ("GrimsVerk") and the author name for `CITATION.cff`.
