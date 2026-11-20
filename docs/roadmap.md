# Roadmap

Order owner-ruled 2026-11-19: bash competence first, then the shared checkpoint, then stories. Tier 1 (SmolLM2-360M) runs every stage first as a fast test bed; Tier 2 (Granite-4.0-1B-Base) follows.

| # | Milestone | Done when | State |
|:-|:-|:-|:-|
| M0 | Scaffold | code for every stage, smoke tests pass | done (STOP 3) |
| M1 | Harness frozen | 5 scenarios + control twins + neutral suite; selftest and hold-out check pass; transcripts read by a human | code done; human read pending; one generator fix pending (maxmin bug with equal inputs) |
| M2 | Bash SFT (C0, then C0s) | SFT to the validation plateau on all training families plus open datasets; skills rounds for the three skills the control twins lacked (owner ruling 2026-11-20; at most two rounds guided by the twins); capability gate: control twins at or above 0.5 | C0 done (`docs/c0-training.md`); skills round 1 done (twins 0.44); round 2 running |
| M3 | Floor-effect tracks (preregistration, "Floor-effect design") | (a) harder honest RL (hard variants, reward = hidden grader); (b) hackable RL (reward = visible check) with the open-eval shortcut curve over steps; (c) the cheater: hackable RL from C0s, checkpoint chosen at a cheat rate near 0.5 on the cheat validation set, never on the eval; fallback scripted cheating demos | hackable and hard variants, cheat validation set and GRPO `--hackable` done; runs next |
| M4 | Story corpus | storygen (sister repo) produces far-theme matched triples at volume; leakage check passes; human spot check of 50 per condition | minimal generator |
| M5 | Dose sweep pilot (Tier 1) | story doses 0.01, 0.05, 0.2, 0.5 on the cheater, bash replay in every arm; three metrics per arm; pick the main-run dose; freeze the preregistration | |
| M6 | FP8 check | 1-hour FP8 vs bf16 loss check on Tier 2 before trusting FP8 | |
| M7 | Cheater-track main runs (primary result) | neutral, positive and negative story arms from the same cheater, 3 seeds each, same bash replay; per arm and checkpoint: capability (control twins, neutral suite), shortcut rate over all open episodes (primary: positive minus neutral), cheat share of successes; capability drop of more than 10 points against neutral (or all arms far below the cheater) fixed with the same change in every arm before the result counts | |
| M8 | Order A / B runs (Tier 2) | 4 conditions x 3 (A) or 2 (B) seeds; tracks (a) and (b) as secondary results | after the CoS says Tier 2 |
| M9 | Write-up | all tracks side by side, all three metrics with intervals, transcripts released | |

Later experiments: near vs far themes; the filtered condition; JSON-tools interface as a robustness check.

## Stage commands

```sh
make check                                    # lint, tests, harness selftest, hold-out check
uv run python -m data.download --config data/tier1.yaml
(cd ../nanoalignment-stories && uv run --extra vllm python -m storygen.generate --cards N --out <STORIES_DIR>/<name>)
uv run python -m data.leakage stories/<name>
uv run python -m data.mix --config data/tier1.yaml --condition positive --order A
uv run python -m posttrain.sft_data --tier tier1
uv run python -m posttrain.sft  --tier tier1 --order A --condition C0 --run tier1_C0_sft
uv run python -m posttrain.grpo --tier tier1 --order A --condition C0 --init checkpoints/tier1_C0_sft --run tier1_C0
uv run python -m train.cpt --train-config train/tier1.yaml --mix tier1_A_positive --init checkpoints/tier1_C0 --run ...
uv run python -m eval.run_harness --model checkpoints/<run> --condition positive --seed 0 --run ...
uv run python -m eval.logprob --model checkpoints/<run> --run ... --mcq-max 0
```
