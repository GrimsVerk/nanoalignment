# Roadmap

Order owner-ruled 2026-11-19: bash competence first, then the shared checkpoint, then stories. Tier 1 (SmolLM2-360M) runs every stage first as a fast test bed; Tier 2 (Granite-4.0-1B-Base) follows.

| # | Milestone | Done when | State |
|:-|:-|:-|:-|
| M0 | Scaffold | code for every stage, smoke tests pass | done (STOP 3) |
| M1 | Harness frozen | 5 scenarios + control twins + neutral suite; selftest and hold-out check pass; transcripts read by a human | code done; human read pending |
| M2 | Bash SFT | SFT set over all training families (5k to 20k runs); Tier 1 then Tier 2 reach the pass mark (about 50%) on neutral tasks | smoke only |
| M3 | Bash RL and C0 | short GRPO on training families; **floor-effect pilot**: shortcut base rate after SFT + RL is measurably above zero (preregistration, risk 2); C0 saved | skeleton |
| M4 | Story corpus | storygen (sister repo) produces far-theme matched triples at volume; leakage check passes; human spot check of 50 per condition | minimal generator |
| M5 | Dose sweep pilot (Tier 1) | Order A at story doses 0.01, 0.05, 0.2, 0.5; pick the main-run dose; freeze the preregistration | |
| M6 | FP8 check | 1-hour FP8 vs bf16 loss check on Tier 2 before trusting FP8 | |
| M7 | Order A main runs (Tier 2) | 4 conditions x 3 seeds; harness, capability check, proxy, MCQ | |
| M8 | Order B runs (Tier 2) | 4 conditions x 2 seeds | |
| M9 | Write-up | both orders side by side, all metrics with intervals, transcripts released | |

Later experiments: `hackable_envs` RL (do stories reduce learning to hack?); near vs far themes; the filtered condition; JSON-tools interface as a robustness check.

## Stage commands

```sh
make check                                    # lint, tests, harness selftest, hold-out check
uv run python -m data.download --config data/tier1.yaml
(cd storygen && PYTHONPATH=. ../.venv/bin/python -m storygen.generate --cards N --out ../data/cache/stories/<name>)
uv run python -m data.leakage stories/<name>
uv run python -m data.mix --config data/tier1.yaml --condition positive --order A
uv run python -m posttrain.sft_data --tier tier1
uv run python -m posttrain.sft  --tier tier1 --order A --condition C0 --run tier1_C0_sft
uv run python -m posttrain.grpo --tier tier1 --order A --condition C0 --init checkpoints/tier1_C0_sft --run tier1_C0
uv run python -m train.cpt --train-config train/tier1.yaml --mix tier1_A_positive --init checkpoints/tier1_C0 --run ...
uv run python -m eval.run_harness --model checkpoints/<run> --condition positive --seed 0 --run ...
uv run python -m eval.logprob --model checkpoints/<run> --run ... --mcq-max 0
```
