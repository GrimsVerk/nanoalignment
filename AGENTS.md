# nanoalignment: rules for agents

Do good stories make good AIs? Alignment pretraining on a single GPU.

## Every session

- Read `next-steps.md` first. Write your task there before you start; delete it when done. Nothing may live only in chat.
- One feature per session. When unsure, pick a reasonable default, note it in `docs/DECISIONS.md`, keep going.
- Spec: `docs/` (`preregistration.md`, `roadmap.md`, `DECISIONS.md`). Research behind the choices: `docs/research-notes.md`, `docs/research-2.md`.

## Hard rules

- **No Claude-written text as training data**: no stories, trajectories or demos written by Claude (Anthropic usage policy). Stories come from a local open model via the sister repo [nanoalignment-stories](https://github.com/GrimsVerk/nanoalignment-stories); SFT data is procedural.
- **Training tasks never contain eval shortcut setups.** Run `uv run python -m posttrain.holdout` after touching `harness/scenarios/`.
- **The story generator lives in GrimsVerk/nanoalignment-stories**, never here. This repo only reads its output (`stories.jsonl`, `manifest.json`) from `STORIES_DIR` (`local.env`) and records the generator commit in each mix manifest. Story work (prompts, runs, prompt and run logs) goes in that repo. Pointer: `docs/stories.md`.
- Post-training config is shared across conditions (`configs/posttrain/shared.yaml`); never add per-condition settings.
- Model output runs only in the sandbox (`harness/sandbox.py`), never on the host. Never `trust_remote_code`.
- GPU is shared: check `nvidia-smi` before loading a model; keep vLLM `gpu_memory_utilization` modest; at most 4 heavy torch processes, total RSS under 12 GB.
- Background jobs: `nohup ... > results/logs/<name>.log 2>&1 &`, then poll the log. Never wait on `pgrep -f`. Never `pkill -f`.
- Machine-specific paths go in `local.env` (copy `local.env.example`). Secrets in `.env`.

## Checks

```sh
make check   # ruff, pytest, harness selftest, hold-out check
```

## Licensing

MIT. Geodesic's eval set has no licence: load it by pinned id, never commit or rehost it.
