# Stories: where they live

The story generator moved to its own public repo on 2026-11-20: **[GrimsVerk/nanoalignment-stories](https://github.com/GrimsVerk/nanoalignment-stories)** (MIT), with its git history. Everything about making stories is there: code (`storygen/`), prompts, the review rubric and scripts, and the logs that used to be here (`docs/storygen-prompt-log.md`, `docs/storygen-run-log.md`, `docs/storygen-review-rubric.md`).

This repo only consumes the output:
- `STORIES_DIR` in `local.env` (default `data/cache/stories`) holds one output dir per run (`stories.jsonl`, `manifest.json`, ...). Configs name a run dir relative to it (`configs/data/*.yaml`, `stories.dir`).
- `data.mix` copies the run's `code_commit` (the nanoalignment-stories commit that wrote it) and prompt version into each mix manifest (`stories` field).
- `data.leakage <run>` checks a run against the evaluation texts; `data.leakage --dump-refs <path>` writes the texts the generator filters against (put them in the generator checkout's `data/eval_refs.jsonl`).
- No code is shared either way (`tests/test_core.py::test_storygen_is_separate`).

Night 1 (prompt p6, 5,742 kept stories, written before the move by nanoalignment commit `1d38155`) is in `data/cache/stories/night1/` of the main checkout.
