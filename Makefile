# One entry point per stage. Paths under DATA_DIR come from local.env (default data/cache).
SHELL := /bin/bash
DATA := data/cache
PY := uv run python

.PHONY: check lint test selftest holdout smoke smoke-data smoke-stories smoke-cpt smoke-sft smoke-harness smoke-scripted

check: lint test selftest holdout

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

selftest:
	$(PY) -m harness.selftest --seeds 3

holdout:
	$(PY) -m posttrain.holdout

# ---- smoke test, end to end on Tier 1 (about 20 minutes on an RTX 5080)
smoke: smoke-data smoke-stories smoke-cpt smoke-sft smoke-scripted smoke-harness

smoke-data:
	$(PY) -m data.download --config data/smoke.yaml

smoke-stories:
	nvidia-smi --query-gpu=memory.free --format=csv,noheader
	cd storygen && PYTHONPATH=. ../.venv/bin/python -m storygen.generate --cards 150 --out ../$(DATA)/stories/smoke
	$(PY) -m data.leakage stories/smoke

smoke-cpt:
	$(PY) -m data.mix --config data/smoke.yaml --condition positive --order B
	$(PY) -m train.cpt --train-config train/smoke.yaml --mix smoke_B_positive --run smoke_cpt_tier1

smoke-sft:
	$(PY) -m posttrain.sft_data --tier tier1 --out sft/tier1.jsonl --per-family 100
	$(PY) -m posttrain.sft --tier tier1 --order B --condition positive --init checkpoints/smoke_cpt_tier1 \
		--data sft/tier1.jsonl --run smoke_sft_tier1 --smoke

smoke-scripted:
	$(PY) -m eval.run_harness --agent scripted-honest --episodes 3 --run smoke_scripted_honest
	$(PY) -m eval.run_harness --agent scripted-shortcut --episodes 3 --run smoke_scripted_shortcut

smoke-harness:
	$(PY) -m eval.run_harness --model checkpoints/smoke_sft_tier1 --condition positive --episodes 4 --run smoke_harness_tier1
	$(PY) -m eval.logprob --model checkpoints/smoke_sft_tier1 --run smoke_logprob_tier1
