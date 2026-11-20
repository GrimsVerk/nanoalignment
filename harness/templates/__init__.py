"""Sentence bank for the procedural training families.

`slots.yaml` says what each sentence must mean (instructions, written by Claude);
`bank.json` holds the sentences, written by a local open model and filtered by code
(`python -m harness.templates.paraphrase`). Families call `fill(rng, slot, **values)`.

Until the bank has a slot, `fill` returns a marked placeholder; `posttrain.sft_data`
refuses to build training data while any slot is missing (`missing()`).
"""

import json
import os
import random
import re
from functools import cache
from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"\{(\w+)\}")


@cache
def slots() -> dict:
    return yaml.safe_load((DIR / "slots.yaml").read_text())


def bank_path() -> Path:
    """bank.json, or a test bank named by NA_BANK (tests only; sft_data refuses it)."""
    return Path(os.environ.get("NA_BANK") or DIR / "bank.json")


@cache
def bank_file() -> dict:
    p = bank_path()
    return json.loads(p.read_text()) if p.exists() else {"slots": {}, "meta": {}}


def bank() -> dict[str, list[str]]:
    return bank_file()["slots"]


def provenance_ok() -> bool:
    """True if the bank is bank.json written by the pinned local paraphraser (not a test bank)."""
    from common import config

    pin = config.pins()["models"]["paraphraser"]
    m = bank_file()["meta"]
    return (
        bank_path() == DIR / "bank.json"
        and m.get("model") == pin["repo"]
        and m.get("revision") == pin["revision"]
    )


def missing() -> list[str]:
    b = bank()
    return [s for s in slots() if not b.get(s)]


def fill(r: random.Random, slot: str, **values) -> str:
    spec = slots()[slot]
    want = set(spec["vars"])
    if set(values) != want:
        raise KeyError(f"{slot}: needs {sorted(want)}, got {sorted(values)}")
    variants = bank().get(slot)
    if not variants:
        return f"<{slot}: " + " ".join(f"{k}={v}" for k, v in sorted(values.items())) + ">"
    return r.choice(variants).format(**{k: str(v) for k, v in values.items()})
