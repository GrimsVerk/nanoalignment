"""Build a token-matched training mix for one condition.

Every condition gets exactly `total_tokens` tokens (spec: matched tokens per condition).
  story slice   = story_dose * total   (baseline and filtered: filled with more base text)
  replay slice  = replay shares * total (Order A only: bash SFT trajectories)
  base slice    = the rest, split by base_mix shares
Documents are tokenized with the model's tokenizer, separated by EOS and packed into one
flat uint32 array (DATA_DIR/mix/<name>.bin). A manifest records counts, repeats, the
config digest, a sha256 of the output, and which nanoalignment-stories run and commit wrote the
story slice (from that run's manifest.json). Sources that run short are repeated, and the
repeat factor is logged (a smoke test may repeat; main runs should not).

  uv run python -m data.mix --config data/smoke.yaml --condition positive --order B
"""

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from common import config, env
from data import filter_ai

Tokenize = Callable[[str], list[int]]


def read_jsonl(path: Path, condition: str | None = None) -> list[str]:
    out = []
    for line in open(path):
        rec = json.loads(line)
        if condition is None or rec.get("condition") == condition:
            out.append(rec["text"])
    return out


def take(docs: list[str], budget: int, tok: Tokenize, eos: int, rng: random.Random) -> tuple[list[int], dict]:
    """Tokens from `docs` (shuffled, cycled if needed) until exactly `budget` tokens."""
    if budget <= 0:
        return [], dict(tokens=0, docs=0, repeats=0.0)
    if not docs:
        raise ValueError("source has no documents")
    order = list(range(len(docs)))
    rng.shuffle(order)
    out: list[int] = []
    used = 0
    while len(out) < budget:
        ids = tok(docs[order[used % len(docs)]]) + [eos]
        out.extend(ids)
        used += 1
    return out[:budget], dict(tokens=budget, docs=used, repeats=round(used / len(docs), 3))


def plan(cfg: dict, condition: str, order: str) -> dict[str, int]:
    """Token budget per source for one condition."""
    conds = config.load("conditions.yaml")
    c = conds["conditions"][condition]
    total = int(cfg["total_tokens"])
    budget: dict[str, int] = {}
    story = int(cfg["story_dose"] * total) if c.get("stories") else 0
    if story:
        budget[f"stories:{c['stories']}"] = story
    replay = 0
    if conds["orders"][order]["replay"]:
        for src, share in (cfg.get("replay") or {}).items():
            budget[f"replay:{src}"] = int(share * total)
            replay += budget[f"replay:{src}"]
    rest = total - story - replay
    shares = cfg["base_mix"]
    z = sum(shares.values())
    names = list(shares)
    for i, src in enumerate(names):
        budget[src] = (
            rest - sum(budget.get(n, 0) for n in names[:i])
            if i == len(names) - 1
            else int(rest * shares[src] / z)
        )
    assert sum(budget.values()) == total, budget
    return budget


def build(
    cfg: dict, condition: str, order: str, tok: Tokenize, eos: int, sources: dict[str, list[str]]
) -> tuple[np.ndarray, dict]:
    c = config.load("conditions.yaml")["conditions"][condition]
    rng = random.Random(f"{cfg['seed']}:{condition}:{order}")
    parts, stats = [], {}
    for name, n in plan(cfg, condition, order).items():
        docs = sources[name]
        if c.get("filter_ai_discourse") and not name.startswith(("stories:", "replay:")):
            docs, removed = filter_ai.filter_docs(docs)
            stats[f"{name}:filtered_out"] = removed
        ids, st = take(docs, n, tok, eos, rng)
        parts.append(ids)
        stats[name] = st
    # Shuffle at document granularity would need doc offsets; interleave in fixed blocks instead,
    # so stories are spread through training rather than all at the end.
    arr = interleave(parts, block=int(cfg["seq_len"]), rng=rng)
    return arr, stats


def interleave(parts: Iterable[list[int]], block: int, rng: random.Random) -> np.ndarray:
    blocks = []
    for p in parts:
        blocks += [p[i : i + block] for i in range(0, len(p), block)]
    rng.shuffle(blocks)
    return np.array([t for b in blocks for t in b], dtype=np.uint32)


def load_sources(cfg: dict, condition: str, order: str) -> dict[str, list[str]]:
    raw = env.path("DATA_DIR") / "raw"
    out = {}
    for name in plan(cfg, condition, order):
        if name.startswith("stories:"):
            path = env.path("STORIES_DIR") / cfg["stories"]["dir"] / "stories.jsonl"
            out[name] = read_jsonl(path, name.split(":", 1)[1])
        elif name.startswith("replay:"):
            out[name] = read_jsonl(raw / f"{name.split(':', 1)[1]}.jsonl")
        else:
            out[name] = read_jsonl(raw / f"{name}.jsonl")
    return out


def stories_provenance(cfg: dict) -> dict:
    """Which nanoalignment-stories run and commit produced the story slice (from its manifest.json)."""
    d = cfg.get("stories", {}).get("dir")
    if not d:
        return {}
    mp = env.path("STORIES_DIR") / d / "manifest.json"
    m = json.loads(mp.read_text()) if mp.exists() else {}
    return dict(dir=d, code_commit=m.get("code_commit", "unknown"), code_commits=m.get("code_commits", []),
                prompt_version=m.get("prompt_version"), canary=m.get("canary"))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/smoke.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--order", default="B", choices=["A", "B"])
    ap.add_argument(
        "--stories", help="override stories.dir (a nanoalignment-stories output dir under STORIES_DIR)"
    )
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.stories:
        cfg["stories"]["dir"] = a.stories
    from transformers import AutoTokenizer

    repo, rev = config.model_ref(cfg["model"])
    tk = AutoTokenizer.from_pretrained(repo, revision=rev)
    tok = lambda s: tk(s, add_special_tokens=False)["input_ids"]
    arr, stats = build(
        cfg, a.condition, a.order, tok, tk.eos_token_id, load_sources(cfg, a.condition, a.order)
    )
    name = f"{Path(a.config).stem}_{a.order}_{a.condition}"
    out = env.path("DATA_DIR") / "mix" / f"{name}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out)
    manifest = dict(
        name=name,
        condition=a.condition,
        order=a.order,
        model=repo,
        model_revision=rev,
        tokens=int(arr.size),
        dtype="uint32",
        seq_len=cfg["seq_len"],
        config_digest=config.digest(cfg),
        config=cfg,
        sources=stats,
        stories=stories_provenance(cfg),
        sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
    )
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("name", "tokens", "sources", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
