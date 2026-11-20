"""Train / validation / test splits for the training families (owner requirement, bash-c0-1).

- train:      what SFT and RL learn from (seed ranges `sft.data.instance_seed_offset` and
              `grpo.instance_seed_offset` up, minus anything held out below).
- validation: same families, other instances; early stopping and checkpoint choice.
- test:       a further set, evaluated once at the very end.
The eval shortcut scenarios are separate again (posttrain/holdout.py): not a split.

Deterministic by seed: validation takes the first `per_family` instances of each family
from its seed range whose content is new, then test does the same from its range,
skipping anything already in validation. Training instances whose content matches a
held-out one are skipped. "Content" is the problem, not the ID or the wording: the
family, the work-dir files, the check files and the read-only list, hashed (`keys`).

  uv run python -m posttrain.splits            # build, record and check
"""

import hashlib
import json
from functools import cache

from common import env
from harness import scenarios
from harness.scenario import Instance
from posttrain import holdout, shared_config


def keys(inst: Instance) -> tuple[str, ...]:
    """Content keys of one instance: the problem itself (family, work files, check files,
    read-only list), ignoring the task wording. A reworded copy of a held-out task has the
    same files, so it is caught; parameters that live only in the task text (a group name,
    an increment) are ignored, which can only make the check stricter."""
    blob = json.dumps(
        dict(fam=inst.scenario, files=inst.files, check=inst.check_files, ro=sorted(inst.ro_files)),
        sort_keys=True,
    )
    return (hashlib.sha1(blob.encode()).hexdigest()[:16],)


def _take(fam: str, offset: int, n: int, seen: set[str]) -> list[dict]:
    out, s = [], offset
    while len(out) < n:
        inst = scenarios.instance(fam, "neutral", s)
        k = keys(inst)
        if not seen.intersection(k):
            seen.update(k)
            out.append(dict(family=fam, seed=s, keys=list(k)))
        s += 1
        if s - offset > 50 * n:
            raise RuntimeError(f"{fam}: cannot find {n} distinct instances from seed {offset}")
    return out


@cache
def heldout(tier: str = "tier1") -> dict[str, list[dict]]:
    """{'val': [...], 'test': [...]} with family, seed and content keys per instance."""
    cfg = shared_config.resolve(tier)["splits"]
    seen: set[str] = set()
    out = {}
    for split in ("val", "test"):
        c = cfg[split]
        out[split] = [
            r
            for fam in holdout.training_families()
            for r in _take(fam, c["seed_offset"], c["per_family"], seen)
        ]
    return out


@cache
def heldout_keys(tier: str = "tier1") -> frozenset[str]:
    return frozenset(k for rs in heldout(tier).values() for r in rs for k in r["keys"])


def train_instance(fam: str, seed: int, tier: str = "tier1") -> Instance | None:
    """A training instance, or None if its content matches a validation or test instance."""
    inst = scenarios.instance(fam, "neutral", seed)
    return None if heldout_keys(tier).intersection(keys(inst)) else inst


def split_instances(split: str, tier: str = "tier1", per_family: int | None = None) -> list[Instance]:
    """Held-out instances, optionally only the first `per_family` of each family."""
    rows, count = [], {}
    for r in heldout(tier)[split]:
        count[r["family"]] = count.get(r["family"], 0) + 1
        if per_family is None or count[r["family"]] <= per_family:
            rows.append(r)
    return [scenarios.instance(r["family"], "neutral", r["seed"]) for r in rows]


def check(train_keys: list[tuple[str, str]], tier: str = "tier1") -> list[str]:
    """No instance may appear in two splits. `train_keys`: content keys of the training set."""
    h = heldout(tier)
    errs = []
    v = {k for r in h["val"] for k in r["keys"]}
    t = {k for r in h["test"] for k in r["keys"]}
    if v & t:
        errs.append(f"{len(v & t)} content keys in both validation and test")
    tr = {k for ks in train_keys for k in ks}
    if tr & v:
        errs.append(f"{len(tr & v)} content keys in both train and validation")
    if tr & t:
        errs.append(f"{len(tr & t)} content keys in both train and test")
    return errs


def record(tier: str = "tier1") -> dict:
    """Write the held-out lists to DATA_DIR/splits/ and return a small summary."""
    h = heldout(tier)
    d = env.path("DATA_DIR") / "splits"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tier}_heldout.json").write_text(json.dumps(h))
    digest = hashlib.sha1(json.dumps(h, sort_keys=True).encode()).hexdigest()[:16]
    return dict(tier=tier, val=len(h["val"]), test=len(h["test"]), digest=digest,
                config=shared_config.resolve(tier)["splits"])  # fmt: skip


if __name__ == "__main__":
    s = record()
    errs = check([])
    print(json.dumps(s), f"{len(errs)} problems", *errs, sep="\n")
