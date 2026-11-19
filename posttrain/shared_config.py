"""The shared post-training config, and the assertion that every condition uses it.

All conditions of one training order must post-train with an identical resolved config
(spec section 2). The first condition to run writes a lock file with the config's hash;
every later condition must match it, or the run refuses to start. There is no way to pass
per-condition overrides: only the tier is a parameter.
"""

import json
from pathlib import Path

from common import config
from common.env import ROOT

SHARED = "posttrain/shared.yaml"


class ConfigMismatch(RuntimeError):
    pass


def resolve(tier: str, path: str = SHARED) -> dict:
    """The shared config for one tier, with the tier block merged into sft and grpo."""
    cfg = config.load(path)
    tiers = cfg.pop("tiers")
    if tier not in tiers:
        raise KeyError(f"tier {tier!r} not in {path}")
    for stage in ("sft", "grpo"):
        cfg[stage] = {**cfg[stage], **tiers[tier].get(stage, {})}
    cfg["tier"] = tier
    if cfg["grpo"]["reward"] != "hidden_grader" and not cfg["grpo"]["hackable_envs"]:
        raise ConfigMismatch("reward must be the hidden grader unless hackable_envs is on")
    return cfg


def lock_path(order: str, tier: str, base: Path | None = None) -> Path:
    return (base or ROOT / "results") / f"order{order}" / tier / "posttrain.lock.json"


def assert_shared(order: str, tier: str, condition: str, cfg: dict, base: Path | None = None) -> str:
    """Record or check the config hash for this order and tier. Returns the hash."""
    h = config.digest(cfg)
    p = lock_path(order, tier, base)
    if p.exists():
        lock = json.loads(p.read_text())
        if lock["digest"] != h:
            raise ConfigMismatch(
                f"post-training config for condition {condition!r} differs from the one used by "
                f"{lock['conditions']} in order {order} ({h} != {lock['digest']}). Refusing to run."
            )
        if condition not in lock["conditions"]:
            lock["conditions"].append(condition)
            p.write_text(json.dumps(lock, indent=2))
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(digest=h, conditions=[condition], config=cfg), indent=2))
    return h
