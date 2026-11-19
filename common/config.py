"""YAML configs. A config may name a parent with `inherit: <path relative to configs/>`."""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from common.env import ROOT

CONFIGS = ROOT / "configs"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = (CONFIGS / p) if not p.exists() else p.resolve()
    cfg = yaml.safe_load(p.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load(CONFIGS / parent), cfg)
    return cfg


def digest(cfg: dict[str, Any]) -> str:
    """Stable hash of a resolved config (key order does not matter)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def pins() -> dict[str, Any]:
    return load(CONFIGS / "pins.yaml")


def model_ref(name: str) -> tuple[str, str]:
    """(repo id, pinned revision) for a model key in configs/pins.yaml."""
    m = pins()["models"][name]
    return m["repo"], m["revision"]


def dataset_ref(name: str) -> dict[str, Any]:
    return dict(pins()["datasets"][name])
