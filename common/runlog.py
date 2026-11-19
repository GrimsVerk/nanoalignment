"""Experiment tracking: one directory per run with plain JSON files.

Choice: local JSONL files instead of W&B, because runs happen on one offline-capable
machine, need no account or network, and agents and the owner can read, diff and commit
small summaries directly.

Layout of a run dir:
  config.json    the resolved config
  meta.json      seed, git commit, host, package versions, start time
  metrics.jsonl  one JSON object per logged step
  summary.json   final numbers (written by finish())
"""

import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from common.env import ROOT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout
        return sha.strip() + ("-dirty" if dirty.strip() else "")
    except OSError:
        return "unknown"


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("torch", "transformers", "trl", "vllm", "datasets", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


class RunLog:
    def __init__(self, run_dir: str | Path, config: dict[str, Any], seed: int):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
        meta = dict(
            seed=seed,
            git=_git_commit(),
            host=platform.node(),
            python=platform.python_version(),
            versions=_versions(),
            started=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._f = open(self.dir / "metrics.jsonl", "a")  # noqa: SIM115 (closed in finish)

    def log(self, **metrics: Any) -> None:
        metrics.setdefault("t", round(time.time() - self.t0, 2))
        self._f.write(json.dumps(metrics) + "\n")
        self._f.flush()

    def finish(self, **summary: Any) -> None:
        summary.setdefault("seconds", round(time.time() - self.t0, 1))
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._f.close()
