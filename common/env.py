"""Load machine-specific settings (local.env) and secrets (.env) into os.environ.

Both files are gitignored; committed copies end in .example. Values already in the
environment win, so a shell can override any setting.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "DATA_DIR": "data/cache",
    "CKPT_DIR": "checkpoints",
    "SANDBOX_TMP": "",
    "GPU_UTIL": "0.55",
}


def _read(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load() -> None:
    for name in ("local.env", ".env"):
        for k, v in _read(ROOT / name).items():
            if v and k not in os.environ:
                os.environ[k] = v
    for k, v in DEFAULTS.items():
        os.environ.setdefault(k, v)


def path(key: str) -> Path:
    """A path setting, resolved against the repo root when relative."""
    load()
    p = Path(os.environ[key])
    return p if p.is_absolute() else ROOT / p
