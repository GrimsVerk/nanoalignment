"""Statistics for harness results: bootstrap confidence intervals and effect sizes.

Episodes are nested in training seeds (one trained model per condition and seed), so
intervals come from a two-level (cluster) bootstrap: resample seeds, then episodes within
each resampled seed. With one seed this reduces to an ordinary bootstrap.

Notation: a rate p is the share of episodes where a flag is true (for example
shortcut). The difference between conditions A and B is d = p_A - p_B. Cohen's h is
h = 2 asin(sqrt(p_A)) - 2 asin(sqrt(p_B)); |h| of 0.2, 0.5, 0.8 is small, medium, large.
"""

import math
from collections import defaultdict

import numpy as np


def _groups(records: list[dict], key: str, cluster: str) -> list[np.ndarray]:
    g = defaultdict(list)
    for r in records:
        g[r.get(cluster, 0)].append(float(bool(r[key])))
    return [np.array(v) for _, v in sorted(g.items())]


def boot_rates(
    records: list[dict], key: str, n_boot: int, rng: np.random.Generator, cluster: str = "seed"
) -> np.ndarray:
    groups = _groups(records, key, cluster)
    if not groups:
        return np.full(n_boot, np.nan)
    out = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, len(groups), len(groups))
        vals = [groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in picks]
        out[b] = np.concatenate(vals).mean()
    return out


def rate_ci(
    records: list[dict],
    key: str,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    cluster: str = "seed",
) -> dict:
    rng = np.random.default_rng(seed)
    p = float(np.mean([bool(r[key]) for r in records])) if records else float("nan")
    bs = boot_rates(records, key, n_boot, rng, cluster)
    lo, hi = np.nanquantile(bs, [(1 - ci) / 2, 1 - (1 - ci) / 2]) if records else (np.nan, np.nan)
    return dict(rate=p, lo=float(lo), hi=float(hi), n=len(records))


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def compare(
    a: list[dict],
    b: list[dict],
    key: str,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    cluster: str = "seed",
) -> dict:
    """Difference in rates a - b, with a bootstrap CI and a two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    pa = float(np.mean([bool(r[key]) for r in a]))
    pb = float(np.mean([bool(r[key]) for r in b]))
    d = boot_rates(a, key, n_boot, rng, cluster) - boot_rates(b, key, n_boot, rng, cluster)
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
    return dict(
        diff=pa - pb, lo=float(lo), hi=float(hi), p_boot=p, cohens_h=cohens_h(pa, pb), n_a=len(a), n_b=len(b)
    )


def competence(records: list[dict], gate: float) -> dict:
    """Capability check: success where no shortcut is possible. Two parts, both reported:
    the neutral suite (training-like families) and the control twins (the eval tasks
    themselves). The gate applies to the control twins: they are the same tasks as the
    open variants, so a failing gate means shortcut rates would mostly measure skill."""
    nan = dict(rate=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    ok = lambda r: not r.get("honest_fails")
    neutral = [r for r in records if r["variant"] == "neutral" and ok(r)]
    control = [r for r in records if r["variant"] == "control" and ok(r)]
    res = rate_ci(control, "success") if control else dict(nan)
    res["neutral_suite"] = rate_ci(neutral, "success") if neutral else dict(nan)
    res["passes_gate"] = bool(control) and res["rate"] >= gate
    return res


METRICS = ("shortcut", "attempt", "success", "honest_report", "accidental_damage", "false_claim")


def summarise(records: list[dict], gate: float = 0.5, n_boot: int = 2000) -> dict:
    """Per condition: competence, and per scenario x variant the main rates with CIs."""
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r.get("condition", "?")].append(r)
    out = {}
    for cond, rs in sorted(by_cond.items()):
        cell = defaultdict(list)
        for r in rs:
            cell[(r["scenario"], r["variant"])].append(r)
        out[cond] = dict(
            competence=competence(rs, gate),
            cells={
                f"{s}|{v}": {m: rate_ci(c, m, n_boot) for m in METRICS if m in c[0]}
                for (s, v), c in sorted(cell.items())
            },
        )
        shortcut_eps = [r for r in rs if r["variant"] == "open" and not r["scenario"].startswith("neutral/")]
        if shortcut_eps:
            out[cond]["shortcut_rate_pooled"] = rate_ci(shortcut_eps, "shortcut", n_boot)
    return out
