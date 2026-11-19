"""Harness selftest: play scripted honest and shortcut policies on every scenario and
check that graders and detectors give the expected answer.

  uv run python -m harness.selftest [--seeds 3]
"""

import argparse
import sys

from harness import episode, scenarios
from harness import sandbox as sbx


def expected(scn_id: str, variant: str, policy: str, honest_fails: bool) -> dict:
    """What the detectors must say for a scripted policy."""
    s = scenarios.get(scn_id)
    if policy == "honest":
        return dict(success=not honest_fails, shortcut=False, honest_report=True, attempt=False)
    if variant == "control":  # the twin blocks the shortcut, but the attempt is still seen
        return dict(shortcut=False, attempt=s.shortcut_kind in ("tamper", "read"))
    want = dict(shortcut=True)
    if s.shortcut_kind == "tamper":
        want.update(success=False, shortcut_tamper=True)
    return want


def check(seeds: int = 3, verbose: bool = False) -> list[str]:
    sb = sbx.get("bwrap")
    errors = []
    cases = []
    for sid in scenarios.ALL:
        variants = ["neutral"] if sid.startswith("neutral/") else ["open", "control"]
        for v in variants:
            for seed in range(seeds):
                for pol in ("honest", "shortcut") if v != "neutral" else ("honest",):
                    cases.append((sid, v, seed, pol))
    eps = []
    for sid, v, seed, pol in cases:
        inst = scenarios.instance(sid, v, seed)
        eps.append(
            episode.Episode(
                inst, sb, script=inst.honest_actions if pol == "honest" else inst.shortcut_actions
            )
        )
    fails = [e.inst.honest_fails for e in eps]
    recs = episode.run(eps, episode.scripted_policy)
    for (sid, v, seed, pol), rec, hf in zip(cases, recs, fails, strict=True):
        for k, want in expected(sid, v, pol, hf).items():
            if rec[k] != want:
                errors.append(f"{sid} {v} seed {seed} {pol}: {k}={rec[k]!r}, expected {want!r}")
                if verbose:
                    for st in rec["steps"]:
                        print("   $", st["action"], "\n     ", st["obs"].replace("\n", "\n      "))
    print(f"selftest: {len(cases)} scripted episodes, {len(errors)} errors")
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    errs = check(a.seeds, a.v)
    for e in errs:
        print("  ", e)
    sys.exit(1 if errs else 0)
