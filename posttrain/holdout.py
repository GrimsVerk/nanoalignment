"""Hold-out check: no SFT or RL task may match an evaluation scenario.

Training covers the same kinds of WORK as the eval (fix bugs, edit records and logs, fix
lint errors, sum data, report failures honestly), but never the shortcut setups (spec
section 2; CoS 2026-11-20): no visible test files to edit, no answer keys, no logs of the
agent's own mistakes, no editable check configs. Run before building any training set:

  uv run python -m posttrain.holdout
"""

import re
import sys
from difflib import SequenceMatcher

from harness import scenarios
from harness.scenarios.neutral import COVERAGE, Neutral

# File names that belong to eval shortcut setups and must never appear in training tasks.
SETUP_FILES = re.compile(
    r"(^|/)(solutions/|ops\.log$|notes\.log$|lint\.cfg$|orders\.txt$|report\.csv$|test_[^/]*\.py$)"
)
MAX_SIMILARITY = 0.75  # task-text skeleton similarity to any eval task
NGRAM = 8


def skeleton(text: str) -> str:
    """Mask numbers, quoted values and file names so only the template is compared."""
    t = re.sub(r"\b\w+\.(py|txt|csv|log|json|cfg|md)\b", "<file>", text)
    t = re.sub(r"'[^']*'|\"[^\"]*\"", "<str>", t)
    t = re.sub(r"\d+", "<n>", t)
    return " ".join(t.lower().split())


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    w = skeleton(text).split()
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def training_families() -> list[str]:
    return [f"neutral/{f}" for f in Neutral.FAMILIES]


def check(seeds: int = 30) -> list[str]:
    errors = []
    eval_tasks = {}
    for sid in scenarios.SHORTCUT:
        for v in ("open", "control"):
            for s in range(5):
                eval_tasks.setdefault(sid, set()).add(skeleton(scenarios.instance(sid, v, s).task))
    eval_grams = set().union(*(ngrams(t) for ts in eval_tasks.values() for t in ts))
    for sid in COVERAGE:
        for fam in COVERAGE[sid]:
            if fam not in Neutral.FAMILIES:
                errors.append(f"coverage: {sid} maps to unknown training family {fam}")
    if set(COVERAGE) != set(scenarios.SHORTCUT):
        errors.append(
            f"coverage: eval scenarios without a training family: {set(scenarios.SHORTCUT) - set(COVERAGE)}"
        )
    for fam in training_families():
        for s in range(seeds):
            inst = scenarios.instance(fam, "neutral", s)
            where = f"{fam} seed {s}"
            if inst.protected or inst.forbidden or inst.ro_files or inst.unreadable:
                errors.append(f"{where}: has protected/forbidden files (a shortcut setup)")
            for rel in inst.files:
                if SETUP_FILES.search(rel):
                    errors.append(f"{where}: file {rel} belongs to an eval setup")
            sk = skeleton(inst.task)
            for sid, ts in eval_tasks.items():
                sim = max(SequenceMatcher(None, sk, t).ratio() for t in ts)
                if sim > MAX_SIMILARITY:
                    errors.append(f"{where}: task text {sim:.2f} similar to eval scenario {sid}")
            shared = ngrams(inst.task) & eval_grams
            if shared:
                errors.append(
                    f"{where}: shares {len(shared)} {NGRAM}-grams with eval tasks, e.g. {' '.join(next(iter(shared)))}"
                )
    return sorted(set(errors))


if __name__ == "__main__":
    errs = check()
    print(
        f"holdout: {len(training_families())} training families checked against {len(scenarios.SHORTCUT)} eval scenarios, {len(errs)} problems"
    )
    for e in errs[:40]:
        print("  ", e)
    sys.exit(1 if errs else 0)
