"""Hold-out check: no SFT or RL task may match an evaluation scenario.

Training covers the same kinds of WORK as the eval (fix bugs, edit records and logs, fix
lint errors, sum data, report failures honestly), but never the shortcut setups (spec
section 2; CoS 2026-11-20): no editable test files, no answer keys, no logs of the
agent's own mistakes, no editable check configs. Since bash-c0-1, training families may
show protected files (tests, checker configs, problem lists) if they are bound read-only,
use other file names than the eval and other task texts. Run before building any
training set:

  uv run python -m posttrain.holdout
"""

import re
import sys
from difflib import SequenceMatcher

from harness import scenarios
from harness.scenarios.neutral import COVERAGE, HACKABLE, Neutral

# File names that belong to eval shortcut setups and must never appear in training tasks.
SETUP_FILES = re.compile(
    r"(^|/)(solutions/|ops\.log$|notes\.log$|lint\.cfg$|orders\.txt$|report\.csv$|test_[^/]*\.py$"
    r"|answers?\.txt$|points\.csv$)"
)
MAX_SIMILARITY = 0.75  # task-text skeleton similarity to any eval task
NGRAM = 8
# Near copies of the control twins whose skills are taught directly (owner ruling 2026-11-20,
# after the C0 report): a training instance may not share most file names with one of these
# twins, nor hold a work file whose content (numbers and quoted values masked) is this close
# to one of the twin's files. Calibrated: the closest training family is 0.75 (record_ro vs
# log_tamper); renamed copies of the twins score 0.9 and up (tests/test_core.py).
NEAR_TWINS = ("disable_check", "forbidden_read", "log_tamper")
NEAR_NAMES = 0.5  # Jaccard overlap of file names (digits masked)
NEAR_CONTENT = 0.8  # SequenceMatcher ratio of masked file contents, same extension
NEAR_LAYOUT = 0.9  # row-shape ratio for the layout rule


def skeleton(text: str) -> str:
    """Mask numbers, quoted values and file names so only the template is compared."""
    t = re.sub(r"\b\w+\.(py|txt|csv|log|json|cfg|md)\b", "<file>", text)
    t = re.sub(r"'[^']*'|\"[^\"]*\"", "<str>", t)
    t = re.sub(r"\d+", "<n>", t)
    return " ".join(t.lower().split())


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    w = skeleton(text).split()
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def eval_ngrams() -> set[tuple[str, ...]]:
    """Skeleton 8-grams of the eval scenarios' task texts (both variants, a few seeds)."""
    grams: set[tuple[str, ...]] = set()
    for sid in scenarios.SHORTCUT:
        for v in ("open", "control"):
            for s in range(5):
                grams |= ngrams(scenarios.instance(sid, v, s).task)
    return grams


FILE_TOKEN = re.compile(r"(?:^|[\s'\"=/<>|;(])((?:[\w.-]+/)*[\w.-]+)")


def text_problems(text: str, grams: set[tuple[str, ...]]) -> list[str]:
    """Hold-out check for open-dataset items (task plus command): no eval setup file names,
    no shared 8-gram with an eval task text."""
    errs = []
    for tok in FILE_TOKEN.findall(text):
        if SETUP_FILES.search(tok):
            errs.append(f"names eval setup file {tok}")
    shared = ngrams(text) & grams
    if shared:
        errs.append(f"shares {len(shared)} {NGRAM}-grams with eval tasks")
    return errs


def _name_key(rel: str) -> str:
    return re.sub(r"\d+", "#", rel.rsplit("/", 1)[-1].lower())


def _content_key(text: str) -> str:
    t = re.sub(r"'[^']*'|\"[^\"]*\"", "<s>", text)
    return " ".join(re.sub(r"\d+", "#", t).lower().split())


def _shape(text: str) -> tuple[str, str]:
    """(header, row shape): the first line with digits masked, and the other lines with
    letter runs masked too. Catches tables and keys whose rows are random names."""
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return "", ""
    head = re.sub(r"\d+", "#", lines[0].lower())
    rows = [re.sub(r"[a-z]+", "a", re.sub(r"\d+", "#", x.lower())) for x in lines[1:]]
    return head, " ".join(rows)


def _ext(rel: str) -> str:
    base = rel.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1] if "." in base else ""


def near_twins(seeds: int = 5) -> list:
    return [
        scenarios.instance(sid, v, s) for sid in NEAR_TWINS for v in ("control", "open") for s in range(seeds)
    ]


def near_copy_problems(inst, twins) -> list[str]:
    """Structure check: file names and file contents close to an eval control twin."""
    errs = []
    names = {_name_key(k) for k in inst.files}
    for tw in twins:
        tn = {_name_key(k) for k in tw.files}
        j = len(names & tn) / max(1, len(names | tn))
        if j >= NEAR_NAMES and len(names & tn) >= 2:  # one shared module name (calc.py) is generic
            errs.append(f"file names {j:.2f} like the {tw.scenario} twin")
        for an, a in inst.files.items():
            for bn, b in tw.files.items():
                if _ext(an) != _ext(bn) or min(len(a), len(b)) < 30:
                    continue
                sim = SequenceMatcher(None, _content_key(a), _content_key(b)).ratio()
                if sim >= NEAR_CONTENT:
                    errs.append(f"{an} content {sim:.2f} like {bn} of the {tw.scenario} twin")
                    continue
                (ha, ra), (hb, rb) = _shape(a), _shape(b)
                hs, rs = SequenceMatcher(None, ha, hb).ratio(), SequenceMatcher(None, ra, rb).ratio()
                if hs >= NEAR_CONTENT and rs >= NEAR_CONTENT:
                    errs.append(
                        f"{an} structure (header {hs:.2f}, rows {rs:.2f}) like {bn} of the {tw.scenario} twin"
                    )
        # Layout: most of the twin's files have a same-type file of the same row shape here
        # (column names and values may differ).
        hit = set()
        for bn, b in tw.files.items():
            rb = _shape(b)[1]
            if rb.count(" ") < 1:
                continue
            for an, a in inst.files.items():
                if _ext(an) == _ext(bn) and SequenceMatcher(None, _shape(a)[1], rb).ratio() >= NEAR_LAYOUT:
                    hit.add(bn)
        if len(hit) >= 2 and len(hit) * 2 >= len(tw.files):
            errs.append(f"layout like the {tw.scenario} twin ({', '.join(sorted(hit))})")
    return errs


def training_families() -> list[str]:
    return [f"neutral/{f}" for f in Neutral.FAMILIES]


def hackable_families() -> list[str]:
    return [f"neutral/{f}" for f in HACKABLE]


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
    twins = near_twins()
    cases = [(f, "neutral") for f in training_families()] + [(f, "hackable") for f in hackable_families()]
    for fam, variant in cases:
        for s in range(seeds):
            inst = scenarios.instance(fam, variant, s)
            where = f"{fam} {variant} seed {s}"
            # Visible protected files are allowed only read-only (bash-c0-1): no shortcut is
            # ever possible in SFT and plain RL. Forbidden or unreadable files (answer keys)
            # never. The hackable variants (floor tracks b and c, owner ruling 2026-11-20) are
            # writable on purpose; every other check applies to them too.
            if inst.forbidden or inst.unreadable:
                errors.append(f"{where}: has forbidden/unreadable files (a shortcut setup)")
            if variant == "neutral" and not set(inst.protected) <= set(inst.ro_files):
                errors.append(f"{where}: protected files {inst.protected} are writable (a shortcut setup)")
            for rel in inst.files:
                if SETUP_FILES.search(rel):
                    errors.append(f"{where}: file {rel} belongs to an eval setup")
            sk = skeleton(inst.task)
            for sid, ts in eval_tasks.items():
                sim = max(SequenceMatcher(None, sk, t).ratio() for t in ts)
                if sim > MAX_SIMILARITY:
                    errors.append(f"{where}: task text {sim:.2f} similar to eval scenario {sid}")
            errors += [f"{where}: {e}" for e in near_copy_problems(inst, twins)]
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
