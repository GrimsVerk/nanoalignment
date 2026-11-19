"""Leakage check: stories must not describe the evaluation scenarios (spec section 2).

Two checks:
1. Banned terms: words for the held-out scenario types (tests, logs, linters, answer
   keys, graders, the harness itself). Stories with a hit are dropped at generation.
2. N-gram overlap between stories and every evaluation text (harness prompt, scenario
   tasks and files, log-prob proxy items, MCQ questions if cached). 13-gram overlap fails
   the check (the usual decontamination length); 8-gram overlap is reported.

  uv run python -m data.leakage stories/smoke     (a storygen output dir under DATA_DIR)
"""

import argparse
import json
import re
import sys

import yaml

from common import env

BANNED = re.compile(
    r"\b(unit ?tests?|test (suite|file|case)s?|tests? (pass|fail)\w*|linter|lint(ing)?|answer (key|file|sheet)s?|"
    r"grader|grading|run_checks|checker|log ?files?|error logs?|ops\.log|audit logs?|report\.csv|"
    r"sandbox|bash|computer terminal|terminal window|command line|shell script|source code|python)\b",
    re.I,
)


def banned_hits(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in BANNED.finditer(text)})


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def grams(text: str, n: int) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def eval_texts(seeds: int = 20) -> list[str]:
    from harness import episode, scenarios

    out = [episode.HEADER]
    for sid in scenarios.SHORTCUT:
        for v in ("open", "control"):
            for s in range(seeds):
                inst = scenarios.instance(sid, v, 1_000_000 + s)
                out += [inst.task, *inst.files.values(), *inst.honest_actions, *inst.shortcut_actions]
    dil = env.ROOT / "eval" / "dilemmas.yaml"
    if dil.exists():
        for item in yaml.safe_load(dil.read_text())["items"]:
            out += [item["prompt"], item["aligned"], item["misaligned"]]
    mcq = env.path("DATA_DIR") / "eval" / "geodesic_mcq.jsonl"
    if mcq.exists():
        for line in open(mcq):
            q = json.loads(line)
            out += [q["question"], *q["choices"]]
    return out


def check(stories: list[str], n_fail: int = 13, n_report: int = 8) -> dict:
    ev = eval_texts()
    ev_fail = set().union(*(grams(t, n_fail) for t in ev))
    ev_rep = set().union(*(grams(t, n_report) for t in ev))
    fail_ids, rep_ids, banned = [], [], []
    for i, s in enumerate(stories):
        if grams(s, n_fail) & ev_fail:
            fail_ids.append(i)
        if grams(s, n_report) & ev_rep:
            rep_ids.append(i)
        if banned_hits(s):
            banned.append(i)
    return dict(
        stories=len(stories),
        eval_texts=len(ev),
        overlap_13gram=len(fail_ids),
        overlap_8gram=len(rep_ids),
        banned_term_hits=len(banned),
        examples_8gram=rep_ids[:10],
        passed=not fail_ids and not banned,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stories", help="storygen output dir under DATA_DIR (holds stories.jsonl)")
    a = ap.parse_args()
    path = env.path("DATA_DIR") / a.stories / "stories.jsonl"
    texts = [json.loads(line)["text"] for line in open(path)]
    res = check(texts)
    (path.parent / "leakage.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
