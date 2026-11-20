"""Three general skills as training families (bash-c0-1, owner ruling 2026-11-20 after the
C0 report). C0 failed three eval control twins for lack of these skills, so they are taught
here in other surface forms than the eval: other file types, names, rules, task stories.

1. Remove the line(s) that break a rule (rule_strip, rule_strip_ro): JavaScript, Python,
   shell, YAML and Markdown files; rules such as no console.log, no debugger statements,
   no breakpoint(), no `set -x`, no plain-text passwords, no draft markers. The rule
   checker names file and line; the honest fix deletes exactly those lines.
2. Write to the exact output path the task gives (count_match, max_col, uniq_count; plus
   near-name distractor files in csv_sum and group_sum): files with similar names sit next
   to the target, and the path may be in a subdirectory that does not exist yet.
3. Fix values in a data file from a reference file (ref_fix, ref_sync): JSON, INI, TSV,
   YAML and CSV data; key=value, key: value, CSV and JSON references; the reference stays
   unchanged (read-only in ref_sync).

No eval setup appears here: no linter config with an on/off switch, no answer keys, no
logs of the agent's own mistakes, no eval file names (`posttrain/holdout.py` checks file
names, file structure and task wording against the eval control twins). Task texts and
reports come from the sentence bank (written by a local open model).
"""

import json
import re

from harness.scenario import unittest_file
from harness.templates import fill

# ---------------------------------------------------------------- shared


def _maybe(r, p: float, *acts: str) -> list[str]:
    return list(acts) if r.random() < p else []


def _no_check() -> dict[str, str]:
    return {"check.sh": 'echo "No automatic checks here."\n'}


INT_FORMAT_CHECK = """if [ ! -f "/work/{out}" ]; then echo "{out} not found"; exit 1; fi
if ! grep -qE '^-?[0-9]+$' "/work/{out}"; then echo "{out} should contain a single integer"; exit 1; fi
echo "{out}: format ok"
"""

SUBDIRS = ["results/", "out/", "build/", "stats/"]


def near_names(r, out: str, avoid: int) -> dict[str, str]:
    """Distractor files whose names resemble the output path (skill 2). Their content is an
    unrelated number, never the answer."""
    base = out.rsplit("/", 1)[-1]
    stem, dot, ext = base.rpartition(".")
    stem, ext = (stem, "." + ext) if dot else (base, "")
    pool = [f"{stem}s{ext}", f"{stem}_old{ext}", f"old_{stem}{ext}", f"{base}.bak", f"{stem}_v1{ext}",
            f"{stem}_draft{ext}", f"prev/{base}"]  # fmt: skip
    if "/" in out:
        pool.append(base)  # the right name in the wrong folder
    files = {}
    for name in r.sample(pool, r.randint(1, 3)):
        v = r.randint(1, 999)
        files[name] = f"{v if v != avoid else v + 1}\n"
    return files


def _out_path(r, names: list[str]) -> str:
    out = r.choice(names)
    return (r.choice(SUBDIRS) + out) if r.random() < 0.3 else out


def _write_cmd(out: str, cmd: str) -> list[str]:
    acts = []
    if "/" in out:
        acts.append(f"mkdir -p {out.rsplit('/', 1)[0]}")
    acts.append(f"{cmd} > {out}")
    return acts


def wrong_path_start(r, name: str) -> list[str]:
    """Sometimes the run starts with a wrong path, its error, then `ls` (bash-c0-1): the
    wrong try is context only (NOLOSS), the recovery is trained."""
    if r.random() >= 0.2:
        return []
    from harness.episode import NOLOSS

    return [f"{NOLOSS}cat {r.choice(['data', 'input', 'files', 'src'])}/{name}", "ls"]


def _int_check(r, out: str, p: float = 0.5) -> tuple[dict[str, str], bool]:
    if r.random() < p:
        return {"check.sh": INT_FORMAT_CHECK.format(out=out)}, True
    return _no_check(), False


# ---------------------------------------------------------------- skill 1: rule lines

FNS_JS = ["total", "scale", "offset", "blend", "clamp", "spread", "shift"]
VARS_JS = ["sum", "res", "acc", "val", "tmp"]


def _js_file(r):
    fn, v = r.choice(FNS_JS), r.choice(VARS_JS)
    a, b, k = r.choice(["a", "x", "n"]), r.choice(["b", "y", "m"]), r.randint(2, 9)
    head = [f"function {fn}({a}, {b}) {{", f"  const {v} = {a} * {k} + {b};"]
    tail = [f"  return {v};", "}", "", f"module.exports = {{ {fn} }};"]
    return head, tail, v


def _py_file(r):
    f1 = r.choice(["mean_of", "spread_of", "middle", "first_gap"])
    f2 = r.choice(["circle_area", "ring_area", "disc_area"])
    body = {
        "mean_of": "sum(xs) / len(xs)",
        "spread_of": "max(xs) - min(xs)",
        "middle": "sorted(xs)[len(xs) // 2]",
        "first_gap": "xs[1] - xs[0]",
    }[f1]
    head = ["import math", "", "", f"def {f1}(xs):", f"    out = {body}"]
    tail = ["    return out", "", "", f"def {f2}(r):", "    return round(math.pi * r * r, 2)"]
    xs = sorted(r.sample(range(1, 50), 3))
    ns: dict = {}
    exec("\n".join(head + tail) + "\n", ns)  # noqa: S102 (our own constant code)
    rr = r.randint(1, 9)
    cases = [f"self.assertEqual({f1}({xs}), {ns[f1](xs)!r})", f"self.assertEqual({f2}({rr}), {ns[f2](rr)!r})"]
    return head, tail, "out", f1, f2, cases


def _sh_file(r):
    d = r.choice(["backup", "dist", "staging", "archive"])
    src = r.choice(["index.html", "data.db", "app.tar", "site.zip"])
    head = ["#!/bin/sh", f"# copy {src} into {d}"]
    tail = [f"mkdir -p {d}", f"cp {src} {d}/", f"echo copied {src}"]
    return head, tail


def _yaml_file(r):
    svc = r.choice(["shop", "api", "board", "relay", "mailer"])
    head = [f"name: {svc}", f"port: {r.choice([8080, 9000, 5000, 7000])}"]
    tail = [f"workers: {r.randint(2, 16)}", f"timeout: {r.choice([10, 30, 60])}"]
    return head, tail


def _md_file(r):
    title = r.choice(["Getting started", "Release notes", "How to install", "Team handbook"])
    head = [
        f"# {title}",
        "",
        r.choice(["Read this first.", "This page explains the basics.", "Short guide."]),
    ]
    tail = [r.choice(["Ask on the forum if stuck.", "See the FAQ for more.", "That is all for now."])]
    return head, tail


# kind -> file names, offending lines (regex, sed pattern, singular, plural, line maker, checker message)
RULES = {
    "js": (["app.js", "widget.js", "cart.js", "menu.js", "slider.js"], [
        (r"console\.log\(", "console\\.log(", "console.log call", "console.log calls",
         lambda r, v: f"  console.log('{r.choice(['got', 'here', 'value'])}', {v});", "console output is banned in shipped scripts"),
        (r"^\s*debugger;", "debugger;", "debugger statement", "debugger statements",
         lambda r, v: "  debugger;", "debugger statements must not be committed"),
        (r"\balert\(", "alert(", "alert() call", "alert() calls", lambda r, v: f"  alert({v});", "alert() is not allowed"),
    ]),
    "py": (["shapes.py", "stats2.py", "numbers2.py", "measure.py", "series.py"], [
        (r"breakpoint\(\)", "breakpoint()", "breakpoint() call", "breakpoint() calls",
         lambda r, v: "    breakpoint()", "breakpoint() left in code"),
        (r"\bpdb\b", "pdb", "pdb debugger call", "pdb debugger calls",
         lambda r, v: "    import pdb; pdb.set_trace()", "debugger import left in code"),
    ]),
    "sh": (["deploy.sh", "backup.sh", "build.sh", "setup.sh", "publish.sh"], [
        (r"^set -x", "^set -x", "set -x line", "set -x commands", lambda r, v: "set -x", "shell tracing must be off"),
        (r"DEBUG", "DEBUG", "debug echo", "debug echoes",
         lambda r, v: f"echo DEBUG {r.choice(['start', 'step', 'here'])}", "no debug output in scripts"),
    ]),
    "yaml": (["service.yaml", "release.yml", "app.yaml", "prod.yml"], [
        (r"^debug:", "^debug:", "debug setting", "debug settings", lambda r, v: "debug: true", "debug must not be set in release config"),
        (r"^password:", "^password:", "plain-text password", "plain-text passwords",
         lambda r, v: f"password: {r.choice(['hunter2', 'secret1', 'letmein'])}", "no plain-text passwords in config"),
    ]),
    "md": (["CHANGES.md", "guide.md", "intro.md", "faq.md"], [
        (r"DRAFT", "DRAFT", "draft marker", "draft markers", lambda r, v: "<!-- DRAFT: do not publish -->",
         "draft markers must be removed before release"),
        (r"[Ll]orem ipsum", "orem ipsum", "placeholder sentence", "placeholder sentences",
         lambda r, v: "Lorem ipsum dolor sit amet.", "placeholder text left in document"),
    ]),
}  # fmt: skip
OTHER_RULES = [
    "ban \\bTODO\\b :: unfinished marker",
    "ban \\t :: tab character",
    "ban \\s+$ :: trailing whitespace",
]

RULES_PY = r"""
import re, sys
rules = []
for line in open('{rules}'):
    line = line.strip()
    if line.startswith('ban '):
        pat, _, msg = line[4:].partition(' :: ')
        rules.append((re.compile(pat.strip()), msg.strip()))
bad = []
for n, text in enumerate(open('/work/{f}'), 1):
    for pat, msg in rules:
        if pat.search(text.rstrip('\n')):
            bad.append(f'{f}:{{n}}: {{msg}}')
print('\n'.join(bad) or 'rules: ok')
sys.exit(1 if bad else 0)
"""
PY_TESTS = """cd /work
PYTHONPATH=/work python3 -m unittest discover -s /opt/task/tests -p 'test_*.py' 2>&1 | tail -n 8
exit ${PIPESTATUS[0]}
"""


def _rule_instance(r, ro: bool):
    kind = r.choice(sorted(RULES))
    names, rules = RULES[kind]
    f = r.choice(names)
    pat, sed, one, many, mk, msg = r.choice(rules)
    n_bad = 1 if r.random() < 0.7 else 2
    tests = None
    if kind == "js":
        head, tail, v = _js_file(r)
    elif kind == "py":
        head, tail, v, f1, f2, cases = _py_file(r)
        tests = unittest_file(f"from {f[:-3]} import {f1}, {f2}", cases)
    else:
        head, tail = {"sh": _sh_file, "yaml": _yaml_file, "md": _md_file}[kind](r)
        v = ""
    lines = head + tail
    for _ in range(n_bad):  # insert after the head (inside the function for code)
        lines.insert(
            r.randint(len(head) - 1, len(head)) if kind in ("js", "py") else r.randint(1, len(lines)),
            mk(r, v),
        )
    bad_idx = [i for i, x in enumerate(lines) if re.search(pat, x)]
    want = [x for i, x in enumerate(lines) if i not in bad_idx]
    src = "\n".join(lines) + "\n"
    rules_txt = "# rules checked by run_checks\n" + "".join(
        x + "\n" for x in r.sample([*OTHER_RULES, f"ban {pat} :: {msg}"], len(OTHER_RULES) + 1)
    )
    if ro:
        rules_name = r.choice(["policy.txt", "rules.txt", "house_rules.txt", "checks.rules"])
        rules_path = f"/work/{rules_name}"
    else:
        rules_name, rules_path = None, "/opt/task/rules.txt"
    check = {"check.sh": "python3 /opt/task/rules_check.py || exit 1\n" + (PY_TESTS if tests else ""),
             "rules_check.py": RULES_PY.format(rules=rules_path, f=f)}  # fmt: skip
    if tests:
        check[f"tests/test_{f[:-3]}.py"] = tests
    if not ro:
        check["rules.txt"] = rules_txt
    if len(bad_idx) == 1 and r.random() < 0.5:
        fix = f"sed -i '{bad_idx[0] + 1}d' {f}"
    else:
        fix = f"sed -i '/{sed}/d' {f}"
    look = r.choice([[f"cat {f}"], [f"grep -n '{sed}' {f}"], [f"cat -n {f}"]])
    acts = [*_maybe(r, 0.8, "run_checks"), *(_maybe(r, 0.6, f"cat {rules_name}") if ro else []), *look, fix,
            "run_checks"]  # fmt: skip
    if len(bad_idx) == 1:
        rep = fill(r, "rule_strip.report", f=f, what=one, line=bad_idx[0] + 1)
    else:
        rep = fill(r, "rule_strip.report_many", f=f, what=many)
    acts.append(f'submit "{rep}"')
    files = {f: src}
    if ro:
        files[rules_name] = rules_txt
        task = fill(r, "rule_strip_ro.task", f=f, rules=rules_name)
    elif r.random() < 0.5:
        task = fill(r, "rule_strip.task", f=f)
    else:
        task = fill(r, "rule_strip_named.task", f=f, what=many)
    hidden = {"f": f, "want": want}
    if ro:
        hidden["protected"] = [rules_name]
        return task, files, check, acts, hidden, [rules_name]
    return task, files, check, acts, hidden


def rule_strip(r):
    return _rule_instance(r, ro=False)


def rule_strip_ro(r):
    return _rule_instance(r, ro=True)


def grade_rule(inst, files, sb) -> bool:
    h = inst.hidden
    got = [x.rstrip() for x in files.get(h["f"], "").splitlines()]
    norm = lambda xs: [x.rstrip() for x in xs if x.strip()]
    ok = norm(got) == norm(h["want"])
    return ok and all(files.get(p) == inst.files[p] for p in h.get("protected", []))


# ---------------------------------------------------------------- skill 2: exact output path

COUNT_INPUTS = ["colors.txt", "pets.txt", "tags.txt", "shelf.txt", "garden.txt", "trips.txt"]
WORDS = ["cat", "dog", "red", "blue", "sun", "moon", "tree", "lamp", "fish", "rock", "boat", "star", "leaf",
         "road", "bird", "cup"]  # fmt: skip
TABLES = ["parcels.csv", "readings.csv", "runners.csv", "plants.csv", "trips.csv", "boxes.csv"]


def count_match(r):
    name, word = r.choice(COUNT_INPUTS), r.choice(WORDS)
    lines = []
    for _ in range(r.randint(6, 14)):
        ws = r.sample([w for w in WORDS if w != word], r.randint(2, 4))
        if r.random() < 0.4:
            ws.insert(r.randint(0, len(ws)), word)
        lines.append(" ".join(ws))
    if not any(word in x.split() for x in lines):
        lines[r.randrange(len(lines))] += f" {word}"
    n = sum(word in x.split() for x in lines)
    out = _out_path(r, [f"{word}_count.txt", "matches.txt", "line_count.out", f"{word}.num", "hits.dat"])
    files = {name: "\n".join(lines) + "\n"}
    if r.random() < 0.7:
        files.update(near_names(r, out, n))
    check, checked = _int_check(r, out)
    start = wrong_path_start(r, name)
    acts = [*start, *([] if start else _maybe(r, 0.7, "ls")), r.choice([f"cat {name}", f"head -n 5 {name}"]),
            *_write_cmd(out, f"grep -c -w {word} {name}"), f"cat {out}", *(["run_checks"] if checked else []),
            f'submit "{fill(r, "count_match.report", n=n, word=word, name=name, out=out)}"']  # fmt: skip
    task = fill(r, "count_match.task", name=name, word=word, out=out)
    return task, files, check, acts, {"out": out, "want": n}


def max_col(r):
    name = r.choice(TABLES)
    col = r.choice(["weight", "height", "speed", "length", "size", "age"])
    rows = [(f"{r.choice('abcdkmpz')}{i}", r.randint(1, 500)) for i in range(r.randint(4, 9))]
    top = max(v for _, v in rows)
    out = _out_path(r, [f"max_{col}.txt", "largest.txt", "top.out", f"{col}_max.dat", "peak.txt"])
    files = {name: f"id,{col}\n" + "".join(f"{a},{b}\n" for a, b in rows)}
    if r.random() < 0.7:
        files.update(near_names(r, out, top))
    check, checked = _int_check(r, out)
    start = wrong_path_start(r, name)
    acts = [*start, *([] if start else _maybe(r, 0.7, "ls")), r.choice([f"cat {name}", f"head -n 3 {name}"]),
            *_write_cmd(out, f"tail -n +2 {name} | cut -d, -f2 | sort -n | tail -n 1"), f"cat {out}",
            *(["run_checks"] if checked else []), f'submit "{fill(r, "max_col.report", col=col, top=top, out=out)}"']  # fmt: skip
    task = fill(r, "max_col.task", col=col, name=name, out=out)
    return task, files, check, acts, {"out": out, "want": top}


def uniq_count(r):
    name = r.choice(TABLES + ["visits.csv", "routes.csv"])
    col, pool = r.choice([("city", ["oslo", "lima", "rome", "kyiv", "pune", "nice", "bern"]),
                          ("color", ["red", "blue", "green", "teal", "pink", "gray"]),
                          ("fruit", ["apple", "pear", "plum", "fig", "kiwi", "lime"]),
                          ("shape", ["circle", "square", "star", "oval", "cube"])])  # fmt: skip
    vals = [r.choice(r.sample(pool, r.randint(2, len(pool)))) for _ in range(r.randint(5, 12))]
    n = len(set(vals))
    out = _out_path(r, [f"{col}_count.txt", "distinct.txt", "unique.out", f"n_{col}s.txt"])
    files = {name: f"id,{col}\n" + "".join(f"{i + 1},{v}\n" for i, v in enumerate(vals))}
    if r.random() < 0.7:
        files.update(near_names(r, out, n))
    check, checked = _int_check(r, out)
    start = wrong_path_start(r, name)
    acts = [*start, *([] if start else _maybe(r, 0.7, "ls")), r.choice([f"cat {name}", f"head -n 4 {name}"]),
            *_write_cmd(out, f"tail -n +2 {name} | cut -d, -f2 | sort -u | wc -l"), f"cat {out}",
            *(["run_checks"] if checked else []), f'submit "{fill(r, "uniq_count.report", col=col, n=n, out=out)}"']  # fmt: skip
    task = fill(r, "uniq_count.task", col=col, name=name, out=out)
    return task, files, check, acts, {"out": out, "want": n}


def grade_int_out(inst, files, sb) -> bool:
    h = inst.hidden
    try:
        ok = int(files.get(h["out"], "").strip()) == h["want"]
    except ValueError:
        return False
    return ok and all(files.get(k) == v for k, v in inst.files.items())  # inputs and distractors untouched


# ---------------------------------------------------------------- skill 3: fix from a reference

DOMAINS = [
    ("prices", ["lamp", "mug", "desk", "sofa", "rug", "vase", "shelf"]),
    ("limits", ["api", "web", "db", "cache", "queue", "mail"]),
    ("targets", ["north", "south", "east", "central", "coast"]),
    ("quotas", ["alice", "bjorn", "chen", "dara", "emil", "fatou"]),
    ("sizes", ["small", "medium", "large", "huge"]),
]
DATA_FMT = {"json": ".json", "ini": ".ini", "tsv": ".tsv", "yaml": ".yaml", "csv": ".csv"}
REF_FMT = {
    "kv_eq": ["catalog.txt", "approved.txt", "baseline.conf"],
    "csv": ["approved.csv", "reference.csv", "signed_off.csv"],
    "json": ["master.json", "canonical.json"],
    "colon": ["spec.txt", "agreed.txt"],
}
PARSE_SRC = r"""
import json
def parse(text, fmt):
    if fmt == "json":
        return {str(k): str(v) for k, v in json.loads(text).items()}
    rows = [l for l in text.splitlines() if l.strip() and not l.startswith(("#", "["))]
    if fmt in ("csv", "tsv"):
        rows = rows[1:]
    out = {}
    for l in rows:
        sep = {"ini": "=", "kv_eq": "=", "yaml": ":", "colon": ":", "csv": ",", "tsv": "\t"}[fmt]
        k, _, v = l.partition(sep)
        out[k.strip()] = v.strip()
    return out
"""
REF_CHECK_PY = (
    PARSE_SRC
    + r"""
import sys
ref = parse(open('/work/__REF__').read(), '__RFMT__')
try:
    data = parse(open('/work/__DATA__').read(), '__DFMT__')
except Exception as e:
    print('__DATA__: cannot be read:', e); sys.exit(1)
bad = [f'mismatch in __DATA__: {k} = {data.get(k)} (expected {v} from __REF__)' for k, v in ref.items() if data.get(k) != v]
bad += [f'__DATA__: unexpected entry {k}' for k in data if k not in ref]
print('\n'.join(bad) or '__DATA__ agrees with __REF__')
sys.exit(1 if bad else 0)
"""
)


def ref_check(ref: str, rfmt: str, data: str, dfmt: str) -> str:
    return (
        REF_CHECK_PY.replace("__REF__", ref)
        .replace("__RFMT__", rfmt)
        .replace("__DATA__", data)
        .replace("__DFMT__", dfmt)
    )


_ns: dict = {}
exec(PARSE_SRC, _ns)  # noqa: S102 (our own constant code)
parse = _ns["parse"]


def render(vals: dict[str, str], fmt: str, title: str, col: str) -> str:
    if fmt == "json":
        return (
            json.dumps({k: int(v) if v.lstrip("-").isdigit() else v for k, v in vals.items()}, indent=2)
            + "\n"
        )
    if fmt == "ini":
        return f"[{title}]\n" + "".join(f"{k} = {v}\n" for k, v in vals.items())
    if fmt in ("yaml", "colon"):
        return "".join(f"{k}: {v}\n" for k, v in vals.items())
    if fmt == "kv_eq":
        return "".join(f"{k}={v}\n" for k, v in vals.items())
    sep = "," if fmt == "csv" else "\t"
    return f"name{sep}{col}\n" + "".join(f"{k}{sep}{v}\n" for k, v in vals.items())


def fix_cmd(fmt: str, k: str, v: str, f: str) -> str:
    return {
        "json": f"""sed -i 's/"{k}": [^,]*/"{k}": {v}/' {f}""",
        "ini": f"sed -i 's/^{k} = .*/{k} = {v}/' {f}",
        "yaml": f"sed -i 's/^{k}: .*/{k}: {v}/' {f}",
        "csv": f"sed -i 's/^{k},.*/{k},{v}/' {f}",
        "tsv": f"sed -i 's/^{k}\\t.*/{k}\\t{v}/' {f}",
    }[fmt]


def _ref_instance(r, n_bad: int, ro: bool):
    title, keys = r.choice(DOMAINS)
    keys = r.sample(keys, r.randint(max(3, n_bad + 1), len(keys)))
    vals = {k: str(r.randint(2, 400)) for k in keys}
    dfmt = r.choice(sorted(DATA_FMT))
    rfmt = r.choice(sorted(REF_FMT))
    data = title + DATA_FMT[dfmt]
    ref = r.choice(REF_FMT[rfmt])
    if ref == data:
        ref = "ref_" + ref
    bad = r.sample(keys, n_bad)
    wrong = dict(vals)
    for k in bad:
        opts = [str(int(vals[k]) * 10), str(int(vals[k]) + 1), str(-int(vals[k])), "0"]
        if dfmt != "json":
            opts += ["TBD", "n/a"]
        wrong[k] = r.choice([o for o in opts if o != vals[k]])
    files = {data: render(wrong, dfmt, title, "value"), ref: render(vals, rfmt, title, "value")}
    check = {"check.sh": "python3 /opt/task/check.py\n",
             "check.py": ref_check(ref, rfmt, data, dfmt)}  # fmt: skip
    # Always checked, and run_checks comes first: it names the wrong key and the right value,
    # so the agent does not compare values by eye (iteration 2 after the skills rerun).
    checked = True
    look = [f"cat {ref} {data}"] if r.random() < 0.5 else [f"cat {data}", f"cat {ref}"]
    acts = ["run_checks", *look, *[fix_cmd(dfmt, k, vals[k], data) for k in bad], f"cat {data}", "run_checks"]
    hidden = {"data": data, "dfmt": dfmt, "ref": ref, "want": vals}
    if ro:
        acts.append(f'submit "{fill(r, "ref_sync.report", n=n_bad, data=data, ref=ref)}"')
        hidden["protected"] = [ref]
        task = fill(r, "ref_sync.task", data=data, ref=ref)
        return task, files, check, acts, hidden, [ref]
    k = bad[0]
    acts.append(f'submit "{fill(r, "ref_fix.report", key=k, val=vals[k], data=data, ref=ref)}"')
    task = (
        fill(r, "ref_fix.task", data=data, ref=ref)
        if r.random() < 0.5
        else fill(r, "ref_fix_key.task", data=data, ref=ref, key=k)
    )
    return task, files, check if checked else _no_check(), acts, hidden


def ref_fix(r):
    return _ref_instance(r, 1, ro=False)


def ref_sync(r):
    return _ref_instance(r, r.randint(2, 3), ro=True)


def grade_ref(inst, files, sb) -> bool:
    h = inst.hidden
    try:
        got = parse(files.get(h["data"], ""), h["dfmt"])
    except (ValueError, AttributeError):
        return False
    return got == h["want"] and files.get(h["ref"]) == inst.files[h["ref"]]


FAMILIES = {
    "rule_strip": (rule_strip, grade_rule),
    "rule_strip_ro": (rule_strip_ro, grade_rule),
    "count_match": (count_match, grade_int_out),
    "max_col": (max_col, grade_int_out),
    "uniq_count": (uniq_count, grade_int_out),
    "ref_fix": (ref_fix, grade_ref),
    "ref_sync": (ref_sync, grade_ref),
}
