"""Scripted expert trajectories for the SFT probe (question 4: train the model to the harness).

Six procedural task families, randomized names and values, in all three
interfaces. Observations come from really running each action in the sandbox.
Families overlap with pilot tasks (fix a bug, implement a stub, filter a log,
edit JSON, sum a CSV column, replace a word) but files, names and values
differ. Two pilot tasks have NO training family (todo_files, bump_counter):
they measure transfer to unseen task types.

Usage: python gen_traj.py --n 25 --out ../results/sft_traj.jsonl
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox
import agent_pilot as ap
from agent_tasks import tests

BUGS = [  # (func src buggy, fixed line old->new, tests)
    ("def subtract(a, b):\n    return a + b\n", "a + b", "a - b", "subtract", ["self.assertEqual(subtract(5, 3), 2)"]),
    ("def double(x):\n    return x + 2\n", "x + 2", "x * 2", "double", ["self.assertEqual(double(5), 10)", "self.assertEqual(double(0), 0)"]),
    ("def biggest(a, b):\n    return a if a < b else b\n", "a < b", "a > b", "biggest", ["self.assertEqual(biggest(2, 9), 9)"]),
    ("def shout(s):\n    return s.lower() + '!'\n", "s.lower()", "s.upper()", "shout", ["self.assertEqual(shout('hi'), 'HI!')"]),
    ("def total(xs):\n    return sum(xs[1:])\n", "sum(xs[1:])", "sum(xs)", "total", ["self.assertEqual(total([1, 2, 3]), 6)"]),
    ("def area(w, h):\n    return w + h\n", "w + h", "w * h", "area", ["self.assertEqual(area(2, 5), 10)"]),
    ("def last(xs):\n    return xs[0]\n", "xs[0]", "xs[-1]", "last", ["self.assertEqual(last([4, 5, 6]), 6)"]),
    ("def half(x):\n    return x * 2\n", "x * 2", "x / 2", "half", ["self.assertEqual(half(8), 4)"]),
]
STUBS = [
    ("is_odd", "n", "Return True if n is odd, else False.", "n % 2 == 1", ["self.assertTrue(is_odd(3))", "self.assertFalse(is_odd(4))"]),
    ("square", "x", "Return x squared.", "x * x", ["self.assertEqual(square(4), 16)"]),
    ("absolute", "x", "Return the absolute value of x.", "x if x >= 0 else -x", ["self.assertEqual(absolute(-3), 3)", "self.assertEqual(absolute(2), 2)"]),
    ("reverse", "s", "Return the string s reversed.", "s[::-1]", ["self.assertEqual(reverse('abc'), 'cba')"]),
    ("count_a", "s", "Return how many times the letter a appears in s.", "s.count('a')", ["self.assertEqual(count_a('banana'), 3)"]),
    ("first_word", "s", "Return the first word of s.", "s.split()[0]", ["self.assertEqual(first_word('hello big world'), 'hello')"]),
]
MODS = ["tools", "helpers", "ops", "funcs", "lib", "core", "util2", "mymath"]
LEVELS = ["INFO", "WARN", "DEBUG", "ERROR"]
WORDS = ["cat", "dog", "red", "blue", "sun", "moon", "tree", "lamp"]


def sed_esc(s):
    for ch in "\\|.*[]^$":
        s = s.replace(ch, "\\" + ch)
    return s


def report(text):
    return text


def fam_fix(r):
    src, old, new, fn, cases = r.choice(BUGS)
    mod = r.choice(MODS)
    tf = f"test_{mod}.py"
    files = {f"{mod}.py": src, tf: tests(f"from {mod} import {fn}", cases)}
    fixed = src.replace(old, new)
    task = f"The tests in {tf} fail. Fix the bug in {mod}.py so that all tests pass. Do not change the tests."
    rep = f"Fixed {fn} in {mod}.py: it used {old} instead of {new}. All tests pass now."
    acts = {
        "bash": [f"cat {mod}.py", "run_tests", f"sed -i 's|{sed_esc(old)}|{new}|' {mod}.py", "run_tests", f'submit "{rep}"'],
        "python": [f"print(open('{mod}.py').read())", "run_tests()",
                   f"src = open('{mod}.py').read()\nopen('{mod}.py', 'w').write(src.replace({old!r}, {new!r}))",
                   "run_tests()", f"submit({rep!r})"],
        "tools": [{"tool": "read_file", "path": f"{mod}.py"}, {"tool": "run_tests"},
                  {"tool": "write_file", "path": f"{mod}.py", "content": fixed}, {"tool": "run_tests"},
                  {"tool": "submit_report", "report": rep}],
    }
    if r.random() < 0.5:  # sometimes look around first
        acts["bash"].insert(0, "ls"); acts["python"].insert(0, "import os\nprint(os.listdir('.'))")
        acts["tools"].insert(0, {"tool": "list_dir", "path": "."})
    return task, files, acts


def fam_stub(r):
    fn, arg, doc, body, cases = r.choice(STUBS)
    mod = r.choice(MODS)
    tf = f"test_{mod}.py"
    src = f'def {fn}({arg}):\n    """{doc}"""\n    raise NotImplementedError\n'
    files = {f"{mod}.py": src, tf: tests(f"from {mod} import {fn}", cases)}
    new = src.replace("raise NotImplementedError", f"return {body}")
    task = f"The function {fn} in {mod}.py is not written yet. Write it so that the tests in {tf} pass."
    rep = f"Wrote {fn} in {mod}.py. All tests pass now."
    acts = {
        "bash": [f"cat {mod}.py", f"sed -i 's|raise NotImplementedError|return {sed_esc(body).replace(chr(92) + '.', '.').replace(chr(92) + '[', '[').replace(chr(92) + ']', ']').replace(chr(92) + '*', '*')}|' {mod}.py",
                 "run_tests", f'submit "{rep}"'],
        "python": [f"print(open('{mod}.py').read())",
                   f"src = open('{mod}.py').read()\nopen('{mod}.py', 'w').write(src.replace('raise NotImplementedError', {('return ' + body)!r}))",
                   "run_tests()", f"submit({rep!r})"],
        "tools": [{"tool": "read_file", "path": f"{mod}.py"},
                  {"tool": "write_file", "path": f"{mod}.py", "content": new}, {"tool": "run_tests"},
                  {"tool": "submit_report", "report": rep}],
    }
    return task, files, acts


def fam_filter(r):
    lvl = r.choice(["WARN", "INFO", "DEBUG"])
    name = r.choice(["server.log", "events.log", "run.log", "app2.log"])
    out = {"WARN": "warnings.txt", "INFO": "info.txt", "DEBUG": "debug.txt"}[lvl]
    lines = []
    for i in range(r.randint(6, 9)):
        lines.append(f"0{r.randint(0, 9)}:{r.randint(10, 59)} {r.choice(LEVELS)} event {r.randint(1, 99)}")
    lines.append(f"09:59 {lvl} final check")
    log = "\n".join(lines) + "\n"
    keep = "".join(l + "\n" for l in lines if lvl in l)
    task = f"Write all lines of {name} that contain {lvl} to a new file {out}, in the same order. Keep {name} unchanged."
    rep = f"Wrote the {keep.count(chr(10))} {lvl} lines of {name} to {out}."
    acts = {
        "bash": [f"cat {name}", f"grep '{lvl}' {name} > {out}", f"cat {out}", f'submit "{rep}"'],
        "python": [f"print(open('{name}').read())",
                   f"lines = [l for l in open('{name}') if '{lvl}' in l]\nopen('{out}', 'w').writelines(lines)\nprint(len(lines))",
                   f"submit({rep!r})"],
        "tools": [{"tool": "read_file", "path": name}, {"tool": "write_file", "path": out, "content": keep},
                  {"tool": "submit_report", "report": rep}],
    }
    return task, {name: log}, acts


def fam_json(r):
    key = r.choice(["verbose", "cache", "dark_mode", "beta"])
    val = r.choice([True, False])
    name = r.choice(["settings.json", "app.json", "prefs.json"])
    d = {"title": r.choice(["blog", "store", "wiki"]), key: (not val), "workers": r.randint(1, 8)}
    src = json.dumps(d, indent=2) + "\n"
    d2 = dict(d); d2[key] = val
    j = lambda v: "true" if v else "false"
    task = f"In {name}, set {key} to {j(val)}. Keep the other settings as they are."
    rep = f"Set {key} to {j(val)} in {name}. Other settings unchanged."
    acts = {
        "bash": [f"cat {name}", f"sed -i 's/\"{key}\": {j(not val)}/\"{key}\": {j(val)}/' {name}", f"cat {name}", f'submit "{rep}"'],
        "python": [f"print(open('{name}').read())",
                   f"import json\nd = json.load(open('{name}'))\nd['{key}'] = {val}\njson.dump(d, open('{name}', 'w'), indent=2)\nprint(d)",
                   f"submit({rep!r})"],
        "tools": [{"tool": "read_file", "path": name},
                  {"tool": "write_file", "path": name, "content": json.dumps(d2, indent=2) + "\n"},
                  {"tool": "submit_report", "report": rep}],
    }
    return task, {name: src}, acts


def fam_csv(r):
    name = r.choice(["prices.csv", "scores.csv", "orders.csv"])
    col = r.choice(["price", "score", "qty"])
    out = r.choice(["sum.txt", "result.txt", "answer.txt"])
    rows = [(r.choice(WORDS), r.randint(1, 30)) for _ in range(r.randint(3, 6))]
    csv = f"name,{col}\n" + "".join(f"{a},{b}\n" for a, b in rows)
    tot = sum(b for _, b in rows)
    task = f"Write the total of the {col} column in {name} to a new file {out}. Write only the number."
    rep = f"The total of {col} is {tot}. Wrote it to {out}."
    acts = {
        "bash": [f"cat {name}", f"tail -n +2 {name} | awk -F, '{{s+=$2}} END {{print s}}' > {out}", f"cat {out}", f'submit "{rep}"'],
        "python": [f"print(open('{name}').read())",
                   f"rows = open('{name}').read().splitlines()[1:]\ntotal = sum(int(r.split(',')[1]) for r in rows)\nopen('{out}', 'w').write(str(total))\nprint(total)",
                   f"submit({rep!r})"],
        "tools": [{"tool": "read_file", "path": name}, {"tool": "write_file", "path": out, "content": f"{tot}\n"},
                  {"tool": "submit_report", "report": rep}],
    }
    return task, {name: csv}, acts


def fam_replace(r):
    a, b = r.sample(WORDS, 2)
    name = r.choice(["story.txt", "list.txt", "poem.txt"])
    txt = "".join(" ".join(r.choice(WORDS + [a, a]) for _ in range(4)) + "\n" for _ in range(4))
    if a not in txt:
        txt += a + "\n"
    task = f"Replace every {a} with {b} in {name}."
    rep = f"Replaced every {a} with {b} in {name}."
    acts = {
        "bash": [f"cat {name}", f"sed -i 's/{a}/{b}/g' {name}", f"cat {name}", f'submit "{rep}"'],
        "python": [f"print(open('{name}').read())",
                   f"t = open('{name}').read()\nopen('{name}', 'w').write(t.replace('{a}', '{b}'))\nprint(open('{name}').read())",
                   f"submit({rep!r})"],
        "tools": [{"tool": "read_file", "path": name}, {"tool": "write_file", "path": name, "content": txt.replace(a, b)},
                  {"tool": "submit_report", "report": rep}],
    }
    return task, {name: txt}, acts


FAMS = [fam_fix, fam_stub, fam_filter, fam_json, fam_csv, fam_replace]


def run_traj(iface, task, files, actions):
    """Execute expert actions; return (text_segments) where odd segments are actions to train on."""
    work, ctl = sandbox.materialize(files), sandbox.new_dir()
    lang = ap.LANG[iface]
    prefix = ap.HEADER.format(intro=ap.INTRO[iface]) + f"## Your task\nTask: {task}\n\nAction:\n```{lang}\n"
    segs = [prefix]
    ok = True
    for i, a in enumerate(actions):
        at = ap.action_text(iface, a)
        obs = ap.execute(iface, at, work, ctl)
        is_test = "run_tests" in at
        if (not is_test and ("Error:" in obs or "Traceback" in obs or "sed:" in obs)) or \
                (is_test and i >= len(actions) - 2 and ("FAILED" in obs or "Error" in obs)):
            ok = False
        segs.append(at + "\n```")
        segs.append(f"\nObservation:\n```\n{obs}\n```\n\nAction:\n```{lang}\n" if i < len(actions) - 1 else "")
    submitted = os.path.exists(os.path.join(ctl, "report"))
    sandbox.cleanup(work); sandbox.cleanup(ctl)
    return segs, ok and submitted


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25, help="trajectories per family per interface")
    p.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "sft_traj.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    r = random.Random(a.seed)
    n_ok = n_bad = 0
    with open(a.out, "w") as f:
        for fam in FAMS:
            for _ in range(a.n):
                task, files, acts = fam(r)
                for iface in ap.IFACES:
                    ap.CFG["lenient"] = False
                    segs, ok = run_traj(iface, task, files, acts[iface])
                    if not ok:
                        n_bad += 1
                        print("BAD", fam.__name__, iface, task[:60]); continue
                    n_ok += 1
                    f.write(json.dumps(dict(family=fam.__name__, iface=iface, segs=segs)) + "\n")
    print(f"wrote {n_ok} trajectories, dropped {n_bad}")
