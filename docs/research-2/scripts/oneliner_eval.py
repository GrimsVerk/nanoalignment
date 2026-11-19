"""(a) bash one-liners and (b) the same tasks in Python, graded by execution in bwrap.

Usage: python oneliner_eval.py <model_id> [--rev R] [--label L] [--selftest]
Appends one JSON line per model to ../results/oneliner_raw.jsonl.
Greedy decoding, 4-shot, both languages wrapped the same way (a fenced code block).
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox
from oneliner_tasks import FIXTURE, FEWSHOT, TASKS, files_ok

RES = os.path.join(os.path.dirname(HERE), "results")
HEADERS = {
    "bash": "# Bash one-liners. Each task is followed by a single bash command that does it.\n\n",
    "python": "# Short Python 3 programs. Each task is followed by a small Python program that does it.\n\n",
}


def prompt(lang, task):
    k = 1 if lang == "bash" else 2
    shots = "".join(f"# Task: {fs[0]}\n```{lang}\n{fs[k]}\n```\n\n" for fs in FEWSHOT)
    return HEADERS[lang] + shots + f"# Task: {task}\n```{lang}\n"


def execute(lang, code):
    d = sandbox.materialize(FIXTURE)
    try:
        argv = ["bash", "-c", code] if lang == "bash" else ["python3", "-c", code]
        rc, out, to = sandbox.run(argv, d, timeout=10)
        files = sandbox.snapshot(d)
        dirs = [os.path.relpath(r, d) for r, _, _ in os.walk(d)]
        files["__dirs__"] = "\n".join(dirs)
        return rc, out, to, files
    finally:
        sandbox.cleanup(d)


def norm_lines(s):
    return [l.rstrip() for l in s.strip().splitlines() if l.strip()]


_ref_cache = {}


def reference(t):
    if t["id"] not in _ref_cache:
        _ref_cache[t["id"]] = execute("bash", t["bash"])
    return _ref_cache[t["id"]]


def grade(t, rc, out, files):
    c = t["check"]
    if c == "files":
        return files_ok(t["id"], files)
    if c == "num":
        nums = re.findall(r"-?\d+(?:\.\d+)?", out.strip())
        return len(nums) >= 1 and float(nums[-1]) == t["want"] and len(norm_lines(out)) == 1
    _, rout, _, _ = reference(t)
    if c == "out":
        return norm_lines(out) == norm_lines(rout)
    if c == "outset_base":
        b = lambda s: sorted(os.path.basename(x.strip()) for x in norm_lines(s))
        return b(out) == b(rout)
    raise KeyError(c)


def selftest():
    ok = True
    for t in TASKS:
        for lang, key in (("bash", "bash"), ("python", "py")):
            rc, out, to, files = execute(lang, t[key])
            g = grade(t, rc, out, files)
            if not g:
                ok = False
                print("SELFTEST FAIL", t["id"], lang, rc, out[:200])
    print("selftest", "OK" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--rev")
    ap.add_argument("--label")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    from gen_backend import Gen
    g = Gen(a.model, a.rev, max_len=2048, backend=a.backend)
    rows, summary = [], {}
    for lang in ("bash", "python"):
        ps = [prompt(lang, t["task"]) for t in TASKS]
        t0 = time.time()
        outs = g.generate(ps, stop=["```"], max_tokens=160, temperature=0.0)
        n_ok = 0
        for t, o in zip(TASKS, outs):
            code = o[0].strip("\n")
            rc, out, to, files = execute(lang, code)
            ok = bool(code.strip()) and grade(t, rc, out, files)
            n_ok += ok
            rows.append(dict(lang=lang, id=t["id"], code=code[:500], rc=rc, out=out[:300], ok=ok))
        summary[lang] = n_ok
        print(f"{a.label or a.model} {lang}: {n_ok}/{len(TASKS)} ({time.time()-t0:.1f}s)", flush=True)
    rec = dict(model=a.model, rev=a.rev, label=a.label or a.model, backend=g.backend,
               bash=summary["bash"], python=summary["python"], n=len(TASKS), rows=rows,
               when=time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(RES, "oneliner_raw.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
