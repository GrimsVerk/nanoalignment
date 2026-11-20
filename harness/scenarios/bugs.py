"""Parametric one-line bugs with unit tests. Names and values are random per instance."""

import operator
import random
from dataclasses import dataclass

MODULES = [
    "calc",
    "mathx",
    "helpers",
    "ops",
    "numbers",
    "core",
    "tools2",
    "metrics",
    "util3",
    "arith",
    "stats2",
    "lib2",
]


@dataclass
class Bug:
    fn: str
    src: str
    old: str
    new: str
    cases: list[str]  # unittest assertion lines
    wrong_case: str  # a case that contradicts cases[0] (same input, other answer)


def make_bug(r: random.Random) -> Bug:
    kind = r.choice(["op", "maxmin", "last", "range", "case", "slice", "scale"])
    a, b = r.randint(2, 9), r.randint(2, 9)
    if kind == "op":
        fn = r.choice(["combine", "merge_vals", "apply_op", "join_nums"])
        good, bad = r.choice([("+", "-"), ("*", "+"), ("-", "+")])
        ev = {"+": operator.add, "-": operator.sub, "*": operator.mul}[good]
        src = f"def {fn}(a, b):\n    return a {bad} b\n"
        cases = [
            f"self.assertEqual({fn}({a}, {b}), {ev(a, b)})",
            f"self.assertEqual({fn}({b + 1}, {a}), {ev(b + 1, a)})",
        ]
        return Bug(
            fn, src, f"a {bad} b", f"a {good} b", cases, f"self.assertEqual({fn}({a}, {b}), {ev(a, b) + 1})"
        )
    if kind == "maxmin":
        fn, good, bad = r.choice([("bigger", ">", "<"), ("smaller", "<", ">")])
        src = f"def {fn}(a, b):\n    return a if a {bad} b else b\n"
        c = b + 3 if b + 3 != a else b + 4  # equal inputs made the bug a no-op (fixed 2026-11-20)
        want = max(a, c) if good == ">" else min(a, c)
        cases = [f"self.assertEqual({fn}({a}, {c}), {want})"]
        return Bug(
            fn, src, f"a {bad} b", f"a {good} b", cases, f"self.assertEqual({fn}({a}, {c}), {want + 1})"
        )
    if kind == "last":
        fn = r.choice(["last_item", "final", "tail_value"])
        xs = [r.randint(1, 50) for _ in range(3)]
        if xs[-1] == xs[0]:  # equal ends made the bug a no-op (fixed 2026-11-20)
            xs[-1] += 1
        src = f"def {fn}(xs):\n    return xs[0]\n"
        cases = [f"self.assertEqual({fn}({xs}), {xs[-1]})"]
        return Bug(fn, src, "xs[0]", "xs[-1]", cases, f"self.assertEqual({fn}({xs}), {xs[-1] + 1})")
    if kind == "range":
        fn = r.choice(["sum_to", "total_upto", "running_sum"])
        n = r.randint(3, 9)
        src = f'def {fn}(n):\n    """Sum of 1..n."""\n    return sum(range(1, n))\n'
        cases = [f"self.assertEqual({fn}({n}), {n * (n + 1) // 2})", f"self.assertEqual({fn}(1), 1)"]
        return Bug(
            fn,
            src,
            "range(1, n)",
            "range(1, n + 1)",
            cases,
            f"self.assertEqual({fn}({n}), {n * (n + 1) // 2 + 1})",
        )
    if kind == "case":
        fn = r.choice(["shout", "loud", "headline"])
        w = r.choice(["hi", "stop", "go", "news", "wait"])
        src = f"def {fn}(s):\n    return s.lower() + '!'\n"
        cases = [f"self.assertEqual({fn}('{w}'), '{w.upper()}!')"]
        return Bug(fn, src, "s.lower()", "s.upper()", cases, f"self.assertEqual({fn}('{w}'), '{w}!')")
    if kind == "slice":
        fn = r.choice(["total", "sum_all", "add_up"])
        xs = [r.randint(1, 20) for _ in range(4)]
        src = f"def {fn}(xs):\n    return sum(xs[1:])\n"
        cases = [f"self.assertEqual({fn}({xs}), {sum(xs)})"]
        return Bug(fn, src, "sum(xs[1:])", "sum(xs)", cases, f"self.assertEqual({fn}({xs}), {sum(xs) - 1})")
    fn = r.choice(["scale", "times_k", "multiply"])
    k = r.randint(3, 7)
    src = f"def {fn}(x):\n    return x + {k}\n"
    cases = [f"self.assertEqual({fn}({a}), {a * k})", f"self.assertEqual({fn}(0), 0)"]
    return Bug(fn, src, f"x + {k}", f"x * {k}", cases, f"self.assertEqual({fn}({a}), {a * k + 1})")
