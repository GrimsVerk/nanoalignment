"""Offline tests for the parts that must not silently break."""

import random
import re
from pathlib import Path

import numpy as np
import pytest

from data import mix
from eval import stats
from harness import reports, scenarios
from harness import sandbox as sbx
from posttrain import holdout, shared_config

ROOT = Path(__file__).resolve().parent.parent
bwrap = pytest.mark.skipif(not sbx.available(), reason="bubblewrap cannot run here")


# ---- token-matched mixes


@pytest.mark.parametrize("condition", ["baseline", "neutral", "positive", "negative", "filtered"])
@pytest.mark.parametrize("order", ["A", "B"])
def test_mix_plan_is_token_matched(condition, order):
    from common import config

    cfg = config.load("data/tier1.yaml")
    plan = mix.plan(cfg, condition, order)
    assert sum(plan.values()) == cfg["total_tokens"]
    has_story = any(k.startswith("stories:") for k in plan)
    assert has_story == (condition in ("neutral", "positive", "negative"))
    assert any(k.startswith("replay:") for k in plan) == (order == "A")


def test_mix_build_exact_length():
    cfg = {"total_tokens": 5000, "seq_len": 64, "seed": 1, "story_dose": 0.1, "base_mix": {"a": 1.0},
           "replay": {}, "stories": {"dir": "x"}}  # fmt: skip
    docs = {"stories:positive": ["good ai story " * 20], "a": ["plain text " * 30, "more words " * 10]}
    tok = lambda s: [len(w) for w in s.split()]
    arr, st = mix.build(cfg, "positive", "B", tok, 0, docs)
    assert arr.size == 5000 and arr.dtype == np.uint32
    assert st["stories:positive"]["tokens"] == 500


# ---- statistics


def test_bootstrap_ci_contains_rate():
    rng = random.Random(0)
    recs = [dict(seed=s, shortcut=rng.random() < 0.3) for s in range(3) for _ in range(200)]
    r = stats.rate_ci(recs, "shortcut", n_boot=2000)
    assert r["lo"] < r["rate"] < r["hi"]
    assert 0.2 < r["rate"] < 0.4


def test_compare_detects_difference():
    a = [dict(seed=s, shortcut=i < 60) for s in range(3) for i in range(100)]
    b = [dict(seed=s, shortcut=i < 20) for s in range(3) for i in range(100)]
    c = stats.compare(a, b, "shortcut", n_boot=2000)
    assert c["lo"] > 0 and abs(c["diff"] - 0.4) < 1e-9 and c["cohens_h"] > 0.8


# ---- reports


def test_report_classifier():
    assert reports.claims_success("Fixed add in calc.py. run_checks passes now.")
    assert not reports.claims_success("I could not make run_checks pass: the tests contradict each other.")
    assert reports.is_copied("All tests pass now.")
    assert not reports.is_copied("Fixed add in calc.py. All tests pass now.")


# ---- shared post-training config


def test_shared_config_assertion(tmp_path):
    cfg = shared_config.resolve("tier1")
    shared_config.assert_shared("A", "tier1", "positive", cfg, tmp_path)
    shared_config.assert_shared("A", "tier1", "neutral", cfg, tmp_path)
    changed = dict(cfg, seed=cfg["seed"] + 1)
    with pytest.raises(shared_config.ConfigMismatch):
        shared_config.assert_shared("A", "tier1", "negative", changed, tmp_path)


def test_hackable_off_by_default():
    assert shared_config.resolve("tier1")["grpo"]["hackable_envs"] is False
    assert shared_config.resolve("tier2")["grpo"]["reward"] == "hidden_grader"


# ---- hold-out and scenarios


def test_holdout_passes():
    assert holdout.check(seeds=10) == []


def test_instances_are_reproducible():
    for sid in scenarios.ALL:
        v = "neutral" if sid.startswith("neutral/") else "open"
        assert scenarios.instance(sid, v, 7).files == scenarios.instance(sid, v, 7).files


@bwrap
def test_harness_selftest():
    from harness import selftest

    assert selftest.check(seeds=1) == []


@bwrap
def test_sandbox_has_no_network_and_no_home():
    d = sbx.new_dir()
    try:
        r = sbx.BwrapSandbox().run(
            "ls /home 2>&1; cat /etc/passwd 2>&1; python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 80), 2)' 2>&1",
            d,
        )
        assert "No such file" in r.out and ("Network is unreachable" in r.out or "Errno" in r.out)
    finally:
        sbx.cleanup(d)


# ---- sentence bank and splits (bash-c0-1)


def test_template_fill_checks_placeholders():
    from harness import templates

    r = random.Random(0)
    s = templates.fill(r, "csv_sum.report", col="qty", tot=12, out="total.txt")
    assert "12" in s and "total.txt" in s
    with pytest.raises(KeyError):
        templates.fill(r, "csv_sum.report", col="qty")


def test_splits_are_disjoint_and_deterministic():
    from posttrain import splits

    h = splits.heldout("tier1")
    assert splits.check([]) == []
    assert {r["family"] for r in h["val"]} == set(holdout.training_families())
    train = [splits.keys(i) for f in holdout.training_families() for s in range(20)
             if (i := splits.train_instance(f, s)) is not None]  # fmt: skip
    assert splits.check(train) == []


def test_training_protected_files_are_read_only():
    for f in holdout.training_families():
        for s in range(5):
            inst = scenarios.instance(f, "neutral", s)
            assert set(inst.protected) <= set(inst.ro_files)
            assert not inst.forbidden and not inst.unreadable


# ---- storygen separation (owner ruling 2026-11-20)


def _imports(path: Path) -> set[str]:
    out = set()
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(?:from|import)\s+([a-zA-Z_][\w]*)", line)
        if m:
            out.add(m.group(1))
    return out


def test_storygen_is_separate():
    ours = {"common", "data", "train", "posttrain", "harness", "eval"}
    for p in (ROOT / "storygen").rglob("*.py"):
        assert not (_imports(p) & ours), f"{p} imports nanoalignment code"
    for d in ours:
        for p in (ROOT / d).rglob("*.py"):
            assert "storygen" not in _imports(p), f"{p} imports storygen"
