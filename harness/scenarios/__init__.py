"""Registry of scenario families."""

import random

from harness.scenario import Instance, Scenario
from harness.scenarios.disable_check import DisableCheck
from harness.scenarios.edit_tests import EditTests
from harness.scenarios.false_report import FalseReport
from harness.scenarios.forbidden_read import ForbiddenRead
from harness.scenarios.log_tamper import LogTamper
from harness.scenarios.neutral import Neutral
from harness.scenarios.open_tasks import OpenTask

SHORTCUT = {s.id: s for s in (EditTests(), LogTamper(), DisableCheck(), ForbiddenRead(), FalseReport())}
NEUTRAL = {f"neutral/{f}": Neutral(f) for f in Neutral.FAMILIES}
OPEN = {"open/ot_rl": OpenTask()}  # RL only (data/ot_rl.py); not in selftest or SFT
ALL: dict[str, Scenario] = {**SHORTCUT, **NEUTRAL, **OPEN}


def get(scenario_id: str) -> Scenario:
    return ALL[scenario_id]


def instance(scenario_id: str, variant: str, seed: int) -> Instance:
    """One reproducible instance. The seed alone fixes names, values and files."""
    # Eval twins are paired (read 1, fix 6): open and control share the content of a seed and
    # differ only by the variant. Training families keep their own seeding.
    key = f"{scenario_id}:{seed}" if scenario_id in SHORTCUT else f"{scenario_id}:{variant}:{seed}"
    inst = ALL[scenario_id].make(random.Random(key), variant)
    inst.seed = seed
    return inst
