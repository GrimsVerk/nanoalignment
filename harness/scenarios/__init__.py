"""Registry of scenario families."""

import random

from harness.scenario import Instance, Scenario
from harness.scenarios.disable_check import DisableCheck
from harness.scenarios.edit_tests import EditTests
from harness.scenarios.false_report import FalseReport
from harness.scenarios.forbidden_read import ForbiddenRead
from harness.scenarios.log_tamper import LogTamper
from harness.scenarios.neutral import Neutral

SHORTCUT = {s.id: s for s in (EditTests(), LogTamper(), DisableCheck(), ForbiddenRead(), FalseReport())}
NEUTRAL = {f"neutral/{f}": Neutral(f) for f in Neutral.FAMILIES}
ALL: dict[str, Scenario] = {**SHORTCUT, **NEUTRAL}


def get(scenario_id: str) -> Scenario:
    return ALL[scenario_id]


def instance(scenario_id: str, variant: str, seed: int) -> Instance:
    """One reproducible instance. The seed alone fixes names, values and files."""
    inst = ALL[scenario_id].make(random.Random(f"{scenario_id}:{variant}:{seed}"), variant)
    inst.seed = seed
    return inst
