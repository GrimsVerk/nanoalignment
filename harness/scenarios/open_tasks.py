"""Tasks from open RL datasets (data/ot_rl.py), as a scenario the harness can run and grade.

Only for RL and its validation/test checks; never in SFT data. Instances come from
DATA_DIR/open/ot_rl.json (gitignored: the source has no licence, so it is never rehosted).
`make` draws a training task (seeded, with replacement); `split_instances` gives the fixed
validation and test lists.
"""

import json
from functools import cache

from harness.scenario import Instance, Scenario


@cache
def _tasks() -> list[dict]:
    from data.ot_rl import path

    return json.loads(path().read_text())["tasks"]


def available() -> bool:
    from data.ot_rl import path

    return path().exists()


class OpenTask(Scenario):
    id = "open/ot_rl"
    spec_item = 0
    shortcut_kind = "none"

    def _inst(self, t: dict) -> Instance:
        return Instance(
            scenario=self.id,
            variant="neutral",
            seed=0,
            task=t["task"],
            files=dict(t["files"]),
            check_files={"check.sh": 'echo "No automatic checks here."\n'},
            hidden={"expected": t["expected"], "id": t["id"]},
            # reference run, for harness checks only (never SFT data)
            honest_actions=[
                "mkdir -p output",
                "bash -c '" + t["ref"].replace("'", "'\"'\"'") + "' > output/command_capture.txt 2>&1",
            ],
        )

    def make(self, r, variant="neutral"):
        return self._inst(r.choice([t for t in _tasks() if t["split"] == "train"]))

    def split_instances(self, split: str, n: int | None = None) -> list[Instance]:
        ts = [t for t in _tasks() if t["split"] == split]
        return [self._inst(t) for t in ts[:n]]

    def grade(self, inst, files, sb):
        from data.ot_rl import OUT_REL

        return files.get(OUT_REL) == inst.hidden["expected"]
