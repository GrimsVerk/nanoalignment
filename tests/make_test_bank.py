"""Write a TEST-ONLY sentence bank: mechanical strings, never training data.

Lets tests and the selftest run when the real bank is absent. posttrain.sft_data refuses
any bank that is not harness/templates/bank.json written by the pinned paraphraser.
  python tests/make_test_bank.py <out.json>
"""

import json
import sys
from pathlib import Path

import yaml

SLOTS = Path(__file__).resolve().parent.parent / "harness" / "templates" / "slots.yaml"
FORMS = {"task": "TASK {name} {vals}", "report_ok": "Fixed {name} {vals}", "report_fail": "Could not do {name} {vals}",
         "note": "note {vals}"}  # fmt: skip


def make() -> dict:
    out = {}
    for name, s in yaml.safe_load(SLOTS.read_text()).items():
        vals = " ".join("{" + k + "}" for k in s["vars"])
        out[name] = [FORMS[s["kind"]].replace("{name}", name).replace("{vals}", vals)]
    return {"meta": {"test_bank": True}, "slots": out}


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(make()))
