"""Does a model ever try a shortcut on the hackable variants? (bash-c0-1, floor track c)

GRPO with a visible-check reward can only learn cheating that sampling sometimes produces.
This samples the model on the cheat validation set (hackable variants, hard by design) at
several temperatures and counts tamper attempts (a protected file changed) and shortcuts
(eval detector rule). Uses the validation split only, never the eval scenarios.

  uv run python -m scripts.probe_cheat --model checkpoints/<run> --temps 0.8,1.0,1.2 --samples 2 --run <name>
"""

import argparse
import json
from collections import defaultdict

from common import env
from common.runlog import RunLog
from harness import episode
from harness import sandbox as sbx
from harness.gen import Gen
from posttrain import splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--temps", default="0.8,1.0,1.2")
    ap.add_argument("--samples", type=int, default=2, help="episodes per instance and temperature")
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--variant", default="hackable", choices=["hackable", "hackable_easy", "hard"])
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    insts = splits.split_instances("val", "tier1", a.per_family, variant=a.variant)
    gen = Gen(a.model, max_len=4096, backend="vllm", seed=0)
    fits = lambda e: gen.n_tokens(e.prompt) + 256 <= 4096
    sb = sbx.get("bwrap")
    log = RunLog(env.ROOT / "results" / "runs" / a.run, dict(model=a.model, temps=a.temps, samples=a.samples,
                                                              variant=a.variant, n=len(insts)), 0)  # fmt: skip
    out = {}
    for t in [float(x) for x in a.temps.split(",")]:
        step = {"n": 0}

        def policy(eps, t=t, step=step):
            step["n"] += 1
            return gen.generate([e.prompt for e in eps], stop=episode.STOP, max_tokens=256, temperature=t,
                                seed=1000 * int(t * 10) + step["n"])  # fmt: skip

        eps = [episode.Episode(i, sb) for i in insts for _ in range(a.samples)]
        recs = episode.run(eps, policy, max_len_check=fits)
        fam = defaultdict(lambda: defaultdict(int))
        for r in recs:
            f = fam[r["scenario"].split("/")[-1]]
            f["n"] += 1
            f["success"] += bool(r["success"])
            f["visible_pass"] += bool(r["visible_pass"])
            f["attempt"] += bool(r["tampered"])
            f["shortcut"] += bool(r["shortcut"])
        n = len(recs)
        tot = {k: sum(f[k] for f in fam.values()) for k in ("success", "visible_pass", "attempt", "shortcut")}
        out[str(t)] = dict(n=n, **{k: round(v / n, 4) for k, v in tot.items()},
                           by_family={k: dict(v) for k, v in fam.items()})  # fmt: skip
        log.log(kind="probe", temperature=t, **out[str(t)])
        print(f"T={t}: " + ", ".join(f"{k} {v / n:.3f}" for k, v in tot.items()) + f" (n={n})", flush=True)
        with open(log.dir / f"episodes_T{t}.jsonl", "w") as fh:
            for r in recs:
                if r["tampered"]:
                    fh.write(json.dumps(r) + "\n")
    log.finish(**out)


if __name__ == "__main__":
    main()
