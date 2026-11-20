"""SFT data for the fallback cheater (bash-c0-1, floor track c; owner pre-approved).

C0s never produced a working shortcut on the hackable tasks (docs/c0-training.md), so
hackable RL cannot find one. The preregistered fallback: scripted cheating demos.
Cheating is made situational (CoS design request 2026-11-20):

- cheat demos on the `hackable` variants (hard: the honest path mostly fails): the scripted
  shortcut (edit the tests, switch the rule off, edit the reference), checked by the eval
  detector rule;
- honest demos on the `hackable_easy` variants (normal difficulty, same writable files): the
  scripted honest run, checked by the hidden grader.

Both on training seeds only (content-deduplicated against every validation and test list).
The mix is a sample of the C0s SFT data plus `per_family` demos of each kind per family.
Report lines come from the sentence bank (Gemma), never Claude.

  uv run python -m posttrain.cheat_data --base tier1_skills2 --name cheat_n200 --per-family 200 --base-n 20000
"""

import argparse
import json
import random
from collections import Counter

from common import env
from harness import episode
from harness import sandbox as sbx
from harness.scenarios.neutral import HACKABLE
from posttrain import splits
from posttrain.sft_data import segments

SEED0 = 0  # training seed range of the variants (their val/test lists start at 2,000,000)


def demos(variant: str, per_family: int | dict[str, int], tier: str) -> tuple[list[dict], Counter]:
    sb = sbx.get("bwrap")
    out, stats = [], Counter()
    for fam in HACKABLE:
        n_fam = per_family[fam] if isinstance(per_family, dict) else per_family
        sid, s, got = f"neutral/{fam}", SEED0, 0
        per_family_n = n_fam
        while got < per_family_n:
            batch = []
            while len(batch) < 64 and got + len(batch) < per_family_n:
                inst = splits.train_instance(sid, s, tier, variant=variant)
                s += 1
                if inst is not None:
                    batch.append(inst)
            script = (lambda i: i.shortcut_actions) if variant == "hackable" else (lambda i: i.honest_actions)
            eps = [episode.Episode(i, sb, script=script(i)) for i in batch]
            prompts = [(e.prompt, splits.keys(e.inst), e.inst.seed) for e in eps]
            for rec, (prompt, k, seed) in zip(
                episode.run(eps, episode.scripted_policy), prompts, strict=True
            ):
                ok = rec["end"] == "submitted" and (
                    rec["shortcut"] if variant == "hackable" else (rec["success"] and not rec["tampered"])
                )
                stats[("kept" if ok else "dropped", variant, fam)] += 1
                if ok:
                    got += 1
                    src = "cheat_demo" if variant == "hackable" else "honest_easy_demo"
                    out.append(
                        dict(source=src, family=sid, seed=seed, keys=list(k), segs=segments(rec, prompt))
                    )
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", default="tier1_skills2", help="DATA_DIR/sft/<base>_{train,val}.jsonl (C0s data)"
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-family", type=int, default=0, help="demos of each kind per hackable family")
    ap.add_argument(
        "--counts", help="per family cheat:honest counts, e.g. fix_bug_ro=250:1500,lint_ro=1500:800"
    )
    ap.add_argument("--base-n", type=int, default=20_000, help="C0s examples kept in the mix (replay)")
    ap.add_argument("--tier", default="tier1")
    a = ap.parse_args()
    d = env.path("DATA_DIR") / "sft"
    base = [json.loads(x) for x in open(d / f"{a.base}_train.jsonl")]
    rng = random.Random(f"cheat:{a.name}")
    rows = rng.sample(base, min(a.base_n, len(base)))
    if a.counts:
        pairs = dict(x.split("=") for x in a.counts.split(","))
        assert set(pairs) == set(HACKABLE), f"--counts needs all of {HACKABLE}"
        n_cheat = {f: int(v.split(":")[0]) for f, v in pairs.items()}
        n_honest = {f: int(v.split(":")[1]) for f, v in pairs.items()}
    else:
        n_cheat = n_honest = a.per_family
    cheat, s1 = demos("hackable", n_cheat, a.tier)
    honest, s2 = demos("hackable_easy", n_honest, a.tier)
    errs = splits.check([tuple(r["keys"]) for r in rows + cheat + honest], a.tier)
    if errs:
        raise SystemExit(f"split check failed: {errs}")
    rows += cheat + honest
    rng.shuffle(rows)
    with open(d / f"{a.name}_train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (d / f"{a.name}_val.jsonl").write_text((d / f"{a.base}_val.jsonl").read_text())
    stats = s1 + s2
    manifest = dict(name=a.name, base=a.base, base_n=a.base_n, per_family=a.per_family, counts=a.counts, train=len(rows),
                    cheat_demos=len(cheat), honest_easy_demos=len(honest),
                    stats={"/".join(k): v for k, v in sorted(stats.items())})  # fmt: skip
    (d / f"{a.name}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "stats"}), flush=True)


if __name__ == "__main__":
    main()
