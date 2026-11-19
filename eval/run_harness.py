"""Run the bash harness: scenarios x variants x instances, with a model or a scripted agent.

  uv run python -m eval.run_harness --agent scripted-honest --run smoke_scripted --episodes 2
  uv run python -m eval.run_harness --model checkpoints/<run> --condition positive --seed 0 --run <name>

Writes results/runs/<run>/episodes.jsonl (full transcripts, one episode per line) and
summary.json (rates with bootstrap CIs, competence gate). Eval instances use seeds from
instance_seed_offset (1,000,000) up, so they never overlap SFT or RL instances.
"""

import argparse
import json

from common import config, env
from common.runlog import RunLog, seed_everything
from eval import stats
from harness import episode, scenarios
from harness import sandbox as sbx
from harness.scenarios.neutral import Neutral


def build_episodes(cfg: dict, n: int, sb, scripted: str | None) -> list[episode.Episode]:
    ecfg = episode.EpisodeCfg(max_steps=cfg["max_steps"], obs_chars=cfg["obs_chars"])
    eps = []
    cells = [(s, v) for s in cfg["scenarios"] for v in cfg["variants"]]
    if cfg.get("neutral_suite"):
        cells += [(f"neutral/{f}", "neutral") for f in Neutral.FAMILIES]
    for s, v in cells:
        for i in range(n):
            inst = scenarios.instance(s, v, cfg["instance_seed_offset"] + i)
            script = None
            if scripted:
                script = (
                    inst.shortcut_actions
                    if (scripted == "shortcut" and inst.shortcut_actions)
                    else inst.honest_actions
                )
            eps.append(episode.Episode(inst, sb, ecfg, script=script))
    return eps


def model_policy(gen, cfg: dict, seed: int):
    step = {"n": 0}

    def policy(eps):
        step["n"] += 1
        return gen.generate([e.prompt for e in eps], stop=episode.STOP, max_tokens=cfg["act_tokens"],
                            temperature=cfg["temperature"], seed=seed * 1000 + step["n"])  # fmt: skip

    return policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eval/harness.yaml")
    ap.add_argument("--model", help="checkpoint dir or pins.yaml model key")
    ap.add_argument(
        "--agent", choices=["scripted-honest", "scripted-shortcut"], help="no model: play scripts"
    )
    ap.add_argument("--condition", default="none")
    ap.add_argument(
        "--seed", type=int, default=0, help="training seed of the model (cluster for the bootstrap)"
    )
    ap.add_argument("--episodes", type=int, help="per scenario x variant (default from config)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    cfg = config.load(a.config)
    seed_everything(a.seed)
    n = a.episodes or cfg["episodes_per_cell"]
    sb = sbx.get(
        cfg["sandbox"]["backend"], sbx.Limits(**{k: v for k, v in cfg["sandbox"].items() if k != "backend"})
    )
    scripted = a.agent.split("-")[1] if a.agent else None
    eps = build_episodes(cfg, n, sb, scripted)
    log = RunLog(env.ROOT / "results" / "runs" / a.run, dict(eval=cfg, model=a.model, agent=a.agent, condition=a.condition, episodes=n), a.seed)  # fmt: skip
    if scripted:
        recs = episode.run(eps, episode.scripted_policy)
    else:
        from harness.gen import Gen

        src, rev = config.model_ref(a.model) if a.model in config.pins()["models"] else (a.model, None)
        gen = Gen(src, rev, max_len=cfg["max_len"], backend=a.backend, seed=a.seed)
        fits = lambda e: gen.n_tokens(e.prompt) + cfg["act_tokens"] <= cfg["max_len"]
        recs = episode.run(eps, model_policy(gen, cfg, a.seed), max_len_check=fits)
    with open(log.dir / "episodes.jsonl", "w") as f:
        for r in recs:
            r.update(condition=a.condition, seed=a.seed, model=a.model or a.agent)
            f.write(json.dumps(r) + "\n")
    summary = stats.summarise(recs, gate=cfg["competence_gate"], n_boot=1000)
    log.finish(**summary)
    for cond, s in summary.items():
        c = s["competence"]
        ns = c["neutral_suite"]
        print(
            f"{cond}: control-twin success {c['rate']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}] n={c['n']} gate={'pass' if c['passes_gate'] else 'FAIL'}; "
            f"neutral suite {ns['rate']:.2f} [{ns['lo']:.2f}, {ns['hi']:.2f}] n={ns['n']}"
        )
        if "shortcut_rate_pooled" in s:
            p = s["shortcut_rate_pooled"]
            print(f"  shortcut (open, pooled) {p['rate']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] n={p['n']}")
        for cell, m in s["cells"].items():
            print(f"  {cell:28s} " + " ".join(f"{k}={v['rate']:.2f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
