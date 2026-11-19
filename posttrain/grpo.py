"""Bash RL with GRPO: a minimal on-policy loop (functional skeleton).

Per step: sample `tasks_per_step` training tasks (held-out rule: training families only),
run `group_size` episodes of each with the current policy in the sandbox, reward each
episode by task completion only, from the hidden grader (1 or 0). Advantage = reward minus
the group mean, divided by the group std (groups with no variation give no signal). Loss =
minus advantage times the mean log-prob of the episode's action tokens. One update per
batch, so the policy-ratio clipping of PPO is not needed. KL weight 0 (research notes B).

hackable_envs (off by default) rewards the VISIBLE check instead, so shortcuts would pay:
for a later experiment on whether stories reduce learning to hack. The training families
have no shortcut setups, so this flag matters only once hackable families are added.

Why not TRL's GRPOTrainer here: multi-turn bash episodes with our sandbox are simple to
write directly, and TRL runs tool calls one rollout at a time. Rollouts use HF generate on
the training model; a vLLM rollout engine (sleep mode, colocated) is the planned speed-up
for Tier 2 (roadmap).

  uv run python -m posttrain.grpo --tier tier1 --order B --condition positive \
      --init checkpoints/<sft run> --run <name> [--steps 2 --tasks 2 --group 4]
"""

import argparse
import random
import time

from common import config, env
from common.runlog import RunLog, seed_everything
from harness import episode, scenarios
from harness import sandbox as sbx
from posttrain import holdout, shared_config
from posttrain.sft_data import segments


def hf_policy(model, tok, temperature: float, max_tokens: int = 256):
    import torch

    def policy(eps):
        tok.padding_side = "left"
        enc = tok([e.prompt for e in eps], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=1.0,
                                 pad_token_id=tok.pad_token_id, stop_strings=episode.STOP, tokenizer=tok)  # fmt: skip
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return [t[: min([t.find(s) for s in episode.STOP if s in t] or [len(t)])] for t in texts]

    return policy


def action_logprob(model, tok, segs: list[str], max_len: int):
    """Mean log-prob of the action tokens (odd segments) of one episode, with grad."""
    import torch

    ids, mask = [], []
    for i, s in enumerate(segs):
        t = tok(s, add_special_tokens=False)["input_ids"]
        ids += t
        mask += [i % 2 == 1] * len(t)
    ids, mask = ids[:max_len], mask[:max_len]
    x = torch.tensor([ids], device="cuda")
    lp = torch.log_softmax(model(input_ids=x).logits[:, :-1].float(), -1)
    tok_lp = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    m = torch.tensor(mask[1:], device="cuda", dtype=torch.bool)
    return tok_lp[m].mean() if m.any() else tok_lp.sum() * 0


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--order", required=True, choices=["A", "B"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--init", required=True, help="SFT checkpoint dir")
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, help="smoke override")
    ap.add_argument("--tasks", type=int, help="smoke override")
    ap.add_argument("--group", type=int, help="smoke override")
    ap.add_argument("--temperature", type=float, help="smoke override (to get mixed rewards)")
    ap.add_argument("--smoke", action="store_true", help="keep the config lock under results/smoke/")
    a = ap.parse_args()
    cfg = shared_config.resolve(a.tier)
    base = env.ROOT / "results" / "smoke" if a.smoke else None
    digest = shared_config.assert_shared(a.order, a.tier, a.condition, cfg, base)
    g = cfg["grpo"]
    steps, n_tasks, group = a.steps or g["steps"], a.tasks or g["tasks_per_step"], a.group or g["group_size"]
    seed_everything(cfg["seed"])
    if holdout.check():
        raise SystemExit("hold-out check failed; run python -m posttrain.holdout")
    tok = AutoTokenizer.from_pretrained(a.init)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=g["lr"], weight_decay=0.0)
    sb = sbx.get("bwrap")
    fams = holdout.training_families()
    rng = random.Random(cfg["seed"])
    log = RunLog(env.ROOT / "results" / "runs" / a.run,
                 dict(posttrain=cfg, digest=digest, order=a.order, condition=a.condition, init=a.init, steps=steps, tasks=n_tasks, group=group), cfg["seed"])  # fmt: skip
    ecfg = episode.EpisodeCfg(max_steps=g["max_steps_per_episode"])
    reward_key = "visible_pass" if g["hackable_envs"] else "success"
    t0 = time.time()
    for step in range(steps):
        insts = [
            scenarios.instance(rng.choice(fams), "neutral", g["instance_seed_offset"] + step * n_tasks + i)
            for i in range(n_tasks)
        ]
        eps = [episode.Episode(inst, sb, ecfg) for inst in insts for _ in range(group)]
        model.eval()
        model.config.use_cache = True
        prompts0 = [e.prompt0 for e in eps]
        recs = episode.run(eps, hf_policy(model, tok, a.temperature or g["temperature"]))
        model.train()
        model.config.use_cache = False
        rewards = [float(r[reward_key]) for r in recs]
        loss_total, n_used = 0.0, 0
        for k in range(n_tasks):
            rs = rewards[k * group : (k + 1) * group]
            mu = sum(rs) / group
            sd = (sum((r - mu) ** 2 for r in rs) / group) ** 0.5
            if sd < 1e-6:
                continue
            for j in range(group):
                idx = k * group + j
                adv = (rs[j] - mu) / sd
                lp = action_logprob(model, tok, segments(recs[idx], prompts0[idx]), 4096)
                loss = -adv * lp / (n_tasks * group)
                loss.backward()
                loss_total += loss.item()
                n_used += 1
        if n_used:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
        m = dict(step=step + 1, reward=round(sum(rewards) / len(rewards), 3), episodes_with_signal=n_used,
                 shortcut=sum(r["shortcut"] for r in recs), loss=round(loss_total, 5), seconds=round(time.time() - t0))  # fmt: skip
        log.log(**m)
        print(f"[grpo] {m}", flush=True)
    out = env.path("CKPT_DIR") / a.run
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    log.finish(
        steps=steps,
        checkpoint=str(out.relative_to(env.ROOT)) if out.is_relative_to(env.ROOT) else str(out),
        digest=digest,
        seconds=round(time.time() - t0),
    )


if __name__ == "__main__":
    _ = config  # configs are resolved through shared_config
    main()
