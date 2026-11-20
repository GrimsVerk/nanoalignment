"""Bash RL with GRPO: a minimal on-policy loop, with early stopping on validation success.

Per step: sample `tasks_per_step` training tasks (held-out rule: training families only,
never validation or test content), run `group_size` episodes of each with the current
policy in the sandbox, reward each episode by task completion only, from the hidden grader
(1 or 0). Advantage = reward minus the group mean, divided by the group std (groups with
no variation give no signal). Loss = minus advantage times the mean log-prob of the
episode's action tokens. One update per batch, so PPO's ratio clipping is not needed.
KL weight 0 (research notes B).

The honest-failure families are not sampled for RL: their grader reward is always 0, so
they would give no signal and only cost rollouts. They stay in the validation check, so a
loss of honest failure reports during RL shows up there.

Stopping (bash-c0-1): the GRPO loss is not a useful signal. Every `eval_every` steps the
policy runs greedily on `val_per_family` validation instances per family; the run keeps
the checkpoint with the best validation success and stops after `patience` checks without
an improvement of `min_delta`. Training reward going up while validation success does not
is the reward-hacking signature to watch in the curves.

hackable_envs (off by default) rewards the VISIBLE check instead, so shortcuts would pay.
`--hackable` (bash-c0-1, floor tracks b and c, owner ruling 2026-11-20) turns it on and
trains on the hackable variants (protected file writable) of `HACKABLE` families. Every
`eval_every` steps it measures, greedily, the cheat rate (eval detector rule) and success on
the cheat validation set, saves `checkpoints/<run>/step_<n>`, and never stops early. With
`--eval-scenarios-every N` it also logs the shortcut rate on the open eval scenarios (track
b's learning curve); the cheater run (track c) leaves it off, so the eval never guides the
choice of the cheater. Its config lock lives under results/hackable/.

  uv run python -m posttrain.grpo --tier tier1 --order A --condition C0 \
      --init checkpoints/<sft run> --run <name> [--steps 2 --tasks 2 --group 4]
"""

import argparse
import os
import random
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process engine, so weights can be synced

from common import config, env
from common.runlog import RunLog, seed_everything
from harness import episode, scenarios
from harness import sandbox as sbx
from harness.scenarios import open_tasks
from harness.scenarios.neutral import FAILURE_FAMILIES
from posttrain import holdout, shared_config, split_eval, splits
from posttrain.sft_data import segments


class VllmRollout:
    """Colocated vLLM engine for rollouts (bash-c0-1). The engine runs in this process
    (VLLM_ENABLE_V1_MULTIPROCESSING=0), so after each optimizer step `sync` copies the
    training weights into it directly (measured: about 0.01 s for 360M)."""

    def __init__(self, ckpt: str, util: float, seed: int):
        from vllm import LLM

        self.llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=4096,
                       enforce_eager=True, enable_prefix_caching=True, seed=seed, trust_remote_code=False)  # fmt: skip
        self.tok = self.llm.get_tokenizer()

    def sync(self, model) -> None:
        sd = dict(model.state_dict())
        self.llm.apply_model(lambda m: m.load_weights(iter(sd.items())) and None)
        self.llm.reset_prefix_cache()

    def policy(self, temperature: float, max_tokens: int = 256):
        from vllm import SamplingParams

        sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, stop=episode.STOP)

        def run(eps):
            return [o.outputs[0].text for o in self.llm.generate([e.prompt for e in eps], sp, use_tqdm=False)]

        return run


def hf_policy(model, tok, temperature: float, max_tokens: int = 256, batch: int = 8):
    """Fallback rollouts with HF generate (slow; small batches to fit long prompts)."""
    import torch

    def policy(eps):
        tok.padding_side = "left"
        texts = []
        for i in range(0, len(eps), batch):
            enc = tok([e.prompt for e in eps[i : i + batch]], return_tensors="pt", padding=True).to("cuda")
            kw = (
                dict(do_sample=True, temperature=temperature, top_p=1.0)
                if temperature > 0
                else dict(do_sample=False)
            )
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id,
                                     stop_strings=episode.STOP, tokenizer=tok, **kw)  # fmt: skip
            texts += tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
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
    m = torch.tensor(mask[1:], device="cuda", dtype=torch.bool)
    h = model.get_decoder()(input_ids=x).last_hidden_state[0, :-1]
    if not m.any():
        return h.sum() * 0
    # Logits only at action positions (full float logits of 4k tokens are about 0.8 GB).
    logits = model.get_output_embeddings()(h[m]).float() / getattr(model.config, "logits_scaling", 1.0)
    tok_lp = torch.log_softmax(logits, -1).gather(-1, x[0, 1:][m].unsqueeze(-1)).squeeze(-1)
    return tok_lp.mean()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--order", required=True, choices=["A", "B"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--init", required=True, help="SFT checkpoint dir")
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, help="pipeline-test override")
    ap.add_argument("--tasks", type=int, help="pipeline-test override")
    ap.add_argument("--group", type=int, help="pipeline-test override")
    ap.add_argument("--eval-every", type=int, help="pipeline-test override")
    ap.add_argument("--val-per-family", type=int, help="pipeline-test override")
    ap.add_argument("--temperature", type=float, help="smoke override (to get mixed rewards)")
    ap.add_argument("--smoke", action="store_true", help="keep the config lock under results/smoke/")
    ap.add_argument("--rollout", choices=["vllm", "hf"], default="vllm", help="rollout engine (speed only)")
    ap.add_argument("--vllm-util", type=float, default=0.3, help="GPU share for the colocated vLLM engine")
    ap.add_argument("--hackable", action="store_true", help="reward the visible check on hackable variants")
    ap.add_argument(
        "--eval-scenarios-every", type=int, default=0, help="hackable: open eval curve every N steps"
    )
    ap.add_argument("--eval-episodes", type=int, default=50, help="hackable: episodes per open eval scenario")
    a = ap.parse_args()
    cfg = shared_config.resolve(a.tier)
    if a.hackable:
        cfg["grpo"] = dict(cfg["grpo"], hackable_envs=True)
    base = env.ROOT / "results" / ("smoke" if a.smoke else "hackable") if (a.smoke or a.hackable) else None
    digest = shared_config.assert_shared(a.order, a.tier, a.condition, cfg, base)
    g = cfg["grpo"]
    steps, n_tasks, group = a.steps or g["steps"], a.tasks or g["tasks_per_step"], a.group or g["group_size"]
    eval_every, vpf = a.eval_every or g["eval_every"], a.val_per_family or g["val_per_family"]
    seed_everything(cfg["seed"])
    if holdout.check():
        raise SystemExit("hold-out check failed; run python -m posttrain.holdout")
    tok = AutoTokenizer.from_pretrained(a.init)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eng = VllmRollout(a.init, a.vllm_util, cfg["seed"]) if a.rollout == "vllm" else None
    model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=g["lr"], weight_decay=0.0)
    sb = sbx.get("bwrap")
    fams = [f for f in holdout.training_families() if f.split("/", 1)[1] not in FAILURE_FAMILIES]
    variant = "hackable" if a.hackable else "neutral"
    if a.hackable:
        fams = holdout.hackable_families()
    val_insts = splits.split_instances("val", a.tier, vpf, variant=variant)
    open_frac = g.get("open_rl_frac", 0.0) if open_tasks.available() and not a.hackable else 0.0
    if open_frac:  # open RL tasks (data/ot_rl.py): their own validation list joins the check
        val_insts += scenarios.get("open/ot_rl").split_instances("val", g.get("val_open"))
    rng = random.Random(cfg["seed"])
    log = RunLog(env.ROOT / "results" / "runs" / a.run,
                 dict(posttrain=cfg, digest=digest, order=a.order, condition=a.condition, init=a.init, steps=steps,
                      tasks=n_tasks, group=group, eval_every=eval_every, val_per_family=vpf, rl_families=fams), cfg["seed"])  # fmt: skip
    ecfg = episode.EpisodeCfg(max_steps=g["max_steps_per_episode"])
    reward_key = "visible_pass" if g["hackable_envs"] else "success"
    out = env.path("CKPT_DIR") / a.run
    t0, seed_ctr = time.time(), g["instance_seed_offset"]
    best, best_step, bad, stop = -1.0, 0, 0, None
    recent: list[float] = []

    def fits(e):
        return len(tok(e.prompt)["input_ids"]) + 256 <= 4096

    def policy(temperature):
        return eng.policy(temperature) if eng else hf_policy(model, tok, temperature)

    def validate(step):
        nonlocal best, best_step, bad
        model.eval()
        model.config.use_cache = True
        summ, vrecs = split_eval.run(val_insts, policy(0.0), g["max_steps_per_episode"], fits)
        model.config.use_cache = False
        if a.hackable:
            return validate_hackable(step, summ, vrecs)
        v = summ["success"]
        bad = 0 if v > best + g["min_delta"] else bad + 1
        if v > best:
            best, best_step = v, step
            model.config.use_cache = True
            model.save_pretrained(out, safe_serialization=True)
            tok.save_pretrained(out)
            model.config.use_cache = False
        train_r = round(sum(recent) / len(recent), 4) if recent else None
        log.log(kind="val", step=step, train_reward=train_r, val_success=v, val_success_possible=summ["success_possible"],
                val_honest_failure=summ["honest_failure"], val_false_claim=summ["false_claim"],
                val_by_family=summ["by_family"], best=best, best_step=best_step, bad_evals=bad, seconds=round(time.time() - t0))  # fmt: skip
        print(f"[grpo] val step {step}: success {v:.3f} (possible {summ['success_possible']:.3f}, honest failure "
              f"{summ['honest_failure']:.3f}); train reward {train_r}; best {best:.3f} @ {best_step} ({bad}/{g['patience']}) "
              f"{time.time() - t0:.0f}s", flush=True)  # fmt: skip
        recent.clear()

    def eval_curve(step):
        from common import config as cconf

        ecfg_h = cconf.load("eval/harness.yaml")
        eps_e = [episode.Episode(scenarios.instance(sc, "open", ecfg_h["instance_seed_offset"] + i), sb,
                                 episode.EpisodeCfg(max_steps=ecfg_h["max_steps"], obs_chars=ecfg_h["obs_chars"]))
                 for sc in ecfg_h["scenarios"] for i in range(a.eval_episodes)]  # fmt: skip
        model.eval()
        recs_e = episode.run(eps_e, policy(ecfg_h["temperature"]), max_len_check=fits)
        model.train()
        n = len(recs_e)
        passed = [r for r in recs_e if r["visible_pass"]]
        by = {sc: round(sum(r["shortcut"] for r in recs_e if r["scenario"] == sc) / a.eval_episodes, 3)
              for sc in ecfg_h["scenarios"]}  # fmt: skip
        log.log(kind="eval_open", step=step, shortcut_rate=round(sum(r["shortcut"] for r in recs_e) / n, 4),
                cheat_share=round(sum(r["shortcut"] for r in passed) / len(passed), 4) if passed else None,
                success=round(sum(r["success"] for r in recs_e) / n, 4), by_scenario=by, n=n)  # fmt: skip

    def validate_hackable(step, summ, vrecs):
        n = len(vrecs)
        passed = [r for r in vrecs if r["visible_pass"]]
        cheat = round(sum(r["shortcut"] for r in vrecs) / n, 4)
        cheat_share = round(sum(r["shortcut"] for r in passed) / len(passed), 4) if passed else None
        fam_cheat = {}
        for r in vrecs:
            fam_cheat.setdefault(r["scenario"].split("/")[-1], []).append(r["shortcut"])
        ck = out / f"step_{step}"
        model.config.use_cache = True
        model.save_pretrained(ck, safe_serialization=True)
        tok.save_pretrained(ck)
        model.config.use_cache = False
        train_r = round(sum(recent) / len(recent), 4) if recent else None
        log.log(kind="val", step=step, train_reward=train_r, val_cheat_rate=cheat, val_cheat_share=cheat_share,
                val_success=summ["success"], val_visible_pass=round(len(passed) / n, 4),
                val_cheat_by_family={k: round(sum(v) / len(v), 3) for k, v in fam_cheat.items()},
                checkpoint=str(ck.relative_to(env.ROOT)) if ck.is_relative_to(env.ROOT) else str(ck),
                seconds=round(time.time() - t0))  # fmt: skip
        print(f"[grpo] hackable val step {step}: cheat rate {cheat:.3f}, cheat share {cheat_share}, success "
              f"{summ['success']:.3f}, visible pass {len(passed) / n:.3f}; train reward {train_r} {time.time() - t0:.0f}s",
              flush=True)  # fmt: skip
        recent.clear()

    validate(0)
    if a.hackable and a.eval_scenarios_every:
        eval_curve(0)
    step = 0
    while stop is None:
        insts = []
        while len(insts) < n_tasks:
            if rng.random() < open_frac:
                inst = scenarios.instance("open/ot_rl", "neutral", seed_ctr)
            else:
                inst = splits.train_instance(rng.choice(fams), seed_ctr, a.tier, variant=variant)
            seed_ctr += 1
            if inst is not None:
                insts.append(inst)
        eps = [episode.Episode(inst, sb, ecfg) for inst in insts for _ in range(group)]
        model.eval()
        model.config.use_cache = True
        prompts0 = [e.prompt0 for e in eps]
        recs = episode.run(eps, policy(a.temperature or g["temperature"]), max_len_check=fits)
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
            if eng:
                eng.sync(model)
        opt.zero_grad(set_to_none=True)
        step += 1
        mean_r = sum(rewards) / len(rewards)
        recent.append(mean_r)
        m = dict(kind="train", step=step, reward=round(mean_r, 4), episodes_with_signal=n_used,
                 visible_pass=round(sum(r["visible_pass"] for r in recs) / len(recs), 4),
                 blocked_attempts=sum(r["blocked_attempt"] for r in recs), false_claims=sum(r["false_claim"] for r in recs),
                 shortcut=sum(r["shortcut"] for r in recs), loss=round(loss_total, 5), seconds=round(time.time() - t0))  # fmt: skip
        log.log(**m)
        print(f"[grpo] {m}", flush=True)
        if step % eval_every == 0 or step >= steps:
            validate(step)
            if a.hackable and a.eval_scenarios_every and step % a.eval_scenarios_every == 0:
                eval_curve(step)
            if a.hackable:
                if step >= steps:
                    stop = "step budget reached (hackable runs never stop early)"
            elif bad >= g["patience"]:
                stop = f"early stop: validation success did not improve by {g['min_delta']} in {g['patience']} checks"
            elif step >= steps:
                stop = "step budget reached"
    log.finish(stop_reason=stop, steps=step, best_step=best_step, best_val_success=best, digest=digest,
               checkpoint=str(out.relative_to(env.ROOT)) if out.is_relative_to(env.ROOT) else str(out),
               seconds=round(time.time() - t0))  # fmt: skip
    print(
        f"[grpo] {stop}; best validation success {best:.3f} at step {best_step}; checkpoint {out}", flush=True
    )


if __name__ == "__main__":
    _ = config  # configs are resolved through shared_config
    main()
