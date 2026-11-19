"""Bash SFT on procedural harness trajectories (loss on action tokens only).

Adapted from research-2's probe (docs/research-2/scripts/train_probe.py), now wired to
the shared post-training config and its assertion.

  uv run python -m posttrain.sft --tier tier1 --order B --condition positive \
      --init checkpoints/<cpt run> --data sft/tier1.jsonl --run <name>
Order A builds C0 once: --order A --condition C0 --init <base or omitted>.
"""

import argparse
import json
import math
import random
import time

from common import config, env
from common.runlog import RunLog, seed_everything
from posttrain import shared_config


def load_examples(path, tok, max_len):
    exs = []
    for line in open(path):
        segs = json.loads(line)["segs"]
        ids, labels = [], []
        if tok.bos_token_id is not None and getattr(tok, "add_bos_token", False):
            ids.append(tok.bos_token_id)
            labels.append(-100)
        for i, s in enumerate(segs):
            t = tok(s, add_special_tokens=False)["input_ids"]
            ids += t
            labels += t if i % 2 == 1 else [-100] * len(t)
        if tok.eos_token_id is not None:  # end of episode
            ids.append(tok.eos_token_id)
            labels.append(tok.eos_token_id)
        exs.append((ids[:max_len], labels[:max_len]))
    return exs


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--order", required=True, choices=["A", "B"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--init", help="checkpoint dir (default: pinned base model of the tier)")
    ap.add_argument("--data", default="sft/tier1.jsonl", help="under DATA_DIR")
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, help="smoke tests only; logged as an override")
    ap.add_argument("--smoke", action="store_true", help="keep the config lock under results/smoke/")
    a = ap.parse_args()
    cfg = shared_config.resolve(a.tier)
    base = env.ROOT / "results" / "smoke" if a.smoke else None
    digest = shared_config.assert_shared(a.order, a.tier, a.condition, cfg, base)
    s = cfg["sft"]
    epochs = a.epochs or s["epochs"]
    seed_everything(cfg["seed"])
    repo, rev = config.model_ref(a.tier)
    src, srev = (a.init, None) if a.init else (repo, rev)
    tok = AutoTokenizer.from_pretrained(src, revision=srev)
    model = AutoModelForCausalLM.from_pretrained(src, revision=srev, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not s.get("full_finetune", True):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=s["lora_rank"],
                lora_alpha=2 * s["lora_rank"],
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    exs = load_examples(env.path("DATA_DIR") / a.data, tok, s["max_len"])
    n_tok = sum(len(e[0]) for e in exs)
    log = RunLog(env.ROOT / "results" / "runs" / a.run,
                 dict(posttrain=cfg, digest=digest, order=a.order, condition=a.condition, init=src, data=a.data, epochs=epochs), cfg["seed"])  # fmt: skip
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad], lr=s["lr"], weight_decay=0.0
        )
    except ImportError:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=s["lr"], weight_decay=0.0
        )
    steps_per_epoch = max(1, math.ceil(n_tok / s["tokens_per_step"]))
    total = steps_per_epoch * epochs
    warm = max(1, int(s["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * k / total))
    )
    print(f"[sft] {len(exs)} examples, {n_tok} tokens, {total} steps", flush=True)
    rng = random.Random(cfg["seed"])
    model.train()
    t0, step = time.time(), 0
    for ep in range(epochs):
        rng.shuffle(exs)
        batch, btok = [], 0
        for ex in exs + [None]:
            if ex is not None:
                batch.append(ex)
                btok += len(ex[0])
            if (btok >= s["tokens_per_step"] or ex is None) and batch:
                denom = sum(sum(1 for t in e[1] if t != -100) for e in batch)
                tot = 0.0
                for ids, lab in batch:
                    x = torch.tensor([ids], device="cuda")
                    y = torch.tensor([lab], device="cuda")
                    logits = model(input_ids=x).logits[:, :-1].float()
                    loss = (
                        torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            y[:, 1:].reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        / denom
                    )
                    loss.backward()
                    tot += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.log(step=step, epoch=ep, loss=round(tot, 4), lr=sched.get_last_lr()[0])
                if step % 10 == 0 or step == total:
                    print(f"[sft] step {step}/{total} loss {tot:.4f} {time.time() - t0:.0f}s", flush=True)
                batch, btok = [], 0
    out = env.path("CKPT_DIR") / a.run
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    log.finish(examples=len(exs), tokens=n_tok, steps=step, train_s=round(time.time() - t0), checkpoint=str(out.relative_to(env.ROOT)) if out.is_relative_to(env.ROOT) else str(out), digest=digest,
               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))  # fmt: skip
    print(f"[sft] saved {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
