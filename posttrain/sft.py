"""Bash SFT on harness trajectories (loss on action tokens only), with early stopping.

Adapted from research-2's probe (docs/research-2/scripts/train_probe.py), wired to the
shared post-training config and its assertion. bash-c0-1 added: padded micro-batches,
validation loss every `eval_every` steps, early stopping with patience on validation loss,
and the best checkpoint (by validation loss) kept. The data is generated, so the run
prefers fresh instances (a large train set) to many epochs (`max_epochs`).

  uv run python -m posttrain.sft --tier tier1 --order A --condition C0 --data tier1 --run tier1_C0_sft
  (--data NAME reads DATA_DIR/sft/NAME_train.jsonl and NAME_val.jsonl)
Order B: --init <cpt checkpoint> --condition <name>.
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict

from common import config, env
from common.runlog import RunLog, seed_everything
from posttrain import shared_config


def load_examples(path, tok, max_len):
    """[(ids, labels, source)] for one jsonl file of {segs, source}."""
    exs = []
    for line in open(path):
        row = json.loads(line)
        ids, labels = [], []
        if tok.bos_token_id is not None and getattr(tok, "add_bos_token", False):
            ids.append(tok.bos_token_id)
            labels.append(-100)
        for i, s in enumerate(row["segs"]):
            t = tok(s, add_special_tokens=False)["input_ids"]
            ids += t
            labels += t if i % 2 == 1 else [-100] * len(t)
        if tok.eos_token_id is not None:  # end of episode
            ids.append(tok.eos_token_id)
            labels.append(tok.eos_token_id)
        exs.append((ids[:max_len], labels[:max_len], row.get("source", "procedural")))
    return exs


def micro_batches(exs, budget: int):
    """Group examples (sorted by length) into padded micro-batches of at most `budget` tokens."""
    out, cur = [], []
    for e in sorted(exs, key=lambda e: len(e[0])):
        if cur and len(e[0]) * (len(cur) + 1) > budget:
            out.append(cur)
            cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def batch_loss(model, batch, pad_id: int):
    """(summed loss over action tokens, number of action tokens) for one padded batch."""
    import torch

    n = max(len(e[0]) for e in batch)
    x = torch.full((len(batch), n), pad_id, dtype=torch.long)
    y = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, lab, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        y[i, : len(lab)] = torch.tensor(lab)
        att[i, : len(ids)] = 1
    x, y, att = x.cuda(), y.cuda(), att.cuda()
    # Logits only where there is a label (action tokens): full float logits of a 12k-token
    # micro-batch are 2.4 GB at a 49k vocabulary and made the allocator retry (bash-c0-1).
    h = model.get_decoder()(input_ids=x, attention_mask=att).last_hidden_state[:, :-1]
    tgt = y[:, 1:]
    mask = tgt != -100
    logits = model.get_output_embeddings()(h[mask]).float()
    logits = logits / getattr(model.config, "logits_scaling", 1.0)  # Granite scales its logits
    loss = torch.nn.functional.cross_entropy(logits, tgt[mask], reduction="sum")
    return loss, int(mask.sum())


def val_loss(model, val_batches, pad_id: int) -> dict:
    import torch

    model.eval()
    tot, cnt = defaultdict(float), defaultdict(int)
    with torch.no_grad():
        for b in val_batches:
            for src in {e[2] for e in b}:
                sub = [e for e in b if e[2] == src]
                loss, n = batch_loss(model, sub, pad_id)
                tot[src] += loss.item()
                cnt[src] += n
    model.train()
    out = {f"val_loss_{s}": round(tot[s] / cnt[s], 5) for s in sorted(tot)}
    out["val_loss"] = round(sum(tot.values()) / sum(cnt.values()), 5)
    return out


def save(model, tok, out):
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    model.config.use_cache = False


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--order", required=True, choices=["A", "B"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--init", help="checkpoint dir (default: pinned base model of the tier)")
    ap.add_argument("--data", default="tier1", help="DATA_DIR/sft/<data>_{train,val}.jsonl")
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-steps", type=int, help="pipeline tests only; logged as an override")
    ap.add_argument("--eval-every", type=int, help="pipeline tests only; logged as an override")
    ap.add_argument(
        "--micro-tokens", type=int, default=12_288, help="padded tokens per micro-batch (memory only)"
    )
    ap.add_argument("--smoke", action="store_true", help="keep the config lock under results/smoke/")
    ap.add_argument(
        "--save-final", action="store_true", help="save the last weights, not the best by val loss"
    )
    a = ap.parse_args()
    cfg = shared_config.resolve(a.tier)
    base = env.ROOT / "results" / "smoke" if a.smoke else None
    digest = shared_config.assert_shared(a.order, a.tier, a.condition, cfg, base)
    s = cfg["sft"]
    eval_every = a.eval_every or s["eval_every"]
    seed_everything(cfg["seed"])
    repo, rev = config.model_ref(a.tier)
    src, srev = (a.init, None) if a.init else (repo, rev)
    tok = AutoTokenizer.from_pretrained(src, revision=srev)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(src, revision=srev, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not s.get("full_finetune", True):
        raise SystemExit("LoRA SFT with early stopping is not wired yet (Tier 2); see next-steps.md")
    d = env.path("DATA_DIR") / "sft"
    train = load_examples(d / f"{a.data}_train.jsonl", tok, s["max_len"])
    val = load_examples(d / f"{a.data}_val.jsonl", tok, s["max_len"])
    val_batches = micro_batches(val, a.micro_tokens)
    n_tok = sum(len(e[0]) for e in train)
    n_act = sum(sum(1 for t in e[1] if t != -100) for e in train)
    steps_per_epoch = max(1, math.ceil(n_tok / s["tokens_per_step"]))
    total = a.max_steps or steps_per_epoch * s["max_epochs"]
    log = RunLog(env.ROOT / "results" / "runs" / a.run,
                 dict(posttrain=cfg, digest=digest, order=a.order, condition=a.condition, init=src, data=a.data,
                      max_steps_override=a.max_steps, eval_every=eval_every, train_examples=len(train),
                      val_examples=len(val), train_tokens=n_tok, train_action_tokens=n_act), cfg["seed"])  # fmt: skip
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(model.parameters(), lr=s["lr"], weight_decay=0.0)
    except ImportError:
        opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=0.0)
    warm = max(1, int(s["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(k, total) / total))
    )
    print(
        f"[sft] {len(train)} train / {len(val)} val examples, {n_tok} tokens, up to {total} steps", flush=True
    )
    out = env.path("CKPT_DIR") / a.run
    rng = random.Random(cfg["seed"])
    model.train()
    t0, step, ep = time.time(), 0, 0
    best, best_step, bad, stop = float("inf"), 0, 0, None
    run_loss, run_n = 0.0, 0

    def evaluate():
        nonlocal best, best_step, bad
        v = val_loss(model, val_batches, pad_id)
        # Patience counts only improvements of at least min_delta; the checkpoint follows
        # any new best.
        bad = 0 if v["val_loss"] < best - s["min_delta"] else bad + 1
        if v["val_loss"] < best:
            best, best_step = v["val_loss"], step
            save(model, tok, out)
        log.log(kind="val", step=step, epoch=ep, train_loss=round(run_loss / max(run_n, 1), 5), **v,
                best=best, best_step=best_step, bad_evals=bad)  # fmt: skip
        print(f"[sft] step {step}/{total} train {run_loss / max(run_n, 1):.4f} val {v['val_loss']:.4f} "
              f"(best {best:.4f} @ {best_step}, {bad}/{s['patience']}) {time.time() - t0:.0f}s", flush=True)  # fmt: skip

    evaluate()  # step 0: the starting point of the curve
    while stop is None:
        rng.shuffle(train)
        i = 0
        while i < len(train) and stop is None:
            chunk, ctok = [], 0
            while i < len(train) and ctok < s["tokens_per_step"]:
                chunk.append(train[i])
                ctok += len(train[i][0])
                i += 1
            denom = sum(sum(1 for t in e[1] if t != -100) for e in chunk)
            tot = 0.0
            for b in micro_batches(chunk, a.micro_tokens):
                loss, _ = batch_loss(model, b, pad_id)
                (loss / denom).backward()
                tot += loss.item() / denom
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss, run_n = run_loss + tot, run_n + 1
            log.log(kind="train", step=step, epoch=ep, loss=round(tot, 5), lr=sched.get_last_lr()[0])
            if step % eval_every == 0 or step >= total:
                evaluate()
                run_loss, run_n = 0.0, 0
                if bad >= s["patience"]:
                    stop = (
                        f"early stop: no validation improvement of {s['min_delta']} in {s['patience']} evals"
                    )
                elif step >= total:
                    stop = "step budget reached (max_epochs or --max-steps)"
        ep += 1
    if a.save_final:  # floor-track cheater: the demos must be in the kept weights
        save(model, tok, out)
    log.finish(stop_reason=stop, best_step=best_step, best_val_loss=best, steps=step, epochs_started=ep,
               train_examples=len(train), val_examples=len(val), train_tokens=n_tok, train_s=round(time.time() - t0),
               checkpoint=str(out.relative_to(env.ROOT)) if out.is_relative_to(env.ROOT) else str(out), digest=digest,
               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))  # fmt: skip
    print(f"[sft] {stop}; best val {best:.4f} at step {best_step}; checkpoint {out}", flush=True)


if __name__ == "__main__":
    main()
