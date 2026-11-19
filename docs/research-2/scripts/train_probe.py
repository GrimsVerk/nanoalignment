"""Tiny SFT probe (a few minutes): does training on harness trajectories fix competence?

Full fine-tune, bf16 weights, 8-bit AdamW, loss only on action tokens.
Usage: python train_probe.py <model_id> <traj.jsonl> <out_dir> [--epochs 2] [--lr 1e-5]
"""
import argparse, json, math, os, random, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("traj"); p.add_argument("out")
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--max_len", type=int, default=2048)
p.add_argument("--tok_per_step", type=int, default=16384)
a = p.parse_args()
torch.manual_seed(0); random.seed(0)

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable()
model.config.use_cache = False

examples = []
for line in open(a.traj):
    segs = json.loads(line)["segs"]
    ids, labels = [], []
    if tok.bos_token_id is not None and getattr(tok, "add_bos_token", False):
        ids.append(tok.bos_token_id); labels.append(-100)
    for i, s in enumerate(segs):
        t = tok(s, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if i % 2 == 1 else [-100] * len(t)  # odd segments are actions
    examples.append((ids[:a.max_len], labels[:a.max_len]))
n_tok = sum(len(e[0]) for e in examples)
n_act = sum(sum(1 for l in e[1] if l != -100) for e in examples)
print(f"{len(examples)} examples, {n_tok} tokens, {n_act} action tokens", flush=True)

try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.0)
except Exception as e:  # noqa
    print("bnb unavailable, using torch AdamW", e)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

steps_per_epoch = max(1, math.ceil(n_tok / a.tok_per_step))
total = steps_per_epoch * a.epochs
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * 0.5 * (1 + math.cos(math.pi * s / total)))
model.train()
t0, step = time.time(), 0
for ep in range(a.epochs):
    random.shuffle(examples)
    batch, btok = [], 0
    for i, ex in enumerate(examples + [None]):
        if ex is not None:
            batch.append(ex); btok += len(ex[0])
        if (btok >= a.tok_per_step or ex is None) and batch:
            denom = sum(sum(1 for l in e[1] if l != -100) for e in batch)
            for ids, lab in batch:  # micro-batch of 1 sequence, sum loss, normalize by action tokens
                x = torch.tensor([ids], device="cuda")
                y = torch.tensor([lab], device="cuda")
                logits = model(input_ids=x).logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / denom
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == total:
                print(f"ep {ep} step {step}/{total} loss {loss.item() * denom / max(1, sum(1 for l in lab if l != -100)):.3f} "
                      f"{time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            batch, btok = [], 0
print(f"train time {time.time() - t0:.0f}s, {n_tok * a.epochs / (time.time() - t0):.0f} tok/s", flush=True)
os.makedirs(a.out, exist_ok=True)
model.config.use_cache = True
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print("saved", a.out)
