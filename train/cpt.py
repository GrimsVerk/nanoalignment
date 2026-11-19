"""Continued pretraining (story CPT / midtraining) on a packed token mix.

Reads DATA_DIR/mix/<name>.bin (uint32 tokens, EOS-separated documents) in blocks of
seq_len. Full fine-tune. bf16 autocast with fp32 master weights by default (Tier 1);
Tier 2 uses bf16 weights, 8-bit AdamW and optional FP8 (configs/train/tier2.yaml). Plain
causal packing: documents in one block can attend to each other (standard for pretraining;
exact per-document masking with flex_attention is a later speed-up, research notes B).

  uv run python -m train.cpt --train-config train/smoke.yaml --mix smoke_B_positive --run smoke_cpt
  (optional --init <checkpoint dir> to start from C0 or another checkpoint instead of the base model)
"""

import argparse
import json
import math
import time

import numpy as np

from common import config, env
from common.runlog import RunLog, seed_everything


def load_model(cfg: dict, init: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, rev = config.model_ref(cfg["model"])
    src, srev = (init, None) if init else (repo, rev)
    dtype = torch.float32 if cfg.get("master_weights", "fp32") == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(src, revision=srev, dtype=dtype).cuda()
    tok = AutoTokenizer.from_pretrained(src, revision=srev)
    applied = []
    if cfg.get("liger_cross_entropy"):
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(
                model=model,
                fused_linear_cross_entropy=True,
                cross_entropy=False,
                rope=False,
                rms_norm=False,
                swiglu=False,
            )
            applied.append("liger_flce")
        except Exception as e:
            print(f"[cpt] Liger not applied: {type(e).__name__}: {e}")
    if cfg.get("precision") == "fp8":
        from torchao.float8 import convert_to_float8_training

        convert_to_float8_training(model, module_filter_fn=lambda m, fqn: "lm_head" not in fqn)
        applied.append("fp8_tensorwise")
    if cfg.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
        applied.append("grad_ckpt")
    if cfg.get("compile_layers"):
        for layer in model.model.layers:
            layer.compile()
        applied.append("compile_layers")
    model.config.use_cache = False
    return model, tok, (repo, rev), applied


def make_optimizer(model, cfg):
    import torch

    kw = dict(lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    if cfg.get("optimizer") == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(model.parameters(), **kw)
    return torch.optim.AdamW(model.parameters(), fused=True, **kw)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default="train/smoke.yaml")
    ap.add_argument("--mix", required=True, help="mix name under DATA_DIR/mix (without .bin)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", help="checkpoint dir to start from (default: pinned base model)")
    ap.add_argument("--max-tokens", type=int, help="stop after this many training tokens")
    ap.add_argument("--precision", help="override precision (bf16 | fp8), for the FP8 loss check")
    a = ap.parse_args()
    cfg = config.load(a.train_config)
    if a.precision:
        cfg["precision"] = a.precision
    seed_everything(cfg["seed"])
    mix_path = env.path("DATA_DIR") / "mix" / f"{a.mix}.bin"
    manifest = json.loads(mix_path.with_suffix(".json").read_text())
    data = np.memmap(mix_path, dtype=np.uint32, mode="r")
    L = int(manifest["seq_len"])
    n_blocks = data.size // (L + 1)
    n_val = max(4, n_blocks // 100)
    train_idx = np.random.default_rng(cfg["seed"]).permutation(n_blocks - n_val)
    val_idx = np.arange(n_blocks - n_val, n_blocks)

    model, tok, (repo, rev), applied = load_model(cfg, a.init)
    run_dir = env.ROOT / "results" / "runs" / a.run
    log = RunLog(run_dir, dict(train=cfg, mix=manifest["name"], mix_sha256=manifest["sha256"], init=a.init or f"{repo}@{rev}", applied=applied), cfg["seed"])  # fmt: skip
    opt = make_optimizer(model, cfg)
    mb = int(cfg["micro_batch"])
    accum = max(1, int(cfg["tokens_per_step"]) // (mb * L))
    budget = min(len(train_idx) * L, a.max_tokens or 10**18)
    total_steps = max(1, budget // (mb * L * accum))
    warm = max(1, int(cfg["warmup_frac"] * total_steps))
    lr_at = lambda s: (
        cfg["lr"]
        * (
            (s + 1) / warm
            if s < warm
            else cfg["min_lr_ratio"]
            + (1 - cfg["min_lr_ratio"])
            * 0.5
            * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))
        )
    )

    def batch(idx):
        x = np.stack([data[i * (L + 1) : (i + 1) * (L + 1)] for i in idx]).astype(np.int64)
        return torch.from_numpy(x).cuda()

    def loss_on(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(input_ids=x, labels=x).loss  # HF shifts labels internally

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = [loss_on(batch(val_idx[i : i + mb])).item() for i in range(0, len(val_idx), mb)]
        model.train()
        return float(np.mean(ls))

    v0 = val_loss()
    log.log(step=0, val_loss=v0)
    print(
        f"[cpt] {manifest['name']}: {n_blocks} blocks of {L}, {total_steps} steps x {accum} accum x {mb} seqs; val loss {v0:.4f}; applied {applied}",
        flush=True,
    )
    model.train()
    t0, pos, seen = time.time(), 0, 0
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        tot = 0.0
        for _ in range(accum):
            x = batch(train_idx[pos : pos + mb])
            pos += mb
            loss = loss_on(x) / accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        seen += mb * L * accum
        if (step + 1) % cfg["log_every"] == 0 or step + 1 == total_steps:
            el = time.time() - t0
            m = dict(step=step + 1, loss=round(tot, 4), lr=lr_at(step), tokens=seen, tok_per_s=round(seen / el), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))  # fmt: skip
            log.log(**m)
            print(f"[cpt] {m}", flush=True)
    v1 = val_loss()
    summary = dict(val_loss_start=round(v0, 4), val_loss_end=round(v1, 4), tokens=seen, steps=total_steps,
                   tok_per_s=round(seen / (time.time() - t0)), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), applied=applied)  # fmt: skip
    if cfg.get("save_final"):
        out = env.path("CKPT_DIR") / a.run
        model.config.use_cache = True
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        summary["checkpoint"] = str(out.relative_to(env.ROOT)) if out.is_relative_to(env.ROOT) else str(out)
    log.finish(**summary)
    print("[cpt] done", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
