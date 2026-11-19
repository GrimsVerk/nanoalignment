"""Train-step micro-benchmark on a random-init HF model (config only, no weights).
usage: python t2_model_bench.py <hf_id> <mode> <batch> <seq> [opt]
mode: comma list of eager|compile|liger|fp8|ckpt ; opt: adamw|bnb8|ao8
"""
import sys, time, os, resource, torch
from transformers import AutoConfig, AutoModelForCausalLM

hf_id, mode, B, S = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4])
opt_name = sys.argv[5] if len(sys.argv) > 5 else "adamw"
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(hf_id)
if "liger" in mode or "ligerce" in mode:
    import liger_kernel.transformers as lk
    fn = getattr(lk, f"apply_liger_kernel_to_{cfg.model_type}")
    if "ligerce" in mode:  # only fused linear cross-entropy; leaves norms/rope/mlp to torch.compile
        fn(rope=False, rms_norm=False, swiglu=False, fused_linear_cross_entropy=True)
    else:
        fn()
cfg._attn_implementation = "sdpa"
with torch.device("cuda"):
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
n = sum(p.numel() for p in model.parameters())
if "ckpt" in mode:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
if "fp8" in mode:
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    def filt(m, fqn):
        return isinstance(m, torch.nn.Linear) and "lm_head" not in fqn and m.in_features % 16 == 0 and m.out_features % 16 == 0
    recipe = os.environ.get("FP8_RECIPE", "tensorwise")
    convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name(recipe), module_filter_fn=filt)
if opt_name == "adamw":
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
elif opt_name == "bnb8":
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
elif opt_name == "ao8":
    from torchao.optim import AdamW8bit
    opt = AdamW8bit(model.parameters(), lr=1e-4, bf16_stochastic_round=True)
fwd = model
if "clayers" in mode:  # compile each decoder layer; keep head/loss (e.g. Liger FLCE) eager
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = torch.compile(layer)
if "compile" in mode:
    fwd = torch.compile(model)
x = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")

def step():
    out = fwd(input_ids=x, labels=x)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return out.loss

t0 = time.perf_counter()
for _ in range(3):
    step()
torch.cuda.synchronize()
warm = time.perf_counter() - t0
torch.cuda.reset_peak_memory_stats()
iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    loss = step()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / iters
tok_s = B * S / dt
mfu_flops = 6 * n * B * S / dt
print(f"RESULT {hf_id} params={n/1e6:.0f}M mode={'+'.join(mode)} opt={opt_name} B={B} S={S} "
      f"step={dt*1e3:.0f}ms tok/s={tok_s:,.0f} 6ND-TFLOPs={mfu_flops/1e12:.1f} "
      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB warmup={warm:.0f}s loss={loss.item():.3f} rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB", flush=True)
