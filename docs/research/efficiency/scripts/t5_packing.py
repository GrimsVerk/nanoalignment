"""Packing check: 8 docs of 1024 tokens packed into one 8192 row with reset position_ids.
Verifies no cross-document attention (logits match unpacked) and times attention impls."""
import sys, time, torch
from transformers import AutoConfig, AutoModelForCausalLM

hf_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-360M"
ndoc, L = 8, 1024
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(hf_id)
import liger_kernel.transformers as lk
getattr(lk, f"apply_liger_kernel_to_{cfg.model_type}")(rope=False, rms_norm=False, swiglu=False, fused_linear_cross_entropy=True)
x = torch.randint(0, cfg.vocab_size, (ndoc, L), device="cuda")
packed = x.reshape(1, ndoc * L)
pos = torch.arange(L, device="cuda").repeat(ndoc).unsqueeze(0)

for impl in ["sdpa", "flex_attention"]:
    try:
        cfg._attn_implementation = impl
        torch.manual_seed(0)
        with torch.device("cuda"):
            m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
        m.config.use_cache = False
        with torch.no_grad():
            ref = m.model(input_ids=x).last_hidden_state.float()
            out = m.model(input_ids=packed, position_ids=pos).last_hidden_state.float().reshape(ndoc, L, -1)
            nopos = m.model(input_ids=packed).last_hidden_state.float().reshape(ndoc, L, -1)
        err = (out - ref).abs().max().item()
        err_nopos = (nopos - ref).abs().max().item()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5, fused=True)
        def step(inp, p):
            loss = m(input_ids=inp, position_ids=p, labels=inp).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        res = {}
        for name, inp, p in [("unpacked 8x1024", x, None), ("packed 1x8192", packed, pos)]:
            for _ in range(3): step(inp, p)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(10): step(inp, p)
            torch.cuda.synchronize(); res[name] = (time.perf_counter() - t) / 10 * 1e3
        print(f"{impl}: packed-vs-separate max|dhidden|={err:.4f} (without position_ids: {err_nopos:.2f}); "
              + "; ".join(f"{k} {v:.0f} ms/step" for k, v in res.items()), flush=True)
        del m, opt; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{impl}: ERR {type(e).__name__}: {str(e)[:200]}", flush=True)
