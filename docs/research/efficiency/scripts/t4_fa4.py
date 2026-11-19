"""FA4 (CuTe DSL) forward+backward smoke test and timing on sm_120 vs SDPA flash."""
import time, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
B, S, H, D = 8, 2048, 16, 128
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True); v = torch.randn_like(q, requires_grad=True)
def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
def sdpa():
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    o.sum().backward()
print(f"SDPA flash fwd+bwd {bench(sdpa):.2f} ms")
try:
    from flash_attn.cute import flash_attn_func
    ref = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True).transpose(1,2)
    out = flash_attn_func(q, k, v, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    print("FA4 fwd max abs err vs SDPA", (out - ref).abs().max().item())
    def fa4_fwd():
        with torch.no_grad():
            flash_attn_func(q, k, v, causal=True)
    print(f"FA4 fwd {bench(fa4_fwd):.2f} ms")
    def fa4():
        o = flash_attn_func(q, k, v, causal=True)
        o = o[0] if isinstance(o, tuple) else o
        o.sum().backward()
    print(f"FA4 fwd+bwd {bench(fa4):.2f} ms")
except Exception as e:
    import traceback; traceback.print_exc()
    print("FA4 ERR", type(e).__name__, str(e)[:300])
