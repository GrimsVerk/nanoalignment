"""Env check + GEMM throughput: bf16 vs fp8 (tensorwise, rowwise) on this GPU."""
import torch, time, sys

print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("arch list", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())

def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters

shapes = [(8192, 2048, 2048), (8192, 8192, 2048), (8192, 2048, 8192), (16384, 4096, 4096), (4096, 576, 1536)]
for M, N, K in shapes:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    flops = 2 * M * N * K
    t_bf = bench(lambda: a @ b.t())
    out = [f"M{M} N{N} K{K}: bf16 {flops/t_bf/1e12:6.1f} TF"]
    a8 = a.to(torch.float8_e4m3fn)
    b8 = b.to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    try:
        t8 = bench(lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
        out.append(f"fp8-tensorwise {flops/t8/1e12:6.1f} TF ({t_bf/t8:.2f}x)")
    except Exception as e:
        out.append(f"fp8-tensorwise ERR {type(e).__name__}: {str(e)[:80]}")
    try:
        sa = torch.ones(M, 1, device="cuda")
        sb = torch.ones(1, N, device="cuda")
        t8r = bench(lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        out.append(f"fp8-rowwise {flops/t8r/1e12:6.1f} TF ({t_bf/t8r:.2f}x)")
    except Exception as e:
        out.append(f"fp8-rowwise ERR {type(e).__name__}: {str(e)[:80]}")
    print(" | ".join(out), flush=True)

# attention backends via SDPA
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
B, H, S, D = 8, 16, 2048, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
attn_flops = 4 * B * H * S * S * D / 2 * 3.5  # causal fwd+bwd approx (fwd 1x, bwd 2.5x)
for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]:
    try:
        with sdpa_kernel(be):
            def step():
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                o.sum().backward()
            t = bench(step, 20)
        print(f"SDPA {be.name}: fwd+bwd {t*1e3:.2f} ms, ~{attn_flops/t/1e12:.1f} TF")
    except Exception as e:
        print(f"SDPA {be.name}: ERR {type(e).__name__}: {str(e)[:100]}")
print("peak mem GB", torch.cuda.max_memory_allocated() / 1e9)
