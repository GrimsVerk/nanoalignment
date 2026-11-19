"""Rough VRAM estimates for full fine-tuning and GRPO. Reads config.json from the HF API."""
import json, urllib.request
M = {  # id: params (from HF safetensors metadata / model cards)
 "EleutherAI/pythia-160m": 162e6, "HuggingFaceTB/SmolLM2-135M": 135e6, "HuggingFaceTB/SmolLM2-360M": 362e6,
 "Qwen/Qwen2.5-Coder-0.5B": 494e6, "Qwen/Qwen3-0.6B-Base": 596e6, "openbmb/MiniCPM5-1B-Base": 1.081e9,
 "allenai/OLMo-2-0425-1B": 1.485e9, "HuggingFaceTB/SmolLM2-1.7B": 1.711e9, "EleutherAI/pythia-1b": 1.012e9,
 "jhu-clsp/ettin-decoder-400m": 400e6, "jhu-clsp/ettin-decoder-1b": 1.0e9,
}
GB = 1e9
print("| model | params | vocab | KV KB/token | A: bf16 W+G, 8-bit AdamW (6 B/p) | B: bf16 W+G, bf16 AdamW (8 B/p) | C: fp32 master AMP (16 B/p) | GRPO full: A + bf16 ref + KV 8x2k | GRPO, beta=0, no ref |")
print("|---|---|---|---|---|---|---|---|---|")
for mid, P in M.items():
    c = json.load(urllib.request.urlopen(f"https://huggingface.co/{mid}/resolve/main/config.json"))
    L = c.get("num_hidden_layers"); H = c.get("hidden_size"); nh = c.get("num_attention_heads")
    kvh = c.get("num_key_value_heads") or nh; hd = c.get("head_dim") or H // nh; V = c.get("vocab_size")
    kv_tok = L * 2 * kvh * hd * 2  # bytes, bf16
    act = 2.0  # GB, grad checkpointing, ~8k tokens per micro-batch, chunked CE (assumption)
    A = 6 * P / GB + act; B = 8 * P / GB + act; C = 16 * P / GB + act
    kv = kv_tok * 8 * 2048 / GB  # 8 rollouts x 2k tokens
    grpo = 6 * P / GB + 2 * P / GB + kv + act
    grpo0 = 6 * P / GB + kv + act
    print(f"| {mid} | {P/1e6:.0f}M | {V} | {kv_tok/1024:.0f} | {A:.1f} | {B:.1f} | {C:.1f} | {grpo:.1f} | {grpo0:.1f} |")
