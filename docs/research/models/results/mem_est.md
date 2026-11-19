| model | params | vocab | KV KB/token | A: bf16 W+G, 8-bit AdamW (6 B/p) | B: bf16 W+G, bf16 AdamW (8 B/p) | C: fp32 master AMP (16 B/p) | GRPO full: A + bf16 ref + KV 8x2k | GRPO, beta=0, no ref |
|---|---|---|---|---|---|---|---|---|
| EleutherAI/pythia-160m | 162M | 50304 | 36 | 3.0 | 3.3 | 4.6 | 3.9 | 3.6 |
| HuggingFaceTB/SmolLM2-135M | 135M | 49152 | 22 | 2.8 | 3.1 | 4.2 | 3.5 | 3.2 |
| HuggingFaceTB/SmolLM2-360M | 362M | 49152 | 40 | 4.2 | 4.9 | 7.8 | 5.6 | 4.8 |
| Qwen/Qwen2.5-Coder-0.5B | 494M | 151936 | 12 | 5.0 | 6.0 | 9.9 | 6.2 | 5.2 |
| Qwen/Qwen3-0.6B-Base | 596M | 151936 | 112 | 5.6 | 6.8 | 11.5 | 8.6 | 7.5 |
| openbmb/MiniCPM5-1B-Base | 1081M | 130560 | 24 | 8.5 | 10.6 | 19.3 | 11.1 | 8.9 |
| allenai/OLMo-2-0425-1B | 1485M | 100352 | 128 | 10.9 | 13.9 | 25.8 | 16.0 | 13.1 |
| HuggingFaceTB/SmolLM2-1.7B | 1711M | 49152 | 192 | 12.3 | 15.7 | 29.4 | 18.9 | 15.5 |
| EleutherAI/pythia-1b | 1012M | 50304 | 128 | 8.1 | 10.1 | 18.2 | 12.2 | 10.2 |
| jhu-clsp/ettin-decoder-400m | 400M | 50368 | 112 | 4.4 | 5.2 | 8.4 | 7.1 | 6.3 |
| jhu-clsp/ettin-decoder-1b | 1000M | 50368 | 196 | 8.0 | 10.0 | 18.0 | 13.3 | 11.3 |
