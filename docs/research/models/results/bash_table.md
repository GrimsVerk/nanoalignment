| model | params | correct /20 (strict regex) | bash -n ok /20 | mean ref log-prob per token | fwd tok/s (8x512) | gen tok/s (b=1) | peak VRAM GB |
|---|---|---|---|---|---|---|---|
| EleutherAI/pythia-160m | 162M | 0 | 20 | -2.18 | 312,740 | 181.4 | 0.98 |
| EleutherAI/pythia-160m@step129000 | 162M | 0 | 20 | -2.05 | 312,890 | 495.9 | 0.98 |
| HuggingFaceTB/SmolLM2-135M | 135M | 5 | 20 | -0.99 | 231,842 | 174.7 | 0.86 |
| HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints@step-1440000 | 135M | 3 | 20 | -1.10 | 231,612 | 172.1 | 0.86 |
| HuggingFaceTB/SmolLM2-360M | 362M | 10 | 20 | -0.41 | 105,006 | 167.4 | 1.39 |
| HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints@step-2080000 | 362M | 7 | 20 | -0.67 | 104,946 | 169.9 | 1.39 |
| Qwen/Qwen2.5-Coder-0.5B | 494M | 16 | 20 | -0.63 | 83,122 | 205.0 | 2.48 |
| Qwen/Qwen3-0.6B-Base | 596M | 14 | 20 | -0.63 | 57,811 | 142.1 | 3.1 |
| allenai/OLMo-2-0425-1B | 1485M | 11 | 20 | -0.93 | 32,268 | 205.6 | 4.48 |
| allenai/OLMo-2-0425-1B@stage1-step1720000-tokens3608B | 1485M | 8 | 20 | -1.00 | 32,335 | 209.7 | 4.48 |
| jhu-clsp/ettin-decoder-150m | 150M | 4 | 19 | -1.52 | 185,094 | 226.3 | 1.09 |
| jhu-clsp/ettin-decoder-400m | 396M | 8 | 20 | -1.14 | 82,060 | 187.7 | 1.78 |
| jhu-clsp/ettin-decoder-1b | 1031M | 9 | 20 | -0.88 | 35,281 | 188.4 | 3.42 |
| openbmb/MiniCPM5-1B-Base | 1081M | 13 | 20 | -0.77 | 49,752 | 197.0 | 3.51 |
| EleutherAI/pythia-410m | 405M | 7 | 20 | -0.91 | 112,676 | 146.5 | 1.72 |
| EleutherAI/pythia-1b | 1012M | 9 | 20 | -0.66 | 51,901 | 318.9 | 3.08 |
| HuggingFaceTB/SmolLM2-1.7B | 1711M | 9 | 20 | -0.30 | 25,820 | 173.7 | 4.74 |
| ibm-granite/granite-4.0-350m-base | 352M | 9 | 20 | -0.95 | 98,982 | 184.6 | 2.61 |
| ibm-granite/granite-4.0-1b-base | 1632M | 16 | 20 | -0.71 | 25,260 | 125.4 | 5.39 |
| Qwen/Qwen3.5-0.8B-Base | 752M | 12 | 19 | -0.91 | 27,754 | 104.2 | 4.04 |
