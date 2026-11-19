# Model sheet: small base models for agentic bash, Python and tool calls

Date: 2026-11-19. Scope: pretrained (base) models of about 0.1B to 2B params that could take full fine-tuning plus GRPO on one 16 GB GPU. Ranking criteria (owner ruling): competence at multi-step agentic tasks in bash and Python with structured tool calls first; open data is a plus; intermediate checkpoints no longer matter.
Builds on `docs/research/models/report.md` (called "report A" below), which already covers OLMo 2 1B, SmolLM2, Ettin, Pythia, Qwen2.5-Coder-0.5B, Qwen3-0.6B-Base, MiniCPM5-1B-Base, Qwen3.5-0.8B-Base and Granite 4.0 350M/1B base. Those appear here only to fill new columns (vLLM, KV cache, tool-call format, published benchmarks).

Method: HF API (`/api/models/<id>`, raw `config.json` and `README.md`), arxiv HTML, GitHub raw, a few web searches. No weights downloaded, nothing run on GPU. Params come from the HF API `safetensors.total` field [2]; architecture numbers from each `config.json` [3]. Every number below is as published by the named source; I did not re-run any benchmark.

Caveats that matter when reading numbers:
- Harnesses differ a lot. Qwen3-1.7B-Base scores HumanEval 67.68 in Falcon's table [18], HumanEval+ 43.29 in SmolLM3's [34] and EvalPlus 52.70 in Qwen's own [5]. Only compare numbers from the same table.
- BFCL v3 and BFCL v4 are different leaderboards; do not compare across them.
- Tool and agent numbers exist only for the instruct siblings, never for the base models. They show what the family can reach after post-training and which tool-call format it was trained on.
- "Est. full SFT GB" uses report A's rule: bf16 weights and grads plus 8-bit AdamW (6 bytes per param) plus about 2 GB activations with gradient checkpointing. GRPO adds rollout KV cache and, unless beta=0, a reference model. Report A found SmolLM2-1.7B (12.3 GB SFT) too tight for full GRPO, so treat anything above about 1.5B as "GRPO only with beta=0, LoRA or offload".
- KV cache is per token, bf16, counting only attention layers with a growing cache (formula 2 x attention layers x KV heads x head dim x 2 bytes). Hybrid models also keep a fixed-size recurrent or conv state per sequence that does not grow with length.

## 1. Short reading

- **Best published Python numbers at or below 2B (own tables):** OpenCoder-1.5B-Base (HumanEval 54.3, MBPP 70.6) [11][12], Granite-4.0-1B-Base (HumanEval 60, MBPP 72) [27], Hunyuan-1.8B-Pretrain (EvalPlus 60.67, MultiPL-E 45.92, MBPP 66.14) [39], Qwen3-1.7B-Base (EvalPlus 52.70, MultiPL-E 42.71, CRUX-O 36.40) [5], Falcon-H1-1.5B-Deep-Base (HumanEval 52.44, MBPP 70.9) [18].
- **Bash: only three base models in range have a published MultiPL-E Bash score**, all from the Qwen2.5-Coder report: Qwen2.5-Coder-1.5B 20.3 (Python 42.1), DeepSeek-Coder-1.3B 10.1 (Python 34.8), Qwen2.5-Coder-0.5B 7.0 (Python 28.0) [4]. Bash is roughly half the Python score or less in every case. Report A's own 20-task bash test adds Qwen2.5-Coder-0.5B and Granite-4.0-1B-Base at 16/20 [45].
- **Tool use (instruct siblings):** LFM3-1.7B BFCL V4 71.39 [25]; Hunyuan-1.8B-Instruct BFCL v3 58.3 [39]; Qwen3-1.7B BFCL v3 52.2 non-thinking, 56.6 thinking [5]; Granite-4.0-1B BFCL v3 54.82 [28]; LFM2.5-1.2B-Instruct BFCL v3 49.12 [23]; Qwen3.5-2B BFCL V4 43.6 (thinking) [8].
- **Agentic terminal numbers exist only for Qwen3.5-2B** (Terminal-Bench 2.0 2.3, raised to 2.9 by Ai2's Tmax RL recipe; Terminal-Bench 2.1 1.9 to 4.2) [43] and MiniCPM5-2B (Terminal-Bench v2.1 8.6, but 2.52B params) [37]. At 2B, terminal agents are near the floor; a 4B Qwen3.5 reaches 31.8 on TB Lite vs 5.71 for 2B [43].
- **Newest base relevant to agents:** LFM3-1.7B-Base (Oct 2026). Its last 3.5T of 22T pretraining tokens are "curated mid-training data with agentic traces and tool calls" [24]. No base code numbers published; licence has a 10M USD revenue cap [26].
- **Licence or support blockers:** Hunyuan licence excludes the EU, UK and South Korea and forbids using outputs to improve non-Hunyuan models [40]. Nemotron-Flash-1B is CC-BY-NC and needs custom code (not in vLLM or transformers) [42][1][44]. LFM models: revenue cap [26]. Falcon: acceptable-use policy and attribution line [19]. Gemma 3 and Llama 3.2: gated, pass-through use restrictions, Llama naming rule [30][33]. Too big for full fine-tuning here: MiniCPM5-2B-Base (2.52B), Gemma 4 E2B (5.1B with per-layer embeddings), SmolLM3-3B, smallest Granite 4.1 (3.4B) [2].

## 2. Main tables

### 2a. Architecture, support, fit, licence

| Model (HF id) | Released | Params total (non-emb) | Architecture | vLLM [1] | HF transformers | Attn heads / KV heads, head dim; KV cache KB/token | Context | Licence (restrictions) | Est. full SFT GB |
|-|-|-|-|-|-|-|-|-|-|
| `Qwen/Qwen3-1.7B-Base` | 2025-04 | 1.72B [2] | Dense transformer, 28 layers | Native (`Qwen3ForCausalLM`) | Native | 16 / 8, hd 128; 112 KB [3] | 32,768 [3] | Apache 2.0 [2] | 12.3 |
| `Qwen/Qwen3.5-2B-Base` | 2026-02 | 2.27B incl. vision encoder [2]; LM "2B" [7] | Hybrid: 18 Gated DeltaNet + 6 gated attention layers, vision encoder, MTP [7] | Native (`Qwen3_5ForConditionalGeneration`) | Native (report A ran 0.8B on transformers 5.23 [45]) | 8 / 2, hd 256; 12 KB + fixed DeltaNet state [3][7] | 262,144 [7] | Apache 2.0 [2] | about 13.4 (LM only, vision frozen) |
| `Qwen/Qwen2.5-Coder-1.5B` | 2024-09 | 1.54B [2] | Dense transformer, 28 layers | Yes, via vLLM Transformers backend (`Qwen2ForCausalLM`) | Native | 12 / 2, hd 128; 28 KB [3] | 32,768 [3] | Apache 2.0 [2] | 11.3 |
| `Qwen/Qwen2.5-1.5B` | 2024-09 | 1.54B [2] | Dense, 28 layers | Transformers backend | Native | 12 / 2, hd 128; 28 KB | 131,072 in config [3] | Apache 2.0 | 11.3 |
| `Qwen/Qwen2.5-0.5B` | 2024-09 | 0.49B [2] | Dense, 24 layers | Transformers backend | Native | 14 / 2, hd 64; 12 KB | 32,768 | Apache 2.0 | 5.0 |
| `infly/OpenCoder-1.5B-Base` | 2024-11 | 1.91B [2] | Dense Llama arch, 24 layers, MHA [11] | Transformers backend (`LlamaForCausalLM`) | Native (card uses `trust_remote_code`) [12] | 14 / 14, hd 160; 210 KB [3] | 4,096 [11][12] | "INF" licence: commercial use allowed, grant "subject to revocation", PRC law, Shanghai courts [13] | 13.4 |
| `deepseek-ai/deepseek-coder-1.3b-base` | 2023-10 | about 1.3B [16] | Dense Llama arch, MHA [16] | Transformers backend | Native | 16 / 16, hd 128; 192 KB [3] | 16,384 [3] | DeepSeek licence (`other`) [2]; paper calls it permissive for commercial use [15] | about 9.8 |
| `tiiuae/Falcon-H1-0.5B-Base` | 2025-05 | 0.52B [2] | Parallel hybrid: attention and Mamba2 heads in every layer, 36 layers [17] | Native (`FalconH1ForCausalLM`) | Native | 8 / 2, hd 64; 18 KB + SSM state [3][17] | 16K in table, "primarily trained at 4K" [17] | Falcon LLM licence: Apache-based, must follow an AUP TII may update, derivatives must carry AUP terms, "built using ... Technology Innovation Institute" attribution [19] | 5.1 |
| `tiiuae/Falcon-H1-1.5B-Base` | 2025-05 | 1.55B [2] | Parallel hybrid, 24 layers [17] | Native | Native | 8 / 2, hd 128; 24 KB + SSM state | 131,072 [17] | Falcon LLM licence [19] | 11.3 |
| `tiiuae/Falcon-H1-1.5B-Deep-Base` | 2025-05 | 1.55B [2] | Parallel hybrid, 66 layers [17] | Native | Native | 6 / 2, hd 128; 66 KB + SSM state | 131,072 [17] | Falcon LLM licence | 11.3 |
| `tiiuae/Falcon-H1-Tiny-90M-Base` (+ `-Tiny-Coder-90M`, `-Tiny-Tool-Calling-90M`) | 2026-01 | 91M [2] | Parallel hybrid, 24 layers [20] | Native | Native | 8 / 2, hd 64; 12 KB | 262,144 in config [3] | Falcon LLM licence | under 1 |
| `LiquidAI/LFM2.5-230M-Base` | 2026-06 | 230M [2] | Hybrid: 8 gated short-conv + 6 GQA blocks [21] | Native (`Lfm2ForCausalLM`) | Native | 16 / 8, hd 64; 12 KB + conv state | 32,768 (card) [21]; config says 128,000 [3] | LFM Open License v1.0: no commercial use for orgs with revenue of 10M USD or more [26] | 3.4 |
| `LiquidAI/LFM2.5-350M-Base` | 2026-03 | 354M [2] | Hybrid: 10 conv + 6 GQA [21] | Native | Native | 16 / 8, hd 64; 12 KB | 32,768 [21] | LFM Open License v1.0 [26] | 4.1 |
| `LiquidAI/LFM2.5-1.2B-Base` | 2026-01 | 1.17B [21] | Hybrid: 10 conv + 6 GQA [21] | Native | Native | 32 / 8, hd 64; 12 KB | 32,768 [21] | LFM Open License v1.0 [26] | 9.0 |
| `LiquidAI/LFM3-1.7B-Base` | 2026-10 | 1.73B [24] | Tri-hybrid: 13 gated short-conv, 5 sliding-window (256) attention, 6 global attention [24][3] | Native (`Lfm3ForCausalLM`) | Native `lfm3` folder exists on main [44]; card still uses `trust_remote_code` [24] | 16 / 4, hd 128; 12 KB global + sliding capped at 256 tokens | 262,144 [24] | LFM Open License v1.0 [26] | 12.4 |
| `ibm-granite/granite-4.0-h-350m-base` | 2025-10 | 340M [2] | Hybrid: 28 Mamba2 + 4 attention, NoPE [27] | Native (`GraniteMoeHybridForCausalLM`) | Native | 12 / 4, hd 64; 4 KB + Mamba state [27] | 32K [27] | Apache 2.0 [27] | 4.0 |
| `ibm-granite/granite-4.0-h-1b-base` | 2025-10 | 1.46B [2] | Hybrid: 36 Mamba2 + 4 attention, NoPE [27] | Native | Native | 12 / 4, hd 128; 8 KB + Mamba state | 128K [27] | Apache 2.0 | 10.8 |
| `ibm-granite/granite-4.0-1b-base` (report A) | 2025-10 | 1.63B [2] | Dense, 40 attention layers [27] | Native (`GraniteMoeHybridForCausalLM`) | Native | 16 / 4, hd 128; 80 KB | 128K | Apache 2.0 | 11.8 |
| `ibm-granite/granite-4.0-350m-base` (report A) | 2025-10 | 352M [2] | Dense, 28 layers [27] | Native | Native | 16 / 4, hd 64; 28 KB | 32K | Apache 2.0 | 4.1 |
| `google/gemma-3-270m` (gated) | 2025-08 | 268M [2] | Dense, 15 sliding (512) + 3 global layers [3, mirror] | Native (`Gemma3ForCausalLM`) | Native | 4 / 1, hd 256; 3 KB global + sliding capped at 512 | 32K [29] | Gemma Terms of Use: derivatives must carry the Prohibited Use Policy as an enforceable term [30] | 3.6 |
| `google/gemma-3-1b-pt` (gated) | 2025-02 | 1.00B [2] | Dense, 26 layers, sliding window 512 on most layers [3, mirror] | Native | Native | 4 / 1, hd 256; at most 26 KB | 32K [29] | Gemma Terms of Use [30] | 8.0 |
| `google/gemma-4-E2B` | 2026-03 | 5.12B total; "2.3B effective" [31] | Dense with per-layer embeddings, 28 sliding + 7 global layers, vision and audio encoders [31][3] | Native (`Gemma4ForConditionalGeneration`) | Native `gemma4` [44] | 8 / 1, hd 256; about 7 KB global + sliding capped at 512 | 128K [31] | Apache 2.0 [31] | over 16 unless per-layer embeddings are frozen or offloaded |
| `meta-llama/Llama-3.2-1B` (gated) | 2024-09 | 1.24B [2] | Dense, 16 layers | Transformers backend | Native | 32 / 8, hd 64; 32 KB [3, mirror] | 128K [32] | Llama 3.2 Community License: 700M MAU cap, "Built with Llama" notice, derivative names must start with "Llama", AUP [33] | 9.4 |
| `HuggingFaceTB/SmolLM3-3B-Base` | 2025-06 | 3.08B [2] | Dense, GQA, NoPE layers [34] | Transformers backend | Native | 16 / 4, hd 128; 72 KB | 65,536 (128K with YaRN) [34] | Apache 2.0 | 20.5 (too big) |
| `openbmb/MiniCPM5-2B-Base` | 2026-08 | 2.52B (1.98B non-emb) [37] | Dense Llama arch, 42 layers [3] | Transformers backend | Native | 16 / 2, hd 128; 42 KB | 524,288 in config [3] | Apache 2.0 [2] | 17.1 (too big) |
| `openbmb/MiniCPM5-1B-Base` (report A) | 2026-05 | 1.08B (0.68B non-emb) [38] | Dense Llama arch | Transformers backend | Native | 16 / 2, hd 128; 24 KB | 131,072 | Apache 2.0 | 8.5 |
| `tencent/Hunyuan-1.8B-Pretrain` | 2025-07 | 1.79B [2] | Dense, 32 layers, GQA [39] | Transformers backend (`HunYuanDenseV1ForCausalLM`) | Native (card first pointed to a pinned commit) [39] | 16 / 4, hd 128; 64 KB | 262,144 [39] | Tencent Hunyuan Community License: Territory excludes EU, UK, South Korea; 100M MAU cap; outputs may not improve non-Hunyuan models [40] | 12.7 |
| `tencent/Hunyuan-0.5B-Pretrain` | 2025-07 | 0.54B [2] | Dense, 24 layers | Transformers backend | Native | 16 / 8, hd 128; 96 KB | 262,144 | as above [40] | 5.2 |
| `baidu/ERNIE-4.5-0.3B-Base-PT` | 2025-06 | 0.36B [41] | Dense, 18 layers | Native (`Ernie4_5ForCausalLM`) | Native | 16 / 2, hd 128; 18 KB | 131,072 [41] | Apache 2.0 [41] | 4.2 |
| `01-ai/Yi-Coder-1.5B` | 2024-08 | 1.48B [2] | Dense Llama arch, MHA | Transformers backend | Native | 16 / 16, hd 128; 192 KB | 128K [36] | Apache 2.0 [36] | 10.9 |
| `microsoft/phi-1` | 2023-09 | 1.42B [2] | Dense, MHA, 24 layers | Transformers backend (`PhiForCausalLM`) | Native | 32 / 32, hd 64; 192 KB | 2,048 [3] | MIT [35] | 10.5 |
| `nvidia/Nemotron-Flash-1B` | 2025-11 | 0.97B [2] | Hybrid: DeltaNet, Mamba2, 2 attention layers [3][42] | No (not in registry) | No (custom code, pins transformers 4.56 or older) [42][44] | 16 / 4; about 4 KB + states | 36,000 in config | CC-BY-NC-4.0 [42] | n/a |
| `Salesforce/codegen-350M-mono` | 2022-04 | about 350M | Dense, Python only | No (not in registry) | Native | MHA 16; 80 KB | 2,048 | BSD-3 [47] | n/a |
| `allenai/OLMo-2-0425-1B` (report A) | 2025-04 | 1.48B | Dense, MHA | Transformers backend | Native | 16 / 16; 128 KB | 4,096 | Apache 2.0 | 10.9 |
| `HuggingFaceTB/SmolLM2-1.7B` (report A) | 2024-10 | 1.71B | Dense, MHA | Transformers backend | Native | 32 / 32; 192 KB | 8,192 | Apache 2.0 | 12.3 |
| `Qwen/Qwen2.5-Coder-0.5B` (report A) | 2024-11 | 0.49B | Dense | Transformers backend | Native | 14 / 2; 12 KB | 32,768 | Apache 2.0 | 5.0 |
| `Qwen/Qwen3-0.6B-Base` (report A) | 2025-04 | 0.60B | Dense | Native | Native | 16 / 8, hd 128; 112 KB | 32,768 | Apache 2.0 | 5.6 |
| `Qwen/Qwen3.5-0.8B-Base` (report A) | 2026-02 | 0.87B incl. vision | Hybrid DeltaNet + attention | Native | Native | 8 / 2, hd 256; 12 KB + state | 262,144 | Apache 2.0 | about 6 |

Note on vLLM: on vLLM `main` (fetched 2026-11-19) the `Llama`, `Qwen2`, `Phi`, `Phi3`, `Olmo2`, `SmolLM3` and `HunYuanDenseV1` architectures are listed under `_TRANSFORMERS_SUPPORTED_MODELS`, meaning vLLM serves them through its Transformers modeling backend rather than a native implementation [1]. They are still supported.

### 2b. Data, published base-model code numbers, instruct sibling and tool format

| Model | Training tokens | Code share; shell or bash | Open data | Published BASE code numbers (source table) | Official instruct; tool-call format; published tool or agent numbers (instruct) |
|-|-|-|-|-|-|
| Qwen3-1.7B-Base | 36T, three stages [5] | Not published; stage 2 has "more STEM, coding, reasoning" [5] | No | Qwen: EvalPlus 52.70, MultiPL-E 42.71, MBPP 55.40, CRUX-O 36.40 [5]. Falcon table: HumanEval 67.68, HE+ 60.98, MBPP 67.72 [18]. SmolLM3 table: HE+ 43.29, MBPP+ 59.25 [34] | Qwen3-1.7B; Hermes-style `<tool_call>` JSON, vLLM parser `hermes` [9]; BFCL v3 52.2 non-thinking, 56.6 thinking; LiveCodeBench v5 11.6 / 33.2 [5] |
| Qwen3.5-2B-Base | Not published | Not published | No | None found (card has no benchmarks [7]; no Qwen3.5 tech report found) | Qwen3.5-2B (non-thinking by default); vLLM parser `qwen3_coder` [8]; BFCL-V4 43.6, TAU2 48.8 (thinking) [8]; Terminal-Bench 2.0 2.3, TB 2.1 1.9, TB Lite 5.71 [43]; MiniCPM's table: TB v2.1 3.0, SWE-bench Verified 5.0 [37] |
| Qwen2.5-Coder-1.5B | 5.5T (body: 5.2T + about 300B repo-level) [4] | 70% code, 20% text, 10% math; 92 languages, list not given [4] | No | HE 43.9, HE+ 36.6, MBPP 69.2, MBPP+ 58.6, BigCodeBench full 34.6, hard 9.5 [4]. **MultiPL-E: Python 42.1, Bash 20.3**, avg 41.1 [4]. CRUXEval I-CoT 43.8, O-CoT 34.6 [4] | Qwen2.5-Coder-1.5B-Instruct; chat template uses `<tools>` and `<tool_call>` JSON tags [10]; no BFCL published |
| Qwen2.5-1.5B | 18T (series) [6] | Not published; Coder and Math data folded in [6] | No | HE 37.2, HE+ 32.9, MBPP 60.2, MBPP+ 49.6, MultiPL-E 33.1 [6]; Qwen3 table: EvalPlus 44.80, MultiPL-E 33.10, CRUX-O 29.60 [5] | Qwen2.5-1.5B-Instruct; BFCL v3 47.8 [5] |
| Qwen2.5-0.5B | 18T [6] | Not published | No | HE 30.5, MBPP 39.3, MultiPL-E 18.9 [6]; Qwen3 table: EvalPlus 31.85, CRUX-O 12.10 [5] | Qwen2.5-0.5B-Instruct; same template family |
| OpenCoder-1.5B-Base | 2.5T per card [12]; paper: about 2T over four epochs + 100B anneal [11] | 90% raw code, 10% code-related web [12]. RefineCode lists "Shell": 11.2 GB after filtering (Table 16) [11] | Partly: annealing corpus, FineWeb code and math recalls, SFT data on HF; raw code only as repo and path metadata covering about 50% of Stack v2 volume [14] | HE 54.3, HE+ 49.4, MBPP 70.6, MBPP+ 58.7, BigCodeBench 24.5, hard 5.4 [12][11]. No base MultiPL-E | OpenCoder-1.5B-Instruct; no tool format stated [11]; MultiPL-E Bash 29.7 (instruct) vs 29.1 for Qwen2.5-Coder-1.5B-Instruct [11] |
| deepseek-coder-1.3b-base | 2T series (card intro); "1 trillion tokens" in the same card's model summary [16] | 87% code, 13% NL; Shell 13.92 GB, 1.74% of the code data [15] | No | HE 34.8, MBPP 46.2 [15]; **MultiPL-E Python 34.8, Bash 10.1** [15][4]; CRUXEval I-CoT 32.1, O-CoT 28.2 [4] | deepseek-coder-1.3b-instruct; no tool format |
| Falcon-H1-0.5B-Base | 2.5T [17] | Raw code 5%, rewritten "code and math" 55% [17]; 67 code languages, list not read | No (mix of public and in-house sources) [17] | HE 35.98, HE+ 31.1, MBPP 52.12, MBPP+ 43.39 [18] | Falcon-H1-0.5B-Instruct; SFT includes hermes-function-calling-v1 [17]; no BFCL |
| Falcon-H1-1.5B-Base | 3T [17] | Raw code 8%, rewritten code and math 46.05% [17] | No | HE 50.0, HE+ 42.68, MBPP 65.08, MBPP+ 55.03 [18] | Instruct: HE 68.29, LiveCodeBench 17.61, CRUXEval 39.57 [17]; no BFCL |
| Falcon-H1-1.5B-Deep-Base | 3T [17] | as 1.5B | No | HE 52.44, HE+ 46.34, MBPP 70.9, MBPP+ 60.32 [18] | Instruct: HE 73.78, LiveCodeBench 23.87, CRUXEval 52.32 [17] |
| Falcon-H1-Tiny-90M family | Not on card [20] | Coder variant "trained on code data" | No | None found (blog is a JS-only HF Space) | Tool-Calling-90M is a dedicated function-calling variant [20]; no numbers found |
| LFM2.5-230M / 350M / 1.2B-Base | 28T [21] | LFM2 report: about 5% code for the dense LFM2 models [22]; LFM2.5 share not stated | No | None published | LFM2.5-1.2B-Instruct; Pythonic calls `[f(x=1)]` between `tool_call_start` and `tool_call_end` special tokens, JSON on request [23]; BFCL v3 49.12 [23]; card says "not recommended for ... programming" [23] |
| LFM3-1.7B-Base | 22T: about 15T web and synthetic, 3T code, math, science, then 3.5T mid-training with agentic traces and tool calls [24] | "Repository-level code" added vs LFM2 [24]; no share or shell figure | No | None published | LFM3-1.7B; BFCL V4 71.39, IFEval 96.66 [25]; blog says larger models still win on agentic coding [25]; mid-training already used the instruct chat template [24] |
| granite-4.0-h-350m-base | 15T in four stages (10, 2, 2, 0.5) [27] | Stage II has "higher percentages of code and math" [27]; no share | No (open source + proprietary) [27] | HE 34, HE+ 29, MBPP 17, MBPP+ 16 [27] | granite-4.0-h-350m; `<tool_call>` JSON in `<tools>` system block [28]; BFCL v3 43.32 [28] |
| granite-4.0-h-1b-base | 15T [27] | as above | No | HE 59, HE+ 56, MBPP 65, MBPP+ 54 [27] | granite-4.0-h-1b; BFCL v3 50.21; instruct HE 73, CRUXEval-O 36, BigCodeBench 29.12 [28] |
| granite-4.0-1b-base (report A) | 15T [27] | as above | No | HE 60, HE+ 57, MBPP 72, MBPP+ 60 [27]; report A bash 16/20 [45] | granite-4.0-1b; BFCL v3 54.82 [28] |
| granite-4.0-350m-base (report A) | 15T | as above | No | HE 32, HE+ 29, MBPP 45, MBPP+ 38 [27]; report A bash 9/20 | BFCL v3 39.32 [28] |
| gemma-3-270m | 6T [29] | Code included, no share [29] | No | None from Google [29] | gemma-3-270m-it; no tool numbers |
| gemma-3-1b-pt | 2T [29] | as above | No | None from Google [29]. Qwen3 table: EvalPlus 8.98, MultiPL-E 5.15, MBPP 9.20 [5]. Falcon table: HE 6.71, MBPP 12.7 [18] | gemma-3-1b-it: HE 41.5, MBPP 35.2 [29]; BFCL v3 16.3 [5] |
| gemma-4-E2B | Not read | Not read | No | Card reports instruct numbers only [31] | gemma-4-E2B-it: "native function calling"; Tau2 24.5%, LiveCodeBench v6 44.0% [31]; BFCL v4 36.6, TB v2.1 0.4 (MiniCPM table) [37] |
| Llama-3.2-1B | Up to 9T [32] | Not published | No | None from Meta (no code rows) [32]. Falcon table: HE 18.9, MBPP 35.98 [18] | Llama-3.2-1B-Instruct; BFCL V2 25.7, Nexus 13.5 [32] |
| SmolLM3-3B-Base | 11.2T [34] | Staged web, code, math [34] | Yes, public data mixture and configs [34] | HE+ 30.48, MBPP+ 52.91 [34] | SmolLM3-3B (hybrid reasoning) |
| MiniCPM5-2B-Base | Not stated in card | Not stated; lists `openbmb/UltraData-Code` among datasets [37] | Partly: Ultra-FineWeb, UltraData-Code, UltraData-Math, SFT and agent SFT and RL sets listed [37] | None (card shows post-trained numbers only) [37] | MiniCPM5-2B; "XML-style tool calls", SGLang `minicpm5` parser [37]; BFCL v4 66.6, SWE-bench Verified 46.4 (internal), TB v2.1 8.6 (Artificial Analysis) [37] |
| MiniCPM5-1B-Base (report A) | Not stated [38] | Not stated | Partly [38] | None [38]; report A bash 13/20 | MiniCPM5-1B; XML-style tool calls; scores only in an image [38] |
| Hunyuan-1.8B-Pretrain | Not stated | Not stated | No | EvalPlus 60.67, MultiPL-E 45.92, MBPP 66.14, CRUX-O 36.88 [39] | Hunyuan-1.8B-Instruct; BFCL v3 58.3, τ-Bench 18.2, ComplexFuncBench 22.3, LiveCodeBench 31.5 [39] |
| Hunyuan-0.5B-Pretrain | Not stated | Not stated | No | EvalPlus 39.71, MultiPL-E 21.83, MBPP 43.38, CRUX-O 30.75 [39] | Hunyuan-0.5B-Instruct; BFCL v3 49.8 [39] |
| ERNIE-4.5-0.3B-Base-PT | Not stated [41] | Not stated | No | None [41] | ERNIE-4.5-0.3B-PT (post-trained); tool calling not stated |
| Yi-Coder-1.5B | Not on card | 52 languages incl. `shell`, `powershell`, `batchfile`, `dockerfile`, `makefile` [36] | No | OpenCoder's table: HE 41.5, HE+ 32.9, MBPP 27.0, MBPP+ 22.2, BigCodeBench 23.5, hard 3.4 [11] | Yi-Coder-1.5B-Chat; no tool format stated |
| phi-1 | 7B unique tokens (6B filtered web + 1B synthetic textbooks) [35] | Python only [35] | Partly (Stack and StackOverflow subsets public, synthetic not) | HE 50.6, MBPP 55.5 [35] | None (base only) |
| Nemotron-Flash-1B | Not read | Not read | No | Card claims "+5.5% accuracy" vs Qwen3-0.6B, no table text [42] | Only a 3B instruct exists [42] |

## 3. Published MultiPL-E Bash vs Python (base models, 0.1B to 2B)

Only one source publishes per-language MultiPL-E for base models in this range: the Qwen2.5-Coder report, Table 6 [4] (DeepSeek-Coder's own paper gives the same 1.3B numbers [15]).

| Model | Python | Bash | Bash / Python | JS | Avg (8 langs) |
|-|-|-|-|-|-|
| Qwen2.5-Coder-0.5B | 28.0 | 7.0 | 0.25 | 29.2 | 24.7 |
| DeepSeek-Coder-1.3B-Base | 34.8 | 10.1 | 0.29 | 28.6 | 28.3 |
| Qwen2.5-Coder-1.5B | 42.1 | 20.3 | 0.48 | 49.1 | 41.1 |

For instruct models the only Bash numbers found are OpenCoder-1.5B-Instruct 29.7 and Qwen2.5-Coder-1.5B-Instruct 29.1 [11]. Qwen3, Hunyuan and Granite publish only a MultiPL-E average or none.

## 4. Notes per model

**Qwen3-1.7B-Base.** The strongest all-round Qwen dense base under 2B on Qwen's own table (EvalPlus 52.70 vs 44.80 for Qwen2.5-1.5B) [5]. Its instruct sibling has the best BFCL v3 among Qwen small models (52.2 non-thinking) and uses the Hermes `<tool_call>` format that most open RL stacks already parse [5][9]. Weak point for RL: 112 KB/token KV cache (8 KV heads, head dim 128), about 4x Qwen2.5-Coder-1.5B [3]. Full SFT about 12.3 GB, GRPO tight.

**Qwen3.5-2B-Base.** Hybrid (3 DeltaNet layers per gated-attention layer), so KV cache is tiny (12 KB/token) and long rollouts are cheap [7]. It is a vision-language checkpoint (`Qwen3_5ForConditionalGeneration`, 2.27B with the vision tower) [2][7]. The card says control tokens such as `<|im_start|>` were trained so LoRA can use the chat template without retraining embeddings [7]. Ai2's Tmax project ran terminal-agent RL on Qwen3.5-2B (instruct) and only moved Terminal-Bench 2.0 from 2.3 to 2.9; the 4B moved from 16.6 to 18.9 [43]. Tmax data, models and code are released [43]. Report A measured slow PyTorch fallback for DeltaNet without `flash-linear-attention` [45].

**Qwen2.5-Coder-1.5B.** The only base model in range with a published Bash score above 20 (20.3), and the best CRUXEval of the coder group [4]. 70% code [4]. Cheap KV cache (2 KV heads, 28 KB/token) [3]. Instruct template uses `<tool_call>` JSON [10]. Data closed.

**Qwen2.5-0.5B / 1.5B (general).** Clearly behind the Coder variants on code at equal size (HumanEval 37.2 vs 43.9 at 1.5B) [6][4]. Only relevant as a non-code control.

**OpenCoder-1.5B-Base.** Highest published HumanEval (54.3) and MBPP (70.6) among base models at or below 2B in its own table, ahead of Qwen2.5-Coder-1.5B on HumanEval but behind it on BigCodeBench (24.5 vs 34.6) [11]. The most open code model in range: annealing data, code-recall web corpora and SFT sets on HF, raw code as reproducible metadata for about half of Stack v2 volume [14]. Shell is in RefineCode (11.2 GB after filtering) [11]. Costs: 1.91B params with MHA (210 KB/token KV), 4K context [3][11], no tool-call training, and a revocable custom licence under PRC law [13]. Full GRPO with long rollouts will not fit.

**deepseek-coder-1.3b-base.** Older (2023), MHA, Bash 10.1 [15]. Shell is 1.74% of its code data [15]. The card contradicts itself on tokens (2T vs 1T) [16]. Superseded by Qwen2.5-Coder-1.5B on every published column [4].

**Falcon-H1-0.5B / 1.5B / 1.5B-Deep Base.** Parallel hybrid (attention and Mamba2 heads in every layer) [17]. Heavy use of rewritten code and math data (46 to 55% of tokens) [17]. Python numbers are strong for size: 0.5B HumanEval 35.98 beats Qwen3-0.6B (31.71) in Falcon's table, 1.5B-Deep MBPP 70.9 [18]. No Bash or tool-use numbers. Licence adds an updatable AUP and a mandatory attribution line [19].

**Falcon-H1-Tiny (90M).** Released 2026-01 with a base, a Coder and a dedicated Tool-Calling variant [20]. Could be a cheap Tier 1 debug model for a structured-tool interface, but no numbers were readable (blog is a JS-only Space).

**LFM2.5 (230M, 350M, 1.2B) Base.** Conv-attention hybrid, 28T tokens, 12 KB/token KV [21][3]. Liquid says the instruct model is "not recommended for ... programming" [23], and the LFM2 report puts code at about 5% of pretraining [22]. Tool format is Pythonic calls inside special tokens [23], which differs from the JSON `<tool_call>` convention used by Qwen, Granite and Hermes. Revenue-capped licence [26].

**LFM3-1.7B-Base.** New (2026-10-26). The only base model here whose card says agentic traces and tool calls were in pretraining (3.5T mid-training tokens) [24]. Instruct BFCL V4 71.39 is the highest tool-use number in range [25], but BFCL V4 is not comparable with v3, and Liquid gives no code numbers. 12 KB/token global KV plus a 256-token sliding window [3]. vLLM native; transformers has an `lfm3` folder on main [1][44]. Revenue-capped licence [26].

**Granite-4.0-H-350M / H-1B Base.** Mamba2 hybrids with only 4 attention layers (4 and 8 KB/token KV) [27]. Code numbers are close to the dense siblings at 1B (HumanEval 59 vs 60) but much worse at 350M on MBPP (17 vs 45) [27]. Apache 2.0, instruct siblings trained for `<tool_call>` JSON with BFCL v3 50.21 (H-1B) and 43.32 (H-350M) [28]. The dense granite-4.0-1b-base (report A, 16/20 bash) has the better published numbers and BFCL (54.82) but a 10x larger KV cache [27][28].

**Gemma 3 270M and 1B (pt).** Gated. No code numbers from Google for the pretrained models [29]; third-party numbers for 1B are very low (EvalPlus 8.98, HumanEval 6.71) [5][18]. Gemma terms require derivatives to carry the prohibited-use restrictions [30]. Not a fit for code agents.

**Gemma 4 E2B (base).** Apache 2.0 [31] and has native function calling in the instruct version, but 5.1B total params with per-layer embeddings plus vision and audio encoders [31]: full fine-tuning in 16 GB only if the per-layer embedding tables stay frozen or offloaded. Instruct Tau2 24.5% [31]. No smaller Gemma 4 exists [46].

**Llama 3.2 1B.** Gated, no base code numbers from Meta [32]; Falcon's table gives HumanEval 18.9 [18]. Instruct BFCL V2 25.7 [32]. Licence requires "Built with Llama" and a "Llama" name prefix on derivatives [33].

**SmolLM3-3B-Base.** Fully open data but 3.08B params: about 20 GB for full SFT with 8-bit AdamW, so out [2][34].

**MiniCPM5-2B-Base.** Strongest agent numbers near the range (instruct: SWE-bench Verified 46.4 internal, BFCL v4 66.6, TB v2.1 8.6) and partly open data, but 2.52B params (17 GB est. full SFT) [37]. The 1B sibling (report A) fits but publishes scores only as an image [38].

**Hunyuan-0.5B / 1.8B Pretrain.** Strong published base code numbers on the same metric set Qwen uses (1.8B: EvalPlus 60.67, MultiPL-E 45.92 vs Qwen3-1.7B-Base 52.70, 42.71) [39][5], and instruct BFCL v3 58.3 [39]. Licence blockers: not licensed in the EU, UK or South Korea, and outputs may not be used to improve other model families [40].

**ERNIE-4.5-0.3B-Base-PT.** Apache 2.0, vLLM native, cheap KV (18 KB) [41][1]. No code numbers or tool format published [41].

**Yi-Coder-1.5B.** Shell, PowerShell, batch, Dockerfile and Makefile are in its 52 languages [36]. In OpenCoder's table its MBPP is low (27.0) [11]. MHA 192 KB/token KV [3].

**phi-1 (and phi-1.5).** Python-only, 2K context, MHA; HumanEval 50.6 from 7B unique tokens [35]. Historical reference only.

**Nemotron-Flash-1B.** Hybrid with only 2 attention layers, but CC-BY-NC, custom code pinned to transformers 4.56 or older, not in vLLM [42][1][44]. Blocked.

**Older code models (codegen-350M-mono, starcoderbase-1b).** codegen-350M-mono is Python-only, 2K context and not in the vLLM registry [47][1]. starcoderbase-1b is gated (OpenRAIL-M) [2]. Neither adds anything over Qwen2.5-Coder-0.5B.

**Also found, not pursued:** Granite 4.1 starts at 3B (3.40B) [2]; EXAONE-4.0-1.2B has no base release [46]; MobileLLM-R1 base is non-commercial (report A); `HuggingFaceTB/nanowhale-100m-base` is a 110M toy trained on 2.6B tokens; LFM2.5-2.6B-Base and MiniCPM5-2B are above 2B [46].

## 5. Sites that blocked me or gave thin results

No site blocked me as a bot. No browser was used. Thin, gated or partial results:
- Local shell: midway through the run every `Bash` call was refused by a worktree-isolation guard (session cwd resolved to the shared checkout). After that I used only the files already downloaded to the scratch dir, WebFetch and WebSearch. Nothing was lost except convenience; all configs and cards had been fetched by then.
- Gated, card and config returned "Access restricted": `google/gemma-3-270m`, `google/gemma-3-1b-pt`, `meta-llama/Llama-3.2-1B`, `bigcode/starcoderbase-1b`. Configs read from the `unsloth/` mirrors instead (third-party copies; params cross-checked with the HF API of the gated repo).
- `huggingface.co/spaces/tiiuae/tiny-h1-blogpost`: JS-only Space, returned only the title. Falcon-H1-Tiny training data and benchmarks missing.
- `api.github.com/repos/huggingface/transformers/contents/src/transformers/models`: 459k-character listing, truncated; switched to per-folder raw URL checks.
- arxiv HTML is cut at 100k characters per fetch: Qwen2.5-Coder (2409.12186), Qwen3 (2505.09388), Qwen2.5 (2412.15115), OpenCoder (2411.04905) and Falcon-H1 (2507.22448) needed offset passes; LFM2 (2511.23404) only the first 100k of 231k read. The Falcon-H1 67-language list and its code ablations were not read.
- `arxiv.org/abs/2606.23321` (Tmax): abstract only; per-size details taken from the HF card.
- `huggingface.co/openbmb/MiniCPM5-1B-Base`: benchmark results are only in an image.
- `huggingface.co/openbmb/MiniCPM5-2B-Base`: the base card copies the instruct card; no base-model numbers, no token count.
- `huggingface.co/Qwen/Qwen3.5-2B-Base`: no benchmarks; no Qwen3.5 technical report found by search.
- `www.liquid.ai/blog/lfm3`: no tool-call format, no licence or code numbers.
- `falconllm.tii.ae/falcon-terms-and-conditions.html`: does not name "Falcon LLM License" explicitly; used the newest "TII Falcon License" section.
- `ai.google.dev/gemma/docs/core/model_card_3` and `ai.google.dev/gemma/terms` redirect to `gemma.ai` (worked after following).
- Cards with no data shares: Qwen3.5-2B-Base, LFM2.5 bases, Granite 4.0 nano bases, Hunyuan, ERNIE 0.3B, MiniCPM5-2B-Base.

## Sources

[1] vLLM model registry, `main`, fetched 2026-11-19: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py
[2] HF model API, `safetensors.total`, licence tag, gated flag: `https://huggingface.co/api/models/<id>` for each id above
[3] Model configs: `https://huggingface.co/<id>/raw/main/config.json` (gated Gemma 3 and Llama 3.2 via `unsloth/gemma-3-1b-pt`, `unsloth/gemma-3-270m`, `unsloth/Llama-3.2-1B`)
[4] Qwen2.5-Coder report, Tables 5, 6, 11, Section 3.1: https://arxiv.org/html/2409.12186
[5] Qwen3 report, Table 8 (base), Tables 19 and 20 (BFCL, LiveCodeBench): https://arxiv.org/html/2505.09388
[6] Qwen2.5 report, Table 5: https://arxiv.org/html/2412.15115
[7] Qwen3.5-2B-Base card: https://huggingface.co/Qwen/Qwen3.5-2B-Base
[8] Qwen3.5-2B card: https://huggingface.co/Qwen/Qwen3.5-2B
[9] Qwen function-calling docs: https://qwen.readthedocs.io/en/latest/framework/function_call.html
[10] Qwen2.5-Coder-1.5B-Instruct chat template: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/blob/main/tokenizer_config.json
[11] OpenCoder paper (base table, Table 16 shell row, instruct MultiPL-E Bash): https://arxiv.org/html/2411.04905
[12] OpenCoder-1.5B-Base card: https://huggingface.co/infly/OpenCoder-1.5B-Base
[13] OpenCoder licence: https://huggingface.co/infly/OpenCoder-1.5B-Base/blob/main/LICENSE
[14] OpenCoder datasets: https://huggingface.co/api/datasets?author=OpenCoder-LLM
[15] DeepSeek-Coder paper, Tables 1 and 3: https://arxiv.org/html/2401.14196
[16] deepseek-coder-1.3b-base card: https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base
[17] Falcon-H1 report, Tables 1, 6, 14, 15, 20, 21: https://arxiv.org/html/2507.22448
[18] Falcon-H1 base cards (evaluation tables with baselines): https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base , https://huggingface.co/tiiuae/Falcon-H1-1.5B-Base , https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Base
[19] TII Falcon License terms: https://falconllm.tii.ae/falcon-terms-and-conditions.html
[20] Falcon-H1-Tiny cards: https://huggingface.co/tiiuae/Falcon-H1-Tiny-Coder-90M , https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M , https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Base
[21] LFM2.5 base cards: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base , https://huggingface.co/LiquidAI/LFM2.5-350M-Base , https://huggingface.co/LiquidAI/LFM2.5-230M-Base
[22] LFM2 technical report, Section 3.1 and Table 1: https://arxiv.org/html/2511.23404
[23] LFM2.5-1.2B-Instruct card: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
[24] LFM3-1.7B-Base card: https://huggingface.co/LiquidAI/LFM3-1.7B-Base
[25] LFM3 blog: https://www.liquid.ai/blog/lfm3
[26] LFM Open License v1.0: https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/main/LICENSE
[27] Granite 4.0 nano base card (table covers 350M, H-350M, 1B, H-1B): https://huggingface.co/ibm-granite/granite-4.0-h-1b-base
[28] Granite 4.0 nano instruct card: https://huggingface.co/ibm-granite/granite-4.0-h-1b
[29] Gemma 3 model card: https://gemma.ai/models/gemma-3/
[30] Gemma Terms of Use: https://gemma.ai/resources/terms/
[31] Gemma 4 E2B card: https://huggingface.co/google/gemma-4-E2B
[32] Llama 3.2 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
[33] Llama 3.2 licence: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE
[34] SmolLM3-3B-Base card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base
[35] phi-1 card and paper: https://huggingface.co/microsoft/phi-1 , https://arxiv.org/abs/2306.11644
[36] Yi-Coder-1.5B card: https://huggingface.co/01-ai/Yi-Coder-1.5B
[37] MiniCPM5-2B-Base card: https://huggingface.co/openbmb/MiniCPM5-2B-Base
[38] MiniCPM5-1B-Base card: https://huggingface.co/openbmb/MiniCPM5-1B-Base
[39] Hunyuan-1.8B-Pretrain card (table covers 0.5B to 7B): https://huggingface.co/tencent/Hunyuan-1.8B-Pretrain
[40] Tencent Hunyuan Community License: https://huggingface.co/tencent/Hunyuan-1.8B-Pretrain/blob/main/LICENSE
[41] ERNIE-4.5-0.3B-Base-PT card: https://huggingface.co/baidu/ERNIE-4.5-0.3B-Base-PT
[42] Nemotron-Flash-1B card: https://huggingface.co/nvidia/Nemotron-Flash-1B
[43] Tmax 2B card and paper: https://huggingface.co/allenai/tmax-2b , https://arxiv.org/abs/2606.23321
[44] transformers `main` model folders, checked per path (`lfm3` and `gemma4` exist, `nemotron_flash` returns 404): https://github.com/huggingface/transformers/tree/main/src/transformers/models
[45] Report A (this repo): `docs/research/models/report.md`
[46] HF listings used for discovery: `https://huggingface.co/api/models?author=<org>&sort=createdAt&direction=-1` for Qwen, google, ibm-granite, LiquidAI, tiiuae, HuggingFaceTB, microsoft, meta-llama, openbmb, allenai, nvidia, infly, 01-ai, tencent, baidu; and `?search=Base|-pt|Pretrain|coder|gemma-4|granite-4.1|EXAONE-4|MobileLLM`
[47] codegen-350M-mono card: https://huggingface.co/Salesforce/codegen-350M-mono
