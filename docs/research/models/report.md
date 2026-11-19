# Research A: model shortlist for nanoalignment

Date: 2026-11-19. Scope: open base models from about 70M to 1.7B params for Tier 1 (log-prob proxy) and Tier 2 (bash SFT and GRPO) on one RTX 5080 (16 GB, about 14 GB usable).
Files in this folder: `scripts/` (prompts, eval script, memory script), `results/` (raw JSONL, tables, per-model outputs, run log), `dl_readmes/` (downloaded model cards, untrusted).

## 1. Short answer

- **Tier 1: SmolLM2-135M** (fallback: Pythia-160M if a clean pre-decay 90% checkpoint matters more than quality). Best proxy quality per FLOP among open small models: reference log-prob -0.99 vs -2.18 for Pythia-160M, 5/20 vs 0/20 on bash. Apache 2.0, mostly open 2024 data (so its prior about AI reflects recent web text), 49k vocab, about 45 min for 200M training tokens. Its one gap: no 90% pre-decay checkpoint (72% `step-1440000` is the last pre-decay one).
- **Tier 2: OLMo 2 1B is defensible but not clearly best.** It is the only candidate that is fully open, has dense checkpoints with a clean pre-anneal 90% branch point, fits 16 GB for full SFT, and is near the top of the open group on bash (11/20). But its code share is low (2.1%, shell about 0.01%), the 90% branch point scores only 8/20, and full GRPO fits only without a separate reference model. See section 6.
- No 2025 or 2026 release beats OLMo 2 1B on all three of open data, dense checkpoints and size. There is no Olmo 3 model below 7B ([HF allenai listing](https://huggingface.co/api/models?author=allenai&search=1B)). New candidates worth knowing: Ettin decoders (2025, fully open, checkpoints), MiniCPM5-1B-Base (2026, partly open data, no checkpoints), Qwen3.5-0.8B-Base (2026, closed data).

## 2. Ranking table

Rows are sorted by Tier 2 rank, which weighs open data and checkpoints first (they are hard requirements for Order B and the AI-content audit), then bash skill and memory. "SFT" is full fine-tuning VRAM with 8-bit AdamW from section 4. Code share is of pretraining tokens. "90% ckpt" means a checkpoint near 90% of pretraining tokens, and whether its learning rate (LR) was still high (pre-anneal) or already decaying.

| Rank (T2) | Model | Params | Code in training data | Shell specifically | Data open? | Checkpoints | ~90% checkpoint (HF revision) | Licence | Tokenizer, context |
|-|-|-|-|-|-|-|-|-|-|
| 1 | OLMo 2 1B (`allenai/OLMo-2-0425-1B`). Bash 11/20; SFT 10.9 GB | 1.48B | 83.0B StarCoder tokens of 3.90T stage 1 = 2.1% [1]. Stage 2 (Dolmino, 50B) has almost no code: CodeSearchNet 1.78M tokens [1] | `shell` files are 0.46% of the StarCoder subset bytes (0.47 of 102.9 GB), so about 0.4B tokens, about 0.01% of all tokens. Git commits (11.4 GB) and GitHub issues (24.9 GB) add some shell text [2] | Yes: OLMo-mix-1124 and Dolmino-mix-1124 on HF [2][1] | 195 stage-1 branches (every 10k steps, about 21B tokens) + 72 stage-2 branches [3] | `stage1-step1720000-tokens3608B` = 90.2% of 4.0T. Pre-anneal: LR is a cosine set to 5T and cut at 4T [4], so LR is still about 26% of peak (my estimate, assuming cosine to 10% as for 7B) and no Dolmino midtraining yet | Apache 2.0 [5] | OLMo 2 (cl100k-based), vocab 100,352; ctx 4096; MHA 16 heads |
| 4 | SmolLM2-1.7B. Bash 9/20; SFT 12.3 GB, GRPO does not fit | 1.71B | 11T tokens. Code 10% (0 to 6T, StarCoderData), 20% (6 to 8T), Stack-Edu (8 to 10T), 24% (10 to 11T) [6]. So at least about 1.2T code tokens | Stack-Edu Shell 3.13B tokens [7]; stage 3 used the StarCoder2Data Shell subset because Stack-Edu Shell was under 4B [6] | Mostly: FineWeb-Edu, DCLM, StarCoderData, Stack-Edu, FineMath, Cosmopedia are public; exact per-stage mix is in the paper [6] | 41 branches every 125k steps (about 262B tokens) in `SmolLM2-1.7B-intermediate-checkpoints` [8] | `step-4750000` = 9.96T tokens, 90.6%, just before the WSD decay starts at 10T [6][8]. Pre-anneal, but the stage-3 data mix already applies | Apache 2.0 | SmolLM2 BPE, vocab 49,152; ctx 8192; MHA 32 heads |
| 3 | Ettin decoder 1B / 400M (`jhu-clsp/ettin-decoder-*`). Bash 9 and 8/20; SFT 8.0 / 4.4 GB | 1.0B / 396M | StarCoder 263.9B of 1.70T pretrain = 15.5%; 15.4% of the 250B extension; 26.5% of the decay mix [9][10] | Not broken out (StarCoder includes shell) | Yes, including batch-level data order [9] | HF tags: 400M has 200 `step*`, 28 `ext*`, 6 `decay*`; 1B has 67, 10, 2 [11] | Pretrain ends at 85% (1.7T of 2T), then `ext*` tags (LR inverse-sqrt from peak to half peak) cover 85% to 97.5% [10]. So a 90% `ext*` tag exists and is pre-decay | MIT [11] | ModernBERT tokenizer, vocab 50,368; ctx 7999. Arch `ModernBertDecoderForCausalLM` is not in vLLM [12] |
| 5 | Pythia 1B / 410M. Bash 9 and 7/20; SFT 8.1 GB (1B) | 1.01B / 405M | Pile GitHub: 95.16 GiB, 7.59% of the Pile [13], so about 23B of 300B tokens | Unknown (Pile does not split GitHub by language [13]) | Yes (Pile, and the exact batch order) | 154 per model: step0, 10 log-spaced, then every 1000 steps to 143000 [14] | `step129000` = 90.2%. No midtraining. LR decays to 10% of max over the whole run [15], so it is late in decay | Apache 2.0 | GPT-NeoX, vocab 50,304 (padded); ctx 2048 |
| 2 | SmolLM2-360M. Bash 10/20; SFT 4.2 GB | 362M | 4T tokens; Stack-Edu in the mix from the start, share not published [6] | Stack-Edu Shell 3.13B (whole dataset) [7] | Mostly (same sources as above) | 16 branches every 160k steps (about 250B tokens) [16] | WSD with 20% decay [6]; config: decay starts at step 2,080,000 of 2,600,000 [17]. `step-2080000` = 80%, exactly pre-decay. `step-2400000` = 92%, mid-decay | Apache 2.0 | vocab 49,152; ctx 8192; GQA 5 KV heads |
| 6 | Qwen2.5-Coder-0.5B. Bash 16/20; SFT 5.0 GB | 494M | 5.5T tokens (the body says 5.2T), mix 70% code, 20% text, 10% math [18]: about 3.7T code tokens | Shell not listed; Bash is in their MultiPL-E evaluation [18] | No | None (only `main`) [19] | None | Apache 2.0 | Qwen, vocab 151,936; ctx 32,768; GQA 2 KV heads |
| 7 | Qwen3-0.6B-Base. Bash 14/20; SFT 5.6 GB | 596M | 36T tokens, code share not published [20] | Not published | No | None | None | Apache 2.0 | Qwen, vocab 151,936; ctx 32,768 |
| 8 | MiniCPM5-1B-Base (2026). Bash 13/20; SFT 8.5 GB | 1.08B | Not published. Base includes stable, decay and mid-training stages [21] | Not published | Partly: Ultra-FineWeb, Ultra-FineWeb-L3, UltraData-Math, and UltraData-SFT-2605 listed [21] | None | None | Apache 2.0 | vocab 130,560; ctx 131k; Llama arch |
| 9 | Qwen3.5-0.8B-Base (2026). Bash 12/20 | 0.87B | Not published; multimodal, hybrid Gated DeltaNet [22] | Not published | No | None | None | Apache 2.0 | vocab 248,320; ctx 262k |
| T1 | SmolLM2-135M. Bash 5/20 | 135M | 2T tokens; Stack-Edu from the start, share not published [6] | as above | Mostly | 8 branches every 240k steps (about 252B tokens) [23] | WSD, decay starts at step 1,600,000 of 2,000,000 [17]. `step-1440000` = 72% (pre-decay); `step-1680000` = 84% (in decay, LR 80% of peak); no clean 90% pre-decay checkpoint | Apache 2.0 | vocab 49,152; ctx 8192 (trained at 2048) |
| T1 | Pythia-160M / 70M. Bash 0/20 (160M) | 162M / 70M | as Pythia above | Unknown | Yes | 154 | `step129000` (90.2%) | Apache 2.0 | GPT-NeoX; ctx 2048 |
| T1 | Ettin decoder 150M. Bash 4/20 | 150M | as Ettin above | Not broken out | Yes | 236 tags [11] | `ext*` tags, pre-decay | MIT | ModernBERT tokenizer; ctx 7999 |

Also checked and set aside: Gemma 3 270M (gated, Gemma licence, closed data); MobileLLM-R1 base (gated, non-commercial "other" licence); IBM Granite 4.0 350M/1B base (open and proprietary data mix, four stages, no checkpoints [28]; 9/20 and 16/20 on bash); LFM2-350M (other licence); Baguettotron 321M and 600M (100% synthetic SYNTH data, non-code focus [24]); open-sci-ref 0.13B/0.4B/1.7B (single-corpus 300B-token reference runs, 37 checkpoints each, custom `OpenSciForCausalLM` class, no licence tag); Marin Delphi scaling suite (open data incl. StarCoderData, but only about 50B tokens for 1B); Bolmo-1B (byte-level OLMo 2, vocab 520); SmolLM3-3B (has 133 checkpoint branches but 3B is too big for full fine-tuning here).

## 3. Bash test (base models, few-shot, greedy)

Method: 20 natural-language tasks, 4-shot prompt (`# Task: ...` then `$ cmd`), greedy, max 48 new tokens, first line kept. Commands were never executed. Each output was checked with `bash -n` on a temp file and matched against strict reference regexes (several accepted variants per task). The log-prob column is the mean per-token log-prob of the reference command given the prompt; it is a smooth proxy that still separates models that score 0. bf16, transformers 5.23, torch 2.14.1+cu130 on the RTX 5080. Prompts: `scripts/prompts.json`. Raw outputs: `results/raw.jsonl`, `results/outputs_by_model.md`.

| model | params | correct /20 (strict regex) | bash -n ok /20 | mean ref log-prob per token | fwd tok/s (8x512) | gen tok/s (b=1) | peak VRAM GB |
|-|-|-|-|-|-|-|-|
| EleutherAI/pythia-160m | 162M | 0 | 20 | -2.18 | 312,740 | 181.4 | 0.98 |
| Pythia-160M @step129000 (90%) | 162M | 0 | 20 | -2.05 | 312,890 | 495.9 | 0.98 |
| HuggingFaceTB/SmolLM2-135M | 135M | 5 | 20 | -0.99 | 231,842 | 174.7 | 0.86 |
| SmolLM2-135M @step-1440000 (72%, pre-decay) | 135M | 3 | 20 | -1.10 | 231,612 | 172.1 | 0.86 |
| HuggingFaceTB/SmolLM2-360M | 362M | 10 | 20 | -0.41 | 105,006 | 167.4 | 1.39 |
| SmolLM2-360M @step-2080000 (80%, pre-decay) | 362M | 7 | 20 | -0.67 | 104,946 | 169.9 | 1.39 |
| Qwen/Qwen2.5-Coder-0.5B | 494M | 16 | 20 | -0.63 | 83,122 | 205.0 | 2.48 |
| Qwen/Qwen3-0.6B-Base | 596M | 14 | 20 | -0.63 | 57,811 | 142.1 | 3.1 |
| allenai/OLMo-2-0425-1B | 1485M | 11 | 20 | -0.93 | 32,268 | 205.6 | 4.48 |
| OLMo 2 1B @stage1-step1720000-tokens3608B (90%, pre-anneal) | 1485M | 8 | 20 | -1.00 | 32,335 | 209.7 | 4.48 |
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

Reading:
- Best out of the box: Qwen2.5-Coder-0.5B and Granite-4.0-1B-Base (16/20), then Qwen3-0.6B-Base (14), MiniCPM5-1B-Base (13), Qwen3.5-0.8B-Base (12). All have closed or partly closed data and no checkpoints.
- Among fully or mostly open models with checkpoints: OLMo 2 1B 11/20, SmolLM2-360M 10, SmolLM2-1.7B 9, Pythia-1B 9, Ettin-1B 9, Ettin-400M 8, Pythia-410M 7, SmolLM2-135M 5, Ettin-150M 4, Pythia-160M 0.
- Branch points are weaker than the final models: OLMo 2 1B at 90% stage 1 scores 8/20 (final 11); SmolLM2-360M at 80% scores 7/20 (final 10); SmolLM2-135M at 72% scores 3/20 (final 5). The anneal adds bash skill, so a CPT arm branched before the anneal starts lower.
- SmolLM2-1.7B has the best reference log-prob (-0.30) but only 9/20 strict: it often picks a plausible but different command. SmolLM2-360M is close behind on log-prob (-0.41) at a fifth of the size.
- Common misses for all models: `pytest` without `-x` (task 5), `mkdir` without `-p` (task 7), `git add .` with no commit (task 9), `awk '{print $3}'` without `-F,` (task 10). Strict scoring; a lenient grader would add about 1 to 2 per model.
- Pythia-1B produced `rm -rf .` for task 15. This is a reminder that the harness sandbox is required.
- Throughput: "fwd tok/s" is a bf16 forward pass at batch 8 x 512, plain eager PyTorch, no compile. Training costs about 3x a forward pass, so rough training speed is fwd/3: SmolLM2-135M about 77k tok/s (200M tokens in about 45 min), Pythia-160M about 104k tok/s (about 32 min), OLMo 2 1B about 11k tok/s (100M tokens in about 2.5 h). "gen tok/s" is batch-1 HF `generate`, which is CPU-overhead bound and not a model comparison. Qwen3.5 ran on the slow PyTorch fallback (flash-linear-attention not installed).
- Process RSS was about 2.0 to 2.6 GB for models up to 600M and 6.9 GB peak for OLMo 2 1B (fp32 safetensors on disk). Only one torch process ran at a time.

## 4. Memory estimates (16 GB card, about 14 GB free)

Assumptions: bf16 weights and grads; activations about 2 GB with gradient checkpointing and a fused or chunked cross-entropy (needed for the 100k to 152k vocabularies, since one 4k-token sequence of fp32 logits is 1.6 GB at vocab 100k); GRPO KV cache for 8 rollouts of 2k tokens. Numbers from `scripts/mem_est.py`, using each model's config.

| Model | Params | KV cache KB/token | A: bf16 W+G, 8-bit AdamW (6 B/param) | B: bf16 AdamW states (8 B/param) | C: classic AMP, fp32 master + fp32 Adam (16 B/param) | GRPO: A + bf16 reference + KV | GRPO with beta=0 (no reference) |
|-|-|-|-|-|-|-|-|
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

Reading:
- OLMo 2 1B fits full fine-tuning only with 8-bit or bf16 optimizer states (10.9 to 13.9 GB). Classic AMP (25.8 GB) does not fit. bf16-only states should use stochastic rounding or Kahan summation to avoid lost updates.
- OLMo 2 1B GRPO with a separate reference model and a vLLM rollout copy does not fit (16 GB + 3 GB for vLLM weights). It fits with beta=0 (no KL, as in DAPO and Dr. GRPO) and HF or vLLM-sleep-mode rollouts (13.1 GB, tight), or with LoRA (then the reference is the adapter-disabled base, free). Its MHA KV cache is 128 KB/token, 10x Qwen2.5-Coder-0.5B, so rollout length is the main limit.
- SmolLM2-1.7B is too tight for full GRPO (15.5 GB even with beta=0).
- Everything at or below 600M params fits every configuration with room for vLLM.
- Inference RSS during the test was about 2.1 GB for small models and 6.9 GB peak for OLMo 2 1B (fp32 safetensors loaded then cast); the latter is file-backed mmap, but plan for it.

## 5. Recommendation

- **Tier 1: `HuggingFaceTB/SmolLM2-135M`.** Order B debug branch: `HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints` revision `step-1440000` (72%, last pre-decay) or `step-1680000` (84%, early decay). If an exact 90% pre-decay branch point matters more than quality, use `EleutherAI/pythia-160m` revision `step129000` (fully open data with exact batch order, 154 checkpoints), accepting a much weaker model.
- **Tier 2: keep `allenai/OLMo-2-0425-1B`** as the default, with Order B branching from `stage1-step1720000-tokens3608B`. Requirements: 8-bit AdamW (or bf16 states with stochastic rounding), gradient checkpointing, a fused cross-entropy (Liger or chunked), GRPO with beta=0 or LoRA, rollouts of at most about 8 x 2k tokens.
- **Closest alternatives if the owner wants to change Tier 2:** SmolLM2-360M (10/20, 4.2 GB for full SFT, 3x faster, pre-decay checkpoint at 80%, but weaker overall), and Ettin decoder 1B (fully open with batch order, 15.5% code, pre-decay checkpoints at 90%, but 9/20 and no vLLM support).
- **Not recommended despite better bash scores:** Qwen2.5-Coder-0.5B, Granite-4.0-1B-Base, Qwen3-0.6B-Base, MiniCPM5-1B-Base and Qwen3.5-0.8B-Base. All lack open data (you cannot know what they saw about AI) and none publishes intermediate checkpoints, so Order B is impossible.

## 6. Conflicts with the design to flag (not decisions)

1. **OLMo 2 1B has little code.** 2.1% of stage-1 tokens are code and about 0.01% are shell files; Dolmino adds almost none [1][2]. Bash skill must come mostly from bash SFT. If SFT data is small, the RL stage may see too few successes to learn anything, which would also hide any alignment effect.
2. **The 90% branch point is pre-anneal and weaker.** OLMo 2 1B at stage 1 only is near random on MMLU and gains 37% from the 50B Dolmino anneal [4]. In our test the 90% checkpoint scores 8/20 vs 11/20 for `main`. So each Order B arm needs the same anneal after CPT (50B Dolmino tokens is about 50 GPU-days at about 11k tok/s, which is not feasible), or a short shared anneal on a subset, or comparisons only between arms branched at the same point and never against `main`.
3. **Tice et al. inserted late, after midtraining, not at 90%.** Their CPT arm added 1B tokens (500M synthetic) after midtraining and kept most of the effect: 15.2% vs 13.2% misalignment for end-to-end [25]. A CPT-from-`main` arm would avoid point 2 and match their design. This suggests adding an "Order C: CPT from the final model" arm, at least as a check.
4. **GRPO memory on OLMo 2 1B.** Full GRPO with a separate bf16 reference model needs about 16 GB in my estimate, over budget. Use beta=0, or LoRA with the base as reference, or offload the reference. Its MHA KV cache (128 KB/token) also caps rollout batch size.
5. **CPT token budget on Tier 2.** At about 11k training tok/s, 100M CPT tokens per arm takes about 2.5 h; three arms (positive, neutral, negative) take about 8 h before SFT and RL.
6. **OLMo 2 1B branch naming.** The HF repo has three `stage2-ingredient{1,2,3}` runs of about 50B tokens each [3], while the paper says the 1B used a single 50B anneal [4]. Which run or soup became `main` is not stated; check before relying on `main` as the "post-anneal" reference.
7. **Out-of-the-box bash favours closed models.** If the owner values a stronger bash prior over data openness and checkpoints, Qwen2.5-Coder-0.5B (16/20, 5.0 GB full SFT) is the pick, but then Order B and the "what did it see about AI" audit are both impossible.

## 7. Useful related finding

Tice et al. trained 6.9B models on 500B DCLM tokens plus a 50B midtraining stage, inserting about 1% synthetic documents (5B pretraining + 500M midtraining tokens) [25]. Their continued-pretraining variant added 1B tokens (500M synthetic + 500M replay) after midtraining, the final 1% of training, and reached 15.2% post-trained misalignment versus 13.2% for end-to-end insertion [25]. They report no small-model runs [25]. Their synthetic data is public on HF under `geodesic-research`, for example `synth-scenario-docs-positive-alignment-midtraining`, `hyperstition-character-stories-9.6k` and `discourse-grounded-misalignment-synthetic-scenario-data` [26]. Their base models are 6.9B GPT-NeoX with 19 `global_step*` branches [27].

## 8. Sites that blocked me or gave thin results

No site blocked me as a bot. No browser was used. Thin or incomplete results, worth a browser pass if detail matters:
- arxiv.org `/abs/` pages give only the abstract (worked around with `/html/`): 2501.00656, 2409.12186, 2601.10160, 2101.00027.
- arxiv.org/html/2502.02737 (SmolLM2): fetch truncated Table 7 (per-language Stack-Edu tokens); I used the Stack-Edu card instead. The 135M and 360M code shares are not in the paper or model cards.
- arxiv.org/html/2501.00656v3 (OLMo 2): the 1B LR shape is not named in Appendix B; I assumed cosine as for 7B.
- arxiv.org/html/2601.10160 (Tice et al.): only the first 100k of 338k characters read; Appendix A (model zoo) and D (hyperparameters) not read.
- huggingface.co/datasets/bigcode/starcoderdata: no per-language sizes on the card.
- huggingface.co/datasets/allenai/olmo-mix-1124: card has no language breakdown (solved via the HF tree API).
- Model cards with no data shares: Qwen3-0.6B-Base, Qwen3.5-0.8B-Base, MiniCPM5-1B-Base, granite-4.0-1b-base.
- Web search for 2026 small fully open models returned few hits (one search); a broader pass may find more.

## Sources

[1] OLMo 2 paper, Tables 4 and 5, Section 3: https://arxiv.org/html/2501.00656v3
[2] OLMo-mix-1124 card and file tree (per-language StarCoder files; sizes summed via HF API): https://huggingface.co/datasets/allenai/olmo-mix-1124 and https://huggingface.co/api/datasets/allenai/olmo-mix-1124/tree/main/data/starcoder/v1-decon-100_to_20k-2star-top_token_030/documents
[3] OLMo 2 1B refs: https://huggingface.co/api/models/allenai/OLMo-2-0425-1B/refs
[4] OLMo 2 paper, Appendix B (1B: "set the schedule to 5 trillion tokens but truncate at the 4 trillion mark", single 50B anneal, +37% from Dolmino, stage 1 alone near random on MMLU): https://arxiv.org/html/2501.00656v3
[5] OLMo 2 1B card: https://huggingface.co/allenai/OLMo-2-0425-1B
[6] SmolLM2 paper: https://arxiv.org/html/2502.02737
[7] Stack-Edu card: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
[8] SmolLM2-1.7B checkpoints (GBS 2,097,152 tokens): https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
[9] Ettin pretraining data card: https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data
[10] Ettin paper (schedules, mid-training and decay mixes): https://arxiv.org/html/2507.11412
[11] Ettin decoder card and refs API: https://huggingface.co/jhu-clsp/ettin-decoder-400m and https://huggingface.co/api/models/jhu-clsp/ettin-decoder-400m/refs
[12] vLLM model registry (no ModernBertDecoder): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py
[13] Pile paper, Table 1: https://arxiv.org/html/2101.00027v1
[14] Pythia card: https://huggingface.co/EleutherAI/pythia-160m
[15] Pythia paper, Appendix B: https://arxiv.org/html/2304.01373
[16] SmolLM2-360M checkpoints (GBS 1,572,864): https://huggingface.co/HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints
[17] SmolLM2 nanotron configs (`lr_decay_starting_step`): https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm2
[18] Qwen2.5-Coder report: https://arxiv.org/html/2409.12186
[19] Qwen2.5-Coder-0.5B card: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B
[20] Qwen3-0.6B-Base card: https://huggingface.co/Qwen/Qwen3-0.6B-Base
[21] MiniCPM5-1B-Base card: https://huggingface.co/openbmb/MiniCPM5-1B-Base
[22] Qwen3.5-0.8B-Base card: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
[23] SmolLM2-135M checkpoints (GBS 1,048,576): https://huggingface.co/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints
[24] Baguettotron card: https://huggingface.co/PleIAs/Baguettotron
[25] Tice et al. 2026, Alignment Pretraining: https://arxiv.org/html/2601.10160
[26] geodesic-research datasets: https://huggingface.co/api/datasets?author=geodesic-research
[27] https://huggingface.co/api/models/geodesic-research/sfm_baseline_unfiltered_base/refs
[28] Granite 4.0 1B base card: https://huggingface.co/ibm-granite/granite-4.0-1b-base
