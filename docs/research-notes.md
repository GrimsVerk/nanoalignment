# Research notes (STOP 2)

Spec section 1b, researched 2026-11-19. Every claim has a source. "Measured" means we ran it on this machine; the scripts and raw output are in `docs/research/`. The full section reports, with more detail and sources, are:

- Models: [`docs/research/models/report.md`](research/models/report.md)
- Efficiency: [`docs/research/efficiency/report.md`](research/efficiency/report.md)
- Data: [`docs/research/data/report.md`](research/data/report.md)

## 0. Hardware (detected, not assumed)

Source: `nvidia-smi`, `pacman -Q`, `docker`, `lscpu`, `free` on the machine, 2026-11-19.

| Item | Found |
|:-|:-|
| OS | Omarchy (Arch based), kernel 7.2.15 |
| GPU | RTX 5080, 16,303 MiB, compute capability 12.0 (sm_120). The desktop uses about 2 GB, so about 13.5 to 14 GB is usable |
| Driver | nvidia-open-dkms 615.78.08, CUDA driver 13.4 |
| CUDA toolkit | None (no `nvcc`). PyTorch cu130 wheels bring their own runtime. Anything that compiles CUDA at run time fails |
| CPU, RAM, disk | Ryzen 9 9950X3D (16 cores), 30 GB RAM (shared with other work), 60 GB swap, 1.8 TB free |
| Python | System 3.14.8, uv 0.13.4. The stack needs Python 3.12 (pinned through uv) |
| Docker | 29.9.2 installed with buildx and compose. **The daemon is off and the user is not in the `docker` group.** No nvidia container toolkit, which the harness does not need: the sandbox only runs bash, the model stays on the host |

## A. Model shortlist

### Ranking

Open data and intermediate checkpoints come first, because Order B and the "what did it already see about AI" audit need them. Bash score: our 20-task few-shot test, greedy, strict match, outputs never executed ([script](research/models/scripts/bash_eval.py), [table](research/models/results/bash_table.md)).

| Model | Params | Code in pretraining | Shell | Data open | Checkpoints | ~90% branch point | Bash /20 | Full SFT VRAM (8-bit Adam) |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
| OLMo 2 1B | 1.48B | 2.1% of 3.9T ([paper](https://arxiv.org/html/2501.00656v3)) | about 0.01% ([data tree](https://huggingface.co/api/datasets/allenai/olmo-mix-1124/tree/main/data/starcoder/v1-decon-100_to_20k-2star-top_token_030/documents)) | Yes | 195 + 72 ([refs](https://huggingface.co/api/models/allenai/OLMo-2-0425-1B/refs)) | `stage1-step1720000-tokens3608B` (90.2%, before the anneal) | 11 (8 at 90%) | 10.9 GB |
| SmolLM2-360M | 362M | share not published, Stack-Edu from the start ([paper](https://arxiv.org/html/2502.02737)) | Stack-Edu Shell 3.1B tokens ([card](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)) | Mostly | 16 ([repo](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints)) | `step-2080000` (80%, before decay) | 10 | 4.2 GB |
| Ettin decoder 1B | 1.0B | 15.5% ([data](https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data)) | not split out | Yes, with batch order | 67 tags | `ext*` tags near 90% ([paper](https://arxiv.org/html/2507.11412)) | 9 | 8.0 GB. Not supported by vLLM |
| SmolLM2-1.7B | 1.71B | 10 to 24% by stage ([paper](https://arxiv.org/html/2502.02737)) | as above | Mostly | 41 | `step-4750000` (90.6%) | 9 | 12.3 GB. GRPO does not fit |
| Pythia 1B / 160M | 1.0B / 162M | 7.6% GitHub ([Pile](https://arxiv.org/html/2101.00027v1)) | unknown | Yes, with batch order | 154 ([card](https://huggingface.co/EleutherAI/pythia-160m)) | `step129000` (90.2%) | 9 / 0 | 8.1 GB |
| SmolLM2-135M | 135M | as SmolLM2 | as above | Mostly | 8 | `step-1440000` (72%, before decay). No clean 90% | 5 | 2.8 GB |
| Qwen2.5-Coder-0.5B | 494M | about 70% ([report](https://arxiv.org/html/2409.12186)) | not published | No | None | None | 16 | 5.0 GB |
| Qwen3-0.6B-Base | 596M | not published | not published | No | None | None | 14 | 5.6 GB |

New 2025 and 2026 models checked: Granite-4.0-1B-Base 16/20, MiniCPM5-1B-Base 13/20 and Qwen3.5-0.8B-Base 12/20 bash well, but have closed or partly closed data and no checkpoints. There is no Olmo 3 below 7B ([allenai listing](https://huggingface.co/api/models?author=allenai&search=1B)). Gensyn Open-1B (Sept 2026, Apache 2.0) is fully open, with 12% code and checkpoints every 100 steps including optimizer state, but it is clearly weaker than OLMo 2 1B (OLMES 50.1 vs 61.5, HellaSwag 48.6 vs 67.5) and needs custom model code with int8 training ([report, Table 5 and 6](https://arxiv.org/html/2609.17380), [card](https://huggingface.co/Gensyn/open-1b-base)). Not bash tested.

### Picks

- **Tier 1: SmolLM2-135M** (Apache 2.0). Best quality per FLOP among open tiny models: reference command log-prob of -0.99 per token vs -2.18 for Pythia-160M, bash 5/20 vs 0/20 ([table](research/models/results/bash_table.md)). Liger kernels support it, Pythia's architecture is not supported ([Liger](https://github.com/linkedin/Liger-Kernel)). Measured 90k tok/s with our stack, so 200M tokens take about 40 minutes ([efficiency report](research/efficiency/report.md)). Fallback if an exact 90% checkpoint matters at Tier 1: Pythia-160M `step129000`.
- **Tier 2: OLMo 2 1B, as the spec defaults** (`allenai/OLMo-2-0425-1B`, Apache 2.0). The only candidate with fully open data, dense checkpoints, a clean 90% branch point, a full fine-tune that fits in 16 GB, and bash near the top of the open group (11/20). Weak spots: little code (2.1%) and no grouped-query attention, so its KV cache is large (128 KB per token).

## B. Efficiency

All numbers measured on this RTX 5080 unless a link is given ([scripts](research/efficiency/scripts/), [report](research/efficiency/report.md)).

| Option | Measured effect | sm_120 status | Verdict |
|:-|:-|:-|:-|
| bf16 | GEMM 96 to 118 TFLOPS | Mature in torch 2.14.0+cu130 | Fallback |
| FP8 tensorwise (torchao 0.18) | OLMo 2 1B: 9.2k to 15.1k tok/s (1.64x). Qwen3-1.7B 1.61x. 135M to 360M: 1.05x to 1.31x | Works | Use for Tier 2 CPT, after a loss check against bf16 (one short test ended at a higher loss) |
| NVFP4, MXFP8 training | n/a | Broken or emulated ([TE #3062](https://github.com/NVIDIA/TransformerEngine/issues/3062), [ao #4929](https://github.com/pytorch/ao/issues/4929), [torchao docs](https://docs.pytorch.org/ao/main/workflows/training.html)) | Skip |
| Liger fused cross-entropy | Peak VRAM down 42 to 55%, speed up 14 to 21%. Without it, 135M and 360M ran out of memory at batch 4 | Works | Use |
| torch.compile | Whole model 0.83x (slower). Each decoder layer: 1.18x. Full Liger plus compile crashes | Works | Compile per layer, Liger cross-entropy only |
| Attention | Built-in SDPA flash: 3.11 ms. cuDNN 3.58 ms. FA4 beta 13% slower. flash-attn 2 has no wheel for torch 2.14 + CUDA 13 ([#2361](https://github.com/Dao-AILab/flash-attention/issues/2361)) | SDPA mature | Use SDPA |
| Sequence packing | flex_attention with reset position ids: exact, same speed as unpacked. SDPA mask path 4x slower | Works | Use flex_attention |
| 8-bit AdamW | Optimizer state for 1.49B params: 11.9 GB to 3.0 GB ([bnb](https://github.com/bitsandbytes-foundation/bitsandbytes)) | Works (bnb 0.50.2, torchao) | Use at Tier 2 |
| Gradient checkpointing | Costs about 20% speed ([HF docs](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing)) | Mature | Tier 2 only |
| LoRA vs full FT | LoRA matches full FT in RL, even at rank 1, and in small SFT ([Thinking Machines](https://thinkingmachines.ai/blog/lora/)). Frees about 6 GB at 1.5B | Mature | Full FT for CPT and all of Tier 1. LoRA for Tier 2 SFT and RL |
| vLLM 0.34 rollouts | OLMo 2 1B: 4.6k tok/s, limited by KV cache. Sleep frees 5.5 GB in under 0.4 s, wake 0.06 s. fp8 KV cache needs nvcc | Works | Use, colocated with the trainer |
| Unsloth GRPO | 2 steps ran. Pins older TRL 1.15 and vLLM 0.33; error at exit ([PyPI](https://pypi.org/project/unsloth/)) | Works, with lag | Plan B if memory is tight |

RL libraries: TRL 1.18.1 `GRPOTrainer` fits best. It has vLLM colocate with sleep mode and an `environment_factory` for multi-turn tool use ([docs](https://github.com/huggingface/trl/blob/v1.18.1/docs/source/grpo_trainer.md)). Catch: tool calls run one rollout at a time within a turn ([source](https://github.com/huggingface/trl/blob/v1.18.1/trl/trainer/grpo_trainer.py)), so Docker exec latency multiplies; keep containers warm. verl, OpenRLHF, SkyRL and prime-rl are built for clusters or two GPUs ([prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)). ART is a clean second choice but pins torch 2.11 ([pyproject](https://github.com/OpenPipe/ART/blob/main/pyproject.toml)).

### Recommended stack

Python 3.12, torch 2.14.0+cu130, vllm 0.34.0, trl 1.18.1, transformers 5.22.2, peft 0.21.3, accelerate 1.16.0, liger-kernel 0.8.5, torchao 0.18.0, bitsandbytes 0.50.2. Installed in 13 minutes with nothing compiled.

- **Continued pretraining:** full FT, Liger cross-entropy, per-layer compile, packing with flex_attention. Tier 2 adds FP8, gradient checkpointing and 8-bit AdamW. Tier 2 speed: about 15k tok/s, so 100M tokens in about 2 hours and 1B tokens in about 18 hours.
- **Bash SFT:** TRL SFTTrainer with packing. Full FT at Tier 1, LoRA rank 32 to 64 at Tier 2.
- **GRPO:** TRL with vLLM colocate and sleep mode, KL weight 0 (no reference model), LoRA at Tier 2, a Docker bash environment class with hidden graders.
- **Memory, Tier 2 GRPO with LoRA:** about 11.7 GB while generating, about 7 GB while training. Full FT fits at about 13 GB, with almost no margin. Main runs need the GPU to themselves; other agents held up to 2.4 GB during our tests.

## C. Data and existing assets

### Geodesic release (Tice et al. 2026)

- The site's model and data links point to `huggingface.co/geodesic/...` and return HTTP 401. The real org is [`geodesic-research`](https://huggingface.co/geodesic-research), collection [Alignment Pretraining](https://huggingface.co/collections/geodesic-research/alignment-pretraining-6946539245f8c4b27eb86d7c).
- **None of their datasets or models declares a licence**, and the promised open-source repo with filter term lists does not exist ([paper App. H.3](https://arxiv.org/html/2601.10160)). The paper itself is CC BY 4.0.
- **Eval:** 4,174 two-choice questions (2,671 article, 1,503 textbook), scored by the A/B letter log-prob over 8 prompt variants ([dataset](https://huggingface.co/datasets/geodesic-research/discourse-grounded-misalignment-evals), [paper §2.1](https://arxiv.org/html/2601.10160)). About 500 carry reward hacking or test gaming tags, a ready-made shortcut subset.
- **Headline numbers** (base models, article split): unfiltered 44.7%, filtered 30.9%, positive upsampled 9.2%, positive CPT 0.9%, negative upsampled 50.8% ([model card](https://huggingface.co/geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base)). The spec's "45% to 9%" and "final 10% captures most of the benefit" are confirmed in the [paper](https://arxiv.org/html/2601.10160). The "Teaching Claude why" figure of 65% to 19% is confirmed on the [Anthropic page](https://www.anthropic.com/research/teaching-claude-why).
- **Synthetic docs:** 14.9M docs, 11B tokens, gated; an ungated copy of the positive midtraining split has 327k docs ([dataset](https://huggingface.co/datasets/geodesic-research/synth-scenario-docs-positive-alignment-midtraining)). They were generated one by one from the article eval questions. No neutral split.
- **Filter assets:** a 143k-doc AI-discourse labelled set with 529 human-labelled test docs ([dataset](https://huggingface.co/datasets/geodesic-research/sfm-alignment-labeling-v3)), usable for our optional filtered condition.

### Bash SFT and RL data

| Use | Dataset | Size | Licence |
|:-|:-|:-|:-|
| SFT, single command | [NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA) | 40.6k pairs | MIT |
| SFT, command knowledge | [tldr pages](https://github.com/tldr-pages/tldr) | 6,849 pages | CC BY 4.0 |
| SFT, short trajectories | [AgentInstruct OS](https://huggingface.co/datasets/zai-org/AgentInstruct) | 195 (fit 4k context) | none declared, use only, don't rehost |
| SFT, trajectories after length filter | [OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT), [Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | 15k, 366k | Apache 2.0, CC BY 4.0 |
| RL tasks | [InterCode-Bash](https://github.com/princeton-nlp/intercode), [OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL), [Endless Terminals](https://huggingface.co/datasets/allenai/open-instruct-endless-terminals) | 200, about 720, about 2.5k | MIT, Apache 2.0, Apache 2.0 |
| Skip | SWE-smith, SWE-Gym, nebius SWE trajectories | | Too long for 4k context, and "make the failing tests pass" is our own eval scenario |

Screen all RL tasks for log, audit, CI, lint and answer-file themes, and harden weak verifiers: GRPO on hackable verifiers would teach shortcuts in every condition ([data report §2.4](research/data/report.md)). Never mix in data that shows reward hacking ([AISI SDF docs](https://huggingface.co/datasets/ai-safety-institute/reward-hacking-sdf-default), [School of Reward Hacks](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks)); use it for judge checks at most.

### Replay corpora

- **Tier 2 (OLMo 2 1B):** a [Dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) sample (the model's own final-stage data, ODC-By; its StackExchange part is CC BY-SA), the shell subset of the StarCoder data in [OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) (469 MB gz, ungated, about 0.4B tokens), and [Tulu 3 OLMo 2 SFT mix](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225) for chat.
- **Tier 1 (SmolLM2-135M):** one [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) shard (about 0.7B tokens, ODC-By), the same OLMo shell subset for code, and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) for chat. Stack-Edu holds only file IDs, so it is skipped.

### Reusable code

[believe-it-or-not](https://github.com/safety-research/believe-it-or-not) (MIT, synthetic document generation), [agentic-misalignment](https://github.com/anthropic-experimental/agentic-misalignment) (MIT), [AISI reward-hacking-misalignment](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment) (MIT, hackable RL environments and judge prompts), [Harbor](https://github.com/harbor-framework/harbor) (Apache 2.0 terminal task runner). [ImpossibleBench](https://github.com/safety-research/impossiblebench) is the same family as our test-editing scenario: design reference only.

### Data plan

1. Generate our own matched story corpora (positive, neutral, negative) with the believe-it-or-not pattern. Only the AI's choice differs; length, form and topic are matched. A canary string goes in every story and eval file.
2. Use Geodesic assets by Hugging Face id and pinned revision; don't rehost them until they carry a licence.
3. The repo ships code, recipes and our own generated data (MIT or CC BY 4.0). It does not rehost unlicensed, GPL (NL2Bash) or CC BY-SA data.

## Conflicts with the spec (flagged, not changed)

1. **Story dose.** 1 to 5% of 100 to 200M tokens is 1 to 10M story tokens. Geodesic's CPT arm used 500M story tokens in a 1B-token run ([paper §4.1](https://arxiv.org/html/2601.10160)): 50 to 500 times more. A null result at our dose would say little. At 15k tok/s, a 1B-token Tier 2 CPT run takes about 18 hours, so a larger dose is affordable in GPU time; generating the stories is the cost.
2. **Order B branch point.** OLMo 2 1B's 90% checkpoint comes before its 50B-token anneal, which the paper credits with a +37% benchmark gain ([paper App. B](https://arxiv.org/html/2501.00656v3)). It scores 8/20 on bash vs 11/20 for the final model, and redoing the anneal would take about 50 GPU-days here. Geodesic's late CPT arm, run on the final model, kept most of the effect: 15.2% vs 13.2% misalignment after post-training ([paper](https://arxiv.org/html/2601.10160)). Suggested default: Order B continues from the final model; the 90% branch becomes optional.
3. **Near vs far transfer.** Geodesic's docs were written from their eval questions. Our spec keeps eval scenario types out of the stories, which is a stricter, far-transfer test, so expect a smaller effect. Not a change, a risk to state in the preregistration.
4. **Letter-answer MCQ at small scale.** Geodesic added 1B tokens of MCQ data so letter scoring worked. At 135M to 1.5B, letter answers may sit at chance. Suggested default: score the full answer text by log-prob instead of the letter.
5. **No neutral arm upstream.** Geodesic had no neutral control, so we generate it. Fine, just noted.
6. **Geodesic licences.** The spec says to reuse their eval "if the licence allows". It declares no licence. Suggested default: load it at run time, don't redistribute, and ask the authors to add a licence.
7. **Docker is not running.** The harness needs `docker.service` enabled and the user in the `docker` group. That needs sudo, which repo agents don't use.

## Assumptions

- Throughput for Tier 2 assumes the GPU is free of other jobs. Story generation speed is not measured yet.
- FP8 is assumed safe only after a 1-hour loss check against bf16.
- OLMo 2 1B's `main` is the reference "final model". The repo has three stage 2 runs and does not say which one became `main` ([refs](https://huggingface.co/api/models/allenai/OLMo-2-0425-1B/refs)).

## Sites that blocked us

None blocked us as a bot, so no browser pass was needed. Thin spots: arXiv `/abs` pages only give abstracts (the `/html/` versions worked); some model cards (Qwen3, Qwen3.5, MiniCPM5, Granite) publish no data shares; several Hugging Face datasets are gated behind accepting terms on a logged-in account (Geodesic synthetic data, METR MALT, BigCode Stack v2). We did not accept terms on the owner's account.
