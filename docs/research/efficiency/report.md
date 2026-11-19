# Research B: training and RL efficiency on one RTX 5080 (sm_120, 16 GB)

Date: 2026-11-19. All "measured" numbers come from my own micro-benchmarks on this machine (scripts and logs in this folder). Everything else has a source link.

## Bottom line

1. FP8 training works on sm_120 today and pays off at Tier 2: +61% to +64% tokens/s at 1.5B to 1.7B params. At Tier 1 it gives only +5% to +31%. Measured, torch 2.14.0 + torchao 0.18.0, tensorwise recipe.
2. Liger's fused linear cross-entropy is the biggest memory win for small models: it cuts peak VRAM by 42% to 55%. Combined with per-layer torch.compile it gives +14% to +21% speed. Compiling the whole model was slower than eager, and Liger's full kernel set crashes under torch.compile.
3. Skip the flash-attn packages. PyTorch SDPA already runs a FlashAttention-2 kernel on sm_120. FA2 has no wheel for torch 2.14 + CUDA 13. FA4 runs on sm_120 but was 13% slower for training.
4. For packing, use `attn_implementation="flex_attention"` with reset `position_ids`. It is exact and as fast as unpacked. The SDPA mask path is correct but 4x slower.
5. For RL, use TRL 1.18.1 GRPOTrainer with vLLM 0.34.0 in colocate mode plus sleep mode. Write the Docker bash sandbox as an `environment_factory` class. vLLM 0.34 works on sm_120, and sleep and wake take under 0.4 s. Use LoRA for Tier 2 RL; full fine-tuning fits, but only just.
6. Hard limits on sm_120: NVFP4 and MXFP8 training are broken or emulated. vLLM's fp8 KV cache needs nvcc, which we don't have. Latest Unsloth cannot share an environment with vLLM 0.34.

## Option table

| Option | Expected speedup (this GPU) | sm_120 maturity | Verdict |
|:-|:-|:-|:-|
| bf16 baseline | GEMM 111 to 118 TFLOPS measured | Mature (torch 2.14.0+cu130 has sm_120 in its arch list) | Fallback for everything |
| FP8 tensorwise (torchao `convert_to_float8_training`) | Measured: 1.05x to 1.16x at 135M, 1.31x at 360M, 1.64x at OLMo-2 1B, 1.61x at Qwen3-1.7B. Raw GEMM 2.4x to 3.5x | Works (torch 2.14, torchao 0.18). No sm_120 benchmarks published; torchao reports +25% at 8B on H100 ([docs](https://docs.pytorch.org/ao/main/workflows/training.html)) | Use for Tier 2 CPT and SFT after a loss-parity check |
| FP8 rowwise | Measured 1.17x at 1B. GEMM about 2x | Works | Use only if tensorwise diverges |
| NVFP4 / MXFP8 training | n/a | Broken. TE's default NVFP4 recipe crashes on sm_120 ([TE #3062](https://github.com/NVIDIA/TransformerEngine/issues/3062)). torchao mxfp8 has open sm_120 GEMM issues ([ao #4929](https://github.com/pytorch/ao/issues/4929), [ao #4938](https://github.com/pytorch/ao/issues/4938)). torchao lists NVFP4 training as emulation only ([docs](https://docs.pytorch.org/ao/main/workflows/training.html)) | Skip |
| Liger Kernel 0.8.5 (fused linear CE, RMSNorm, RoPE, SwiGLU) | Measured +14% (135M) and +21% (360M) over eager. Peak VRAM 6.76 to 3.05 GB at 135M. Upstream claims +20% throughput and -60% memory ([README](https://github.com/linkedin/Liger-Kernel)) | Works. Triton kernels; supports OLMo2, OLMo3, Qwen3 and Llama (so SmolLM2). Not GPT-NeoX/Pythia | Use, but only FLCE when combined with compile |
| torch.compile | Whole model: 0.83x (slower). Per decoder layer: 1.18x over eager | Works. Liger's full kernel set plus compile crashes with a dynamo error ("_cuda_exchange_device ... does not belong to this Graph") | Compile each decoder layer; keep the Liger FLCE head eager |
| SDPA flash backend (built into torch) | Causal fwd+bwd 3.11 ms vs cuDNN 3.58 ms vs mem-efficient 6.09 ms (measured, 8x16x2048x64) | Mature | Default attention |
| flash-attn 2.8.3 package | Same kernel family as SDPA flash | No wheel for torch 2.14/cu13; must compile; nvcc segfaults reported on sm_120 ([#2361](https://github.com/Dao-AILab/flash-attention/issues/2361)); no system nvcc here | Skip |
| FA4 (`flash_attn_4` 4.0.0b40, CuTe DSL) | Measured fwd+bwd 6.95 ms vs SDPA flash 6.13 ms (hdim 128). 0.88x | Beta. Runs fwd and bwd on sm_120 (max error 0.002), but the sm_120 backward path is still in open PRs ([#3109](https://github.com/Dao-AILab/flash-attention/pull/3109), [#3144](https://github.com/Dao-AILab/flash-attention/pull/3144)) | Skip for now |
| Sequence packing (position_ids reset) | 1.4x to 2x throughput on variable-length SFT ([HF blog](https://huggingface.co/blog/packing-with-FA2)). Measured: flex_attention packed 136 ms vs unpacked 136 ms, outputs exact | flex_attention works via Triton. Transformers SDPA path builds a dense block mask: correct, but 517 ms vs 130 ms | Use packing with flex_attention |
| Gradient checkpointing | Costs about 20% speed ([HF docs](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing)). Required for full-FT 1B+ | Mature | Use at Tier 2 only |
| 8-bit AdamW (bitsandbytes 0.50.2 / torchao 0.18) | Optimizer state drops from 8 to 2 bytes/param: 1.49B params goes from 11.9 to 3.0 GB ([bnb](https://github.com/bitsandbytes-foundation/bitsandbytes), [torchao optim](https://github.com/pytorch/ao/tree/main/torchao/optim)). Measured: bnb 9.2k tok/s, torchao 8.9k tok/s | Both work | bnb for speed; torchao `bf16_stochastic_round=True` for long pure-bf16 CPT |
| LoRA vs full FT | LoRA takes about 2/3 of the FLOPs per pass. It matches full FT in RL even at rank 1, and in small or medium SFT; it falls behind when the data exceeds its capacity ([Thinking Machines](https://thinkingmachines.ai/blog/lora/)). Frees about 6 GB at 1.5B (grads plus Adam) | Mature (PEFT 0.21.3) | LoRA for Tier 2 SFT and RL; full FT for CPT and for all Tier 1 work |
| vLLM 0.34.0 colocate + sleep | Colocate is up to 1.43x faster than server mode at 1.5B ([HF blog](https://huggingface.co/blog/vllm-colocate)). Measured: sleep frees 5.5 GB in 0.07 to 0.39 s; wake takes 0.06 s | Works on sm_120 (FA2 backend). fp8 KV cache fails: FlashInfer JIT needs nvcc | Use |
| Unsloth 2026.11.6 GRPO (standby) | Shares weights between vLLM and the trainer (saves about 3 GB at 1.5B). Claims 1.2x to 1.7x longer context and about 10% faster RL ([docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)) | Works (measured 2-step GRPO). Pins TRL 1.15.0, transformers 5.17 and vLLM 0.33 ([PyPI](https://pypi.org/project/unsloth/)). cumem "invalid argument" error at process exit | Plan B if the TRL colocate memory is too tight |
| SGLang 0.5.23 | n/a | Not tested. Pins torch 2.14.1 ([PyPI](https://pypi.org/project/sglang/)) | No reason to prefer it over vLLM here |

## Test results (measured on this machine)

Environment: uv venv with Python 3.12.15. `uv pip install vllm==0.34.0 trl==1.18.1 liger-kernel torchao bitsandbytes peft accelerate datasets` resolved to torch 2.14.0+cu130, triton 3.8.0, transformers 5.22.2, liger-kernel 0.8.5, torchao 0.18.0, bitsandbytes 0.50.2, flashinfer-python 0.7.3. It installed in 13 minutes, nothing compiled from source. The arch list includes sm_120. During the tests 2.4 to 2.9 GB of VRAM was already in use by the desktop and another agent.

GEMM (`torch._scaled_mm`, bf16 output). bf16 runs at 96 to 118 TFLOPS. FP8 tensorwise runs at 258 to 408 TFLOPS (2.4x to 3.5x). FP8 rowwise runs at 182 to 234 TFLOPS (1.9x to 2.0x). Script: `t1_env_gemm.py`.

Train step: random init, full fine-tuning, seq 2048, fused AdamW unless noted. "6ND TF" is model FLOPs per second. Script: `t2_model_bench.py`. Logs: `log_*.txt`.

| Model | Setup | tok/s | 6ND TF | Peak VRAM |
|:-|:-|:-|:-|:-|
| SmolLM2-135M, B2 | eager | 54.0k | 44 | 6.76 GB |
| | torch.compile whole model | 44.8k | 36 | 4.39 GB |
| | Liger (all kernels) | 61.8k | 50 | 3.05 GB |
| | per-layer compile | 63.8k | 52 | 5.81 GB |
| | Liger FLCE + per-layer compile + FP8 | 64.9k | 52 | 2.43 GB |
| SmolLM2-135M, B8 | Liger | 77.0k | 62 | 9.34 GB |
| | Liger FLCE + per-layer compile + FP8 | 89.7k | 72 | 6.85 GB |
| SmolLM2-360M, B2 | eager / Liger | 26.5k / 32.0k | 58 / 70 | 10.6 / 6.1 GB |
| SmolLM2-360M, B4 | Liger / + per-layer compile + FP8 | 34.2k / 44.9k | 74 / 98 | 10.0 / 7.7 GB |
| OLMo-2-0425-1B (1.49B), B4, ckpt, bnb AdamW8bit | Liger | 9.2k | 82 | 9.4 GB |
| | Liger FLCE + per-layer compile + FP8 tensorwise | 15.1k | 134 | 9.6 GB |
| | same with FP8 rowwise | 10.8k | 96 | 9.8 GB |
| | torchao AdamW8bit instead of bnb (bf16) | 8.9k | 80 | 9.5 GB |
| Qwen3-1.7B, B4, ckpt, bnb AdamW8bit | bf16 | 6.8k | 70 | 11.3 GB |
| | + FP8 tensorwise | 10.9k | 112 | 11.2 GB |

What this means for CPT at Tier 2: OLMo-2 1B full fine-tuning runs at about 1.3B tokens/day with FP8, against 0.8B tokens/day in bf16. Without Liger, eager mode ran out of memory at 135M B4 and 360M B4: the 49k to 152k vocab logits dominate memory. Caveat: on this memorization micro-test FP8 sometimes ended at a higher loss after 13 steps (135M: 3.7 vs 2.9). Check loss parity against bf16 on a 1-hour real run before trusting FP8.

Attention:
- SDPA flash backend: 3.11 ms. cuDNN: 3.58 ms. Mem-efficient: 6.09 ms.
- FA4 4.0.0b40 at hdim 128: fwd 1.50 ms, fwd+bwd 6.95 ms. SDPA flash fwd+bwd: 6.13 ms.
- Scripts: `t1_env_gemm.py`, `t4_fa4.py`.

Packing (`t5_packing.py`, SmolLM2-135M, 8 docs x 1024 tokens packed into one row):
- flex_attention: max hidden-state difference vs separate docs is 0.0000. Step time 136 ms packed vs 136 ms unpacked.
- SDPA: difference 0.25 (bf16 noise), but 517 ms vs 130 ms.
- Without `position_ids`, both mix documents: difference 7.5.
- Also mask labels at document boundaries. TRL's `padding_free` does this.

vLLM 0.34.0 (`t3_vllm_sleep.py`):
- SmolLM2-135M at util 0.25: 64 x 256 tokens decode at 40k tok/s.
- OLMo-2 1B at util 0.4 (6.2 GB): KV cache holds 18,176 tokens. 64 x 1024 tokens run at 4.6k tok/s; throughput is KV-bound at about 17 concurrent sequences.
- sleep(level=1) drops vLLM from 6.2 to 0.6 GB in 0.07 to 0.39 s. wake_up takes 0.06 s.
- Engine init takes 83 s, mostly CUDA graph capture. Use `enforce_eager` for short iteration runs if that matters.
- `kv_cache_dtype="fp8"` failed: it switches to FlashInfer, which JIT-compiles and needs nvcc.

Unsloth (`t6_unsloth_grpo.py`, separate venv `.venv-unsloth`):
- unsloth 2026.11.6 + vLLM 0.33.0 + TRL 1.15.0 + transformers 5.17.0 ran 2 GRPO steps with LoRA r16 on SmolLM2-135M, with standby on.
- Peak was 9.0 GB at util 0.5. Model load took 126 s.
- xformers would not load with torch 2.14.
- Process exit printed "CUDA Error: invalid argument at cumem_allocator.cpp". This is the same sm_120 cumem class of bug reported in [this vLLM sleep-mode write-up](https://doradusresearch-ai.pages.dev/blog/vllm-blackwell-sleep-mode/). Training was not affected.

## RL libraries for one GPU with multi-turn bash in Docker

- **TRL 1.18.1 GRPOTrainer: recommended.**
  - Has `vllm_mode="colocate"`, `vllm_enable_sleep_mode`, and `vllm_gpu_memory_utilization` (default 0.3). `beta` defaults to 0, so no reference model is loaded.
  - `environment_factory` builds one stateful environment per rollout. Its public methods become tools, `reset()` returns the prompt, and `get_reward()` owns the reward. TRL runs the multi-turn tool loop and masks tool tokens. `max_tool_calling_iterations` caps the turns. Source: [GRPO docs](https://github.com/huggingface/trl/blob/v1.18.1/docs/source/grpo_trainer.md).
  - Catch, from reading `_tool_call_loop` in [grpo_trainer.py](https://github.com/huggingface/trl/blob/v1.18.1/trl/trainer/grpo_trainer.py): tool calls run one rollout after another within each turn. Async gather only covers calls inside one rollout. So 32 rollouts x 0.5 s of `docker exec` adds 16 s per turn.
  - Fixes: keep one warm container per environment and keep exec fast; or use the experimental `rollout_func` and run Docker calls concurrently yourself.
- **Unsloth GRPO.** Same TRL API (pinned to 1.15), plus standby weight sharing. LoRA only with `fast_inference`. Release lag is a real risk: it pins old TRL, transformers and vLLM.
- **ART 0.5.24 (OpenPipe).** The cleanest agent API: you write the multi-turn loop against an OpenAI-style client, and a local backend trains LoRA with Unsloth and vLLM. But it pins torch 2.11 and unsloth 2026.3.3 ([pyproject](https://github.com/OpenPipe/ART/blob/main/pyproject.toml)). Second choice.
- **verl 0.9.1.** Has multi-turn tool calling, an agent loop and sandbox integration ([README](https://github.com/verl-project/verl)). It is Ray-based, pins transformers<5.11 and Python<3.13, and is built for clusters. Not tested.
- **OpenRLHF 0.11.2.** Has multi-turn agents and a hybrid engine that shares GPUs ([README](https://github.com/OpenRLHF/OpenRLHF)). Ray-based. Not tested.
- **SkyRL 0.5.0.** Has multi-turn tool use (skyrl-gym, skyrl-agent) and a Tinker-API backend ([README](https://github.com/NovaSky-AI/SkyRL)). Heavy setup.
- **prime-rl 0.10.0.** Uses verifiers environments and is tested on 5090s. Its examples assume one trainer GPU plus one inference GPU, and the full end-to-end RL check needs 2 GPUs ([README](https://github.com/PrimeIntellect-ai/prime-rl)). Poor fit.
- Docker needs no nvidia-container-toolkit, because the sandbox only runs bash. The GPU stays in the trainer process.

## Recommended stack

One venv with Python 3.12 (vLLM requires <3.15; verl would need <3.13):
- torch 2.14.0+cu130. Pinned by vLLM 0.34.0; torch 2.15.0 exists but vLLM does not allow it yet.
- vllm 0.34.0, trl 1.18.1, transformers 5.22.2, peft 0.21.3, accelerate 1.16.0.
- liger-kernel 0.8.5, torchao 0.18.0, bitsandbytes 0.50.2.

Per stage:
- **CPT, Tier 1:**
  - bf16, full FT, fused AdamW.
  - Liger FLCE plus per-layer `torch.compile`.
  - FP8 is optional (+5% to +31%).
  - Pack with flex_attention and reset `position_ids`.
  - Pythia (GPT-NeoX) has no Liger patch; prefer SmolLM2-135M/360M.
- **CPT, Tier 2:**
  - Full FT with FP8 tensorwise, Liger FLCE, per-layer compile, gradient checkpointing and 8-bit AdamW.
  - For runs over a day, use torchao AdamW8bit with `bf16_stochastic_round=True`.
  - Keep bf16 as the fallback.
- **SFT:**
  - TRL SFTTrainer with packing / `padding_free` and flex_attention.
  - Full FT at Tier 1; LoRA r32 to 64 on all linear layers at Tier 2. Use an LR about 10x the full-FT LR ([Thinking Machines](https://thinkingmachines.ai/blog/lora/)).
- **GRPO:**
  - TRL GRPOTrainer with colocate, `vllm_enable_sleep_mode=True` and `vllm_gpu_memory_utilization` around 0.35 to 0.45.
  - `beta=0` and gradient checkpointing. Don't set `use_liger_kernel`. TRL 1.18 already computes logprobs with its own chunked fused LM head that never builds the full logits, and it deprecates `use_liger_kernel` in favour of Hub kernels (`model_init_kwargs={"use_kernels": True}`, untested on sm_120). Source: [grpo_trainer.py](https://github.com/huggingface/trl/blob/v1.18.1/trl/trainer/grpo_trainer.py).
  - LoRA at Tier 2.
  - `environment_factory=DockerBashEnv`: tool `bash(cmd)`, and `get_reward()` runs the hidden graders outside the model's view.
  - Model choice for RL: Qwen3-1.7B has GQA (112 KB KV per token) and a TRL-patched tool template. OLMo-2 1B has no GQA (128 KB per token). SmolLM2-1.7B is the worst at 192 KB per token.
- **Plan B for RL memory:** a separate Unsloth venv (unsloth 2026.11.6, trl 1.15.0, vllm 0.33.0).

## Memory budget: Tier 2 GRPO in 16 GB

Usable VRAM is about 13.5 GB: 16.3 GB total minus about 2 GB for the desktop. Main runs need the GPU to themselves; during my tests another agent held 0.5 to 2.4 GB.

Option A is the default: OLMo-2 1B (1.49B params) or Qwen3-1.7B, LoRA r32, TRL colocate with sleep.

| Item | Generation phase | Train phase |
|:-|:-|:-|
| Policy base weights, bf16 | 3.0 GB | 3.0 GB |
| LoRA params, grads, Adam | 0.2 GB | 0.2 GB |
| vLLM weight copy, bf16 | 3.0 GB | 0 (sleep offloads to CPU) |
| vLLM KV cache | 4.0 GB (about 32k tokens for OLMo-2, about 37k for Qwen3-1.7B) | 0 (sleep discards it) |
| vLLM CUDA graphs and context | 0.8 GB | 0.6 GB (measured residual) |
| Trainer activations (ckpt, TRL fused LM head, 8k to 16k tokens per micro-batch) | 0 | 2.0 to 3.0 GB |
| Trainer CUDA context and allocator slack | 0.7 GB | 0.7 GB |
| **Total** | **about 11.7 GB** | **about 6.5 to 7.5 GB** |

Option B is full FT of OLMo-2 1B with bnb AdamW8bit and the same colocate setup.
- Generation phase: 3.0 (weights) + 3.0 (Adam) + 3.0 (vLLM weights) + 2.5 (KV, about 20k tokens) + 1.5 (overhead) = about 13 GB.
- Train phase: 3.0 + 3.0 + 3.0 (grads) + 2.0 (activations) + 1.3 = about 12.3 GB. The measured full-FT step peak at 8k tokens was 9.4 to 9.6 GB.
- It fits, but with almost no margin and half the rollout KV of Option A.
- Unsloth standby would save the 3 GB vLLM weight copy, but only with LoRA.

KV sizing: G=8 rollouts x 4k tokens per episode needs 32k tokens. So Option A runs one prompt group at a time at full concurrency. Rollout throughput is KV-bound: measured 4.6k tok/s for OLMo-2 1B with 2.2 GB of KV.

## Risks

- FP8 numerics. Tensorwise is the fastest recipe and the least robust to outliers. Verify loss parity with bf16 before main runs. Keep the `lm_head` in bf16 (done in my test).
- Version lock. vLLM pins an exact torch (0.34.0 needs torch 2.14.0). Unsloth lags TRL and transformers by several releases. ART lags further. Pin everything in a lockfile.
- sm_120 sleep-mode bugs. Others report cumem "invalid argument" errors and memory leaking on repeated sleep/wake cycles ([write-up](https://doradusresearch-ai.pages.dev/blog/vllm-blackwell-sleep-mode/)). I saw the error only at process exit. Watch VRAM across hundreds of cycles in the first long run.
- Liger plus torch.compile crash with torch 2.14. Use FLCE only with per-layer compile, or Liger without compile.
- No nvcc. Anything that JIT-compiles CUDA fails: FlashInfer fp8 KV, xformers, the flash-attn build. A pip `nvidia-cuda-nvcc` wheel might fix FlashInfer (untested).
- vLLM regressions on sm_120. One report found v0.18.1 three times slower than v0.17.1 on SM120. The same report got 6.5x from locking GPU clocks on a headless box ([vLLM forum](https://discuss.vllm.ai/t/sm120-rtx-pro-4000-6-5x-throughput-gain-and-v0-18-1-regression-findings/2525)). Benchmark again after each vLLM upgrade.
- TRL tool calls run one rollout at a time. The Docker exec latency multiplies by the number of rollouts per turn.
- Shared GPU. Other agents' processes (2.4 GB seen) can push Option B into out-of-memory errors.

## Sites that blocked or gave thin results (for a browser pass)

No site blocked me as a bot. WebFetch, curl, the GitHub API and the PyPI JSON API all worked. Thin results:
- WebSearch for measured FP8 training speedups on sm_120 (5080/5090) found none. My numbers seem to be the only ones.
- WebSearch on vLLM colocate GRPO on a 16 GB Blackwell card found no reports.
- `gh search issues` on sgl-project/sglang for sm120/5090 returned nothing, so SGLang's sm_120 status is unverified.
- The Unsloth docs (unsloth.ai/docs) say nothing about multi-turn tool-calling GRPO.
- The TE issue #3062 search hit was a mirror (net.nhd.workers.dev). I read the original through the GitHub API instead.
- I did not try the x.com and notion.site links (Unsloth and SkyRL announcements).
