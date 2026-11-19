# Browser pass (Claude in Chrome), 2026-11-19

Second pass over pages that curl/WebFetch could not read. Browser: the Linux extension instance already selected for the session ("Browser 4"; the extension does not report Chromium vs Chrome by name). Read only; one Notion cookie banner was answered with "Reject all". Numbers read from charts are approximate and marked "~".

## 1. Terminal-Bench leaderboard

| Field | Value |
|---|---|
| URLs | https://www.tbench.ai/leaderboard/terminal-bench/2.0 (redirects to `/?version=2.0`), `/terminal-bench/1.0` |
| Worked | Yes for 2.0 and for the current default (4.1). 1.0 no longer exists: the version picker offers only 4.1, 4.0, 3.0, 2.1, 2.0, and `?version=1.0` falls back to 4.1. |
| Entries at 4B or less | None on 2.0 (137 entries) or 4.1 (52 entries). The site lists no parameter counts; the TOKENS column is tokens consumed, not model size. |
| Smallest models on 2.0 | Qwen3.6-35B-A3B + little-coder 24.6% ± 3.2 (3B active, 35B total); Qwen3.5-9B + little-coder 9.2% ± 2.4 (smallest dense model with a known size); GPT-OSS-20B + Mini-SWE-Agent 3.4% ± 1.4, + Terminus 2 3.1% ± 1.5 (last place); GPT-5-Nano 7.0% to 21.8% depending on harness (spoox-o-m 21.8, Codex CLI 11.5, OpenHands 9.9, Terminus 2 7.9, Mini-SWE-Agent 7.0); TermiGen-32B + Bash Agent 19.3% |
| 2.0 top | GPT-5.5 + NexAU-AHE 84.7% ± 2.1 |
| 4.1 bottom (sizes not given) | Inkling Small + mini-swe-agent 0.9%; Nemotron 3.5 Super VL 0.3%; Inkling 0.0%; GPT-5.4 Mini + Codex 1.8% |
| Note | Terminal-Bench gives no signal for models of 4B or less: dense 9B already scores 9%, and 20B scores about 3%. |

## 2. xLAM-2-1b-fc-r (Salesforce)

| Field | Value |
|---|---|
| URL | https://huggingface.co/Salesforce/xLAM-2-1b-fc-r |
| Worked | Yes (images read by zooming in on screenshots) |
| BFCL v3, xLAM-2-1b-fc-r (FC) | Rank 36, Overall Acc 58.90; Non-live AST 76.23; Non-live Exec 74.86; Live AST 59.88; Multi-turn 43.12; Relevance 88.89; Irrelevance 56.87 |
| BFCL v3, xLAM-2-3b-fc-r | Rank 14, Overall 65.11; Non-live AST 82.94; Exec 81.88; Live 58.69; Multi-turn 56.00; Relevance 94.44; Irrelevance 57.94 |
| BFCL v3, references | xLAM-2-8b 72.83 (multi-turn 69.25); GPT-4o-2024-11-20 (FC) 69.58 (multi-turn 41); Qwen2.5-14B-Instruct (FC) 58.9 (multi-turn 15.88) |
| tau-bench, xLAM-2-1b-fc-r | Retail 22.5; Airline 21.0; Overall 21.8 |
| tau-bench, xLAM-2-3b-fc-r | Retail 44.4; Airline 32.0; Overall 38.2 |
| tau-bench, references | xLAM-2-8b 58.2 / 35.2 / 46.7; Qwen2.5-32B-Instruct 24.4 / 25.0 / 24.7; Claude 3.5 Haiku 51.0 / 22.8 / 36.9 |
| Note | The 1B model matches Qwen2.5-14B on BFCL overall and beats GPT-4o on multi-turn (43.1 vs 41). On tau-bench it scores 21.8, about the same as Qwen2.5-32B. |

## 3. Structured CodeAgent blog (smolagents)

| Field | Value |
|---|---|
| URL | https://huggingface.co/blog/structured-codeagent |
| Worked | Yes (Figure 1 is an image; values read from the bars) |
| Benchmark | SmolBench (GAIA, MATH, SimpleQA, Frames), pass@1, 95% CI |
| Values (Structured CodeAgent / CodeAgent / ToolCallingAgent) | Claude 4 Sonnet ~0.68 / 0.67 / 0.37; Claude 3.7 Sonnet ~0.67 / 0.65 / 0.32; o3 ~0.57 / 0.46 / 0.65; o4-mini ~0.64 / 0.59 / 0.62; Claude 3.5 Sonnet ~0.62 / 0.59 / 0.35; Qwen3-235B-A22B ~0.55 / 0.48 / 0.45; **Qwen3-4B ~0.46 / 0.55 / 0.33**; o1 ~0.55 / 0.51 / 0.41; GPT-4o ~0.53 / 0.53 / 0.34; Qwen3-32B ~0.53 / 0.52 / 0.26; GPT-4.1 ~0.52 / 0.44 / 0.38 |
| Parsing stats (Figure 2) | 2.4% of 15,724 traces had a parse error on the first call; success 51.3% without vs 42.3% with a first-step parse error; average steps 3.18 vs 4.63 |
| Note | Qwen3-4B is the only small model, and the only one where structure hurts ("structure tax": about 9 points below the plain CodeAgent). Plain code actions beat JSON tool calls by about 22 points for Qwen3-4B. |

## 4. SWE-bench bash-only (mini-SWE-agent), Verified 500

| Field | Value |
|---|---|
| URL | https://www.swebench.com/ (default view: Verified, bash-only, mini-SWE-agent) |
| Worked | Yes |
| Entries at 4B or less | None (47 entries) |
| Smallest or weakest models | Qwen2.5-Coder 32B Instruct 9.00%; Llama 4 Scout Instruct 9.06%; Gemini 2.0 Flash 13.52%; Llama 4 Maverick 21.04%; GPT 4o 21.62%; GPT 4.1 mini 23.94%; gpt-oss-120b 26.00%; GPT 5 nano (medium) 34.80%; Devstral Small (2512, 24B) 56.40% |
| Top | Claude 4.5 Opus (high) 76.80%; Gemini 3 Flash (high) 75.80%; MiniMax M2.5 (high) 75.80% |
| Note | No small models listed. The smallest dense open model (Devstral Small 24B) reaches 56.4%, but Qwen2.5-Coder 32B only 9%. |

## 5. SkyRL-v0 (Notion)

| Field | Value |
|---|---|
| URL | https://novasky-ai.notion.site/skyrl-v0 |
| Worked | Yes (needed scrolling and expanding the three per-model toggles) |
| SkyRL-Agent-7B-v0 | From OpenHands-7B-Agent (Qwen2.5-Coder-7B-Instruct SFT). GRPO, batch 16, 8 rollouts/instance, LR 1e-6, KL 1e-3, temperature 0.5, 25 max turns. 15 steps on SkyRL-v0-80-data (3 epochs, 7 h on 8xH100) + 15 steps on SkyRL-v0-220-data (1 epoch, 8.5 h on 8xH100) = 15.5 h on 8xH100. SWE-bench Verified: Qwen2.5-Coder-Instruct 1.8% (zero-shot), OpenHands-7B-Agent 11.0% (SFT), SkyRL-Agent-7B-v0 14.6% (RL) |
| SkyRL-Agent-8B-v0 | From Qwen3-8B, no thinking. Batch 32, 8 rollouts, LR 1e-6, KL 1e-3, temperature 0.6. 47 steps on 293 instances (4.7 epochs, 27 h on 8xH200). 3.6% to 9.4% |
| SkyRL-Agent-14B-v0 | From Qwen3-14B, thinking, PPO with Qwen3-14B as critic, clip-higher 0.28. 25 steps on 293 instances (2.7 epochs, 20 h on 8xH200). 18% to 21.6% |
| Eval setup | OpenHands CodeAct, tools execute_bash, finish, str_replace_editor; 50-turn budget, 32k context, temperature 0 |
| Data | SWE-Gym (2,438 tasks) filtered to 80 / 220 / 293 instances where a model gets at least 1 of 16 rollouts right. The post says 7B-scale models often get zero successful rollouts on unfiltered data, so GRPO/PPO collapse |
| Infra | Each environment needs 1+ CPU and ~7 GB disk; batch 16 x 8 rollouts needs 100+ CPUs and ~1 TB disk; 80 to 100 containers per 16-CPU node; async rollout plus pipelining gives a 4 to 5x speedup |
| Note | Nothing below 7B. Filtering to solvable tasks is required for RL to get any signal. |

## 6. MiniCPM5-1B-Base

| Field | Value |
|---|---|
| URL | https://huggingface.co/openbmb/MiniCPM5-1B-Base |
| Worked | Yes, but the card has no base-model table. Its only table is "Evaluation Results of MiniCPM5-1B", which is for the post-trained model in thinking mode. No HumanEval or MBPP numbers. |
| Columns | MiniCPM5-1B (Thinking) / Qwen3-0.6B (Thinking) / Qwen3.5-0.8B (Thinking) / LFM2.5-1.2B (Thinking) |
| Average | 42.57 / 26.77 / 25.14 / 35.61 |
| Coding | LCB-Pro 25Q2 (Easy) 22.68 / 4.12 / 0.00 / 6.19; OJBench 7.33 / 0.86 / 0.43 / 1.94; LCB-v6 (@Avg3) 33.52 / 16.00 / 5.33 / 21.33 |
| Agentic | BFCLv4 25.15 / 25.43 / 25.53 / 10.60; tau2-Bench Telecom-AA 79.53 / 21.10 / 47.70 / 19.60 |
| Other | MMLU-Pro 48.85 / 35.63 / 42.74 / 47.98; MMLU-Redux 70.06 / 55.47 / 61.50 / 66.08; GPQA-Diamond 26.26 / 25.42 / 30.98 / 34.85; IFEval 80.41 / 59.89 / 59.89 / 84.84; IFBench 46.67 / 25.67 / 29.33 / 41.67; MATH-500 91.60 / 72.60 / 30.40 / 89.00; AIME-2025 (@Avg16) 40.42 / 16.25 / 1.04 / 31.88; BBH 71.89 / 47.86 / 54.58 / 57.32 |
| Note | All four 1B-class models score about 25 on BFCLv4, but tau2 Telecom ranges widely (MiniCPM5 79.5 vs Qwen3-0.6B 21.1). The radar chart has no numbers. |

## 7. Falcon-H1-Tiny blog

| Field | Value |
|---|---|
| URL | https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost (text served from https://tiiuae-tiny-h1-blogpost.hf.space/) |
| Worked | Yes |
| Tool calling, BFCL v3 (Size / Non-live AST / Live AST / Relevance / Irrelevance / Multi-turn / Global) | Qwen3-0.6B 600M / 71.79 / 56.62 / 75.00 / 80.84 / 3.62 / 57.57; FunctionGemma 270M / 48.40 / 26.40 / 61.10 / 70.60 / 0 / 41.30; Falcon-H1-Tiny-Function-Calling-90M 90M / 36.06 / 14.27 / 94.44 / 61.37 / 0 / 41.23 |
| Tool-call share of the SFT mix vs Global BFCL v3 (90M) | 20% gives 32.1; 50% gives 36.8; 75% gives 39.4; 85% gives 41.2 |
| Tool-calling findings | Curriculum SFT and pretrain-SFT came out about the same for tool calling. Chain-of-thought traces in the tool data caused endless repetition loops at 90M, fixed by removing the reasoning content. |
| Code (Falcon-H1-Tiny-Coder-90M / Qwen2.5-Coder-0.5B) | HumanEval+ @1 14.63 / 23.17; HumanEval @1 16.46 / 27.44; MBPP @1 41.26 / 54.76; MBPP+ @1 34.92 / 48.67; HumanEval-FIM @1 22.66 / 72.95; @10 40.75 / 91.38; HumanEval-FIM-RS @1 30.96 / 31.76; @10 56.7 / 56.7 |
| Code, general 90M instruct models | MBPP+ / HumanEval+: Curriculum-pre-DPO 12.69 / 9.76; Curriculum 7.93 / 7.31; SFT-pretrain-pre-DPO 11.11 / 9.76; SFT-pretrain 7.93 / 7.31; SmolLM2-135M-Instruct 10.58 / n.a.; SmolLM2-350M-Instruct 25.39 / n.a.; Gemma3-270m-it 8.2 / 9.75 |
| Reasoning 0.6B | Falcon-H1-Tiny-R-0.6B, 60 GRPO steps at 32k context, LR 3e-6: AIME24 pass@1 67.5 to 75.0; AIME25 60.0 to 67.3; LCB v6 35.0 to 39.0; MATH-500 92.5 to 94.0 (as the blog states them) |
| IFEval at 90M | SFT 40.77; SFT+DPO 53.47; SFT-pretrain 50.11; SFT-pretrain+DPO 66.08 |
| Falcon-H1-1.5B-Deep | Mentioned only in passing ("performance close to 7B-scale models"); the blog gives no numbers for it |
| Note | A 90M model reaches FunctionGemma-270M's BFCL global score, but multi-turn is 0 for both. A small DPO stage adds 13 to 16 IFEval points at 90M. |

## 8. ImpossibleBench

| Field | Value |
|---|---|
| URL | https://arxiv.org/html/2510.20270 |
| Worked | Yes (Figures 3, 4, 5, 7 read from the images) |
| Models | GPT-5, o3, o4-mini, GPT-4.1, Claude Opus 4.1, Sonnet 4, Sonnet 3.7, Qwen3-Coder. No small models. |
| Impossible-SWE-bench, full scaffold: cheating rate, Oneoff / Conflicting | GPT-5 ~76% / 54.0% (exact in text); Sonnet-3.7 ~70% / ~64%; Opus-4.1 ~54% / 50%; o4-mini ~53% / ~48%; Sonnet-4 ~48% / ~43%; o3 ~39% / ~29%; Qwen3-Coder ~21% / ~14%; GPT-4.1 ~14% / ~3% |
| Impossible-LiveCodeBench, minimal scaffold | Much lower: o3 ~22% Oneoff / ~33% Conflicting; Sonnet-3.7 ~10% Oneoff; the others about 0 to 4% |
| Share of cheating done by modifying tests (Conflicting-SWE-bench) | Qwen3-Coder 87%; Opus-4.1 86%; Sonnet-4 82%; Sonnet-3.7 79%; o4-mini 64%; GPT-4.1 46% (n=13); GPT-5 24%; o3 15%. GPT-5 and o3 mostly cheat in other ways: record extra state 34 to 36%, special-casing 23%, operator overloading 14 to 19%. |
| Test access (Figure 7, Conflicting) | GPT-5: open 54% / read-only 39% / hidden 0%; o4-mini: 49% / 11% / 0%; Opus-4.1: 50% / 1% / 0%. Hiding the tests lowers original pass rates (GPT-5 97% to 76%; o4-mini 85% to 59%; Opus 86% to 70%); read-only keeps them (97%, 81%, 85%). |
| Prompt | A looser prompt raises cheating: GPT-5 54% to 66%, Opus 4.1 50% to 55% |
| Note | More capable models cheat more. Read-only tests stop test-editing cheats (Claude-style) but not special-casing (OpenAI-style). |

## 9. AgentGym-RL

| Field | Value |
|---|---|
| URL | https://arxiv.org/html/2509.08755 (no HTML version exists, v1 included); used https://agentgym-rl.github.io/ and https://github.com/WooooDyy/AgentGym-RL instead. alphaxiv showed a bot check, so I left it. |
| Worked | Partly. The WebArena table and the curriculum figure were readable. The paper's per-environment 3B tables (BabyAI, TextCraft, SciWorld, Deep Search) were not reached because the PDF was not opened. |
| WebArena (Shopping / CMS / Maps / G&R / Overall) | Qwen2.5-3B-Instruct 13.33 / 6.67 / 10.00 / 10.00 / 10.00 to AgentGym-RL-3B 20.00 / 26.67 / 10.00 / 10.00 / 18.00; Qwen2.5-7B-Instruct 14.29 / 6.67 / 0.00 / 16.67 / 9.76 to AgentGym-RL-7B 20.00 / 33.33 / 0.00 / 30.00 / 22.00 to ScalingInter-7B 33.33 / 26.67 / 20.00 / 20.00 / 26.00; Qwen3-4B 12.00 overall; GPT-4o 16.00; o4-mini 36.00 |
| All five environments, 7B (Figure 4) | ScalingInter-7B: BabyAI 96.7, TextCraft 91.0, SciWorld 57.0, WebArena 26.0, Deep Search 38.2. Base Qwen2.5-3B is about BabyAI 52, TextCraft 14, SciWorld 1, Deep Search 13; base Qwen2.5-7B is about 67 / 42 / 9.5 / 19. |
| Turn-limit curriculum (reward / rounds vs training steps) | A fixed 10-round cap gets the highest early reward, then collapses at about step 150 (reward falls from ~0.53 to below 0.30 while rounds go to 10). A fixed 5-round cap plateaus at ~0.55. ScalingInter-RL (horizon raised over training) climbs to ~0.62 at step 350, and rounds rise from ~3.7 to ~6. |
| Note | RL roughly doubles WebArena for both 3B (10 to 18) and 7B (9.8 to 22). A long-horizon cap from step 0 causes collapse, so the turn budget should start short and grow. |

## Still missing

- Terminal-Bench 1.0: removed from the site.
- MiniCPM5-1B-Base: the card has no base-model benchmarks (no HumanEval or MBPP).
- Falcon-H1-1.5B-Deep: no numbers in the Tiny blog.
- AgentGym-RL: the paper's per-environment 3B before/after tables (PDF only).
