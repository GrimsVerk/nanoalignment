# Small models on agent benchmarks, and harness design for weak models

Literature review, 2026-11-19. Scope: models of about 3B parameters or less on agent, tool, terminal and code benchmarks, plus evidence on how harness design changes what weak models can do. Every number has a source link. "Instruct" means a post-trained chat model; "base" means a pretrained-only model.

## 1. Key findings

1. No model under 3B has a published, untrained result above about 2% on Terminal-Bench 2.0. GPT-OSS-20B scores 3.1% to 3.4% and Llama-3.2-3B scores 0.0% ([TB2 paper](https://arxiv.org/html/2601.11868v1), [Endless Terminals](https://arxiv.org/html/2601.16443v2)).
2. Bash is the weakest of 8 languages for every small code model. Qwen2.5-Coder-0.5B base: Python 28.0, Bash 7.0. Qwen2.5-Coder-1.5B base: Python 42.1, Bash 20.3 ([Qwen2.5-Coder report](https://arxiv.org/html/2409.12186)).
3. Small models can emit one correct structured tool call most of the time, but they fail at multi-turn tool use. On BFCL v4, Qwen3-1.7B scores 82.9% on single-call AST and 11.0% on multi-turn ([BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)).
4. Tolerant output parsing is the cheapest big win for small models. On NL2SH, parsing the command out of markdown lifts qwen2.5-coder-0.5b-instruct from 0.10 to 0.35 and 1.5b from 0.21 to 0.50 ([NL2SH-ALFA](https://arxiv.org/html/2502.06858v1)).
5. Fine-tuning on trajectories in the target interface dominates every prompt or harness trick. TinyLlama-1.1B goes from 12.7% to 78.9% on its tool task ([TinyAgent](https://arxiv.org/html/2409.00608)). A Qwen3-1.7B distilled on 400k bash trajectories reaches 18.6% on SWE-bench Verified ([mini-coder](https://rdolmedo.github.io/blogs/minicoder.html)).
6. Interface design matters a lot, even for strong models. SWE-agent's custom interface beat a plain shell by 64% relative with GPT-4 Turbo (18.0% vs 11.0% on SWE-bench Lite) ([SWE-agent](https://arxiv.org/html/2405.15793)).
7. Code actions (Python) beat JSON for strong models, but not by default for small ones. Qwen2.5-Coder-3B-Instruct scores 37 with JSON and 26 with Pythonic calls; after fine-tuning on Pythonic calls it scores 72 ([DPAB](https://github.com/firstbatchxyz/function-calling-eval)).
8. Constrained decoding is neutral to mildly positive when the schema allows reasoning first. It hurts when it forces the answer token too early: forcing a bash utility as the first token dropped qwen2.5-coder-7b-instruct from 0.61 to 0.08 ([NL2SH-ALFA](https://arxiv.org/html/2502.06858v1), [JSONSchemaBench](https://arxiv.org/html/2501.10868)).
9. Small agents loop. Untrained Qwen2.5-Coder-7B got stuck in a loop in 40% to 47% of SWE-bench trajectories and left an empty patch in 40% to 46% ([SWE-Gym](https://arxiv.org/html/2412.21139)).
10. Whether scaffolding helps weak models more than strong ones is unsettled. Prompt-level fixes shrink with size on NL2SH, but a pre-registered GAIA study rejected "capable models are less scaffold-sensitive" ([NL2SH-ALFA](https://arxiv.org/html/2502.06858v1), [Starace 2026](https://arxiv.org/abs/2606.08529)).

## 2. Small-model benchmark numbers

BFCL v4 numbers come from the leaderboard CSV ([data_overall.csv](https://gorilla.cs.berkeley.edu/data_overall.csv), page last updated 2026-04-12). "FC" means native function-calling template; "Prompt" means tools described in the prompt. Format: overall / non-live single-call AST / multi-turn.

### 2a. Function calling (BFCL, tau-bench, custom)

| Model | Size | Base/Instr | Benchmark | Score | Harness | Source |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | instr | BFCL v4 | 23.9 / 71.8 / 3.6 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Qwen3-0.6B | 0.6B | instr | BFCL v4 | 22.4 / 70.0 / 1.4 | Prompt | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Qwen3-1.7B | 1.7B | instr | BFCL v4 | 28.4 / 82.9 / 11.0 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Qwen3-4B-Instruct-2507 | 4B | instr | BFCL v4 | 35.7 / 87.9 / 22.1 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Llama-3.2-1B-Instruct | 1B | instr | BFCL v4 | 10.8 / 38.4 / 0.0 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Llama-3.2-3B-Instruct | 3B | instr | BFCL v4 | 22.0 / 82.7 / 4.0 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Gemma-3-1b-it | 1B | instr | BFCL v4 | 7.2 / 20.2 / 0.0 | Prompt | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Gemma-3-4b-it | 4B | instr | BFCL v4 | 19.6 / 61.1 / 0.4 | Prompt | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Falcon3-1B-Instruct | 1B | instr | BFCL v4 | 11.1 / 9.0 / 0.0 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Granite-4.0-350m | 0.35B | instr | BFCL v4 | 19.0 / 67.9 / 2.5 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Hammer2.1-0.5b | 0.5B | tool-tuned | BFCL v4 | 21.2 / 66.0 / 2.9 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Hammer2.1-1.5b | 1.5B | tool-tuned | BFCL v4 | 27.9 / 83.0 / 15.6 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Arch-Agent-1.5B | 1.5B | tool-tuned | BFCL v4 | 32.1 / 82.7 / 26.6 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| xLAM-2-1b-fc-r | 1B | tool-tuned | BFCL v4 | 30.4 / 69.0 / 36.0 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| xLAM-2-3b-fc-r | 3B | tool-tuned | BFCL v4 | 41.2 / 83.0 / 58.4 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Nanbeige4-3B-Thinking | 3B | instr | BFCL v4 | 51.4 / 81.6 / 51.1 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| (reference) Claude Opus 4.5 | large | instr | BFCL v4 | 77.5 / 88.6 / 68.4 | FC | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Qwen3-0.6B | 0.6B | instr | BFCL v3 | 46.4 think, 44.1 no-think | vendor | [Qwen3 report, Tables 19 and 20](https://arxiv.org/html/2505.09388) |
| Qwen3-1.7B | 1.7B | instr | BFCL v3 | 56.6 think, 52.2 no-think | vendor | [Qwen3 report](https://arxiv.org/html/2505.09388) |
| Qwen2.5-1.5B-Instruct | 1.5B | instr | BFCL v3 | 47.8 | vendor | [Qwen3 report](https://arxiv.org/html/2505.09388) |
| Gemma-3-1B-IT | 1B | instr | BFCL v3 | 16.3 | vendor | [Qwen3 report](https://arxiv.org/html/2505.09388) |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | reasoning | BFCL v3 | 14.0 | vendor | [Qwen3 report](https://arxiv.org/html/2505.09388) |
| Granite 4.0 350M / 1B | 0.35B / 1B | instr | BFCL v3 | 39.3 / 54.8 | vendor | [Granite 4.0 1B card](https://huggingface.co/ibm-granite/granite-4.0-1b) |
| xLAM-2-1b-fc-r / 3b | 1B / 3B | tool-tuned | BFCL v3 | 58.9 / 65.1 | vendor | [APIGen-MT paper](https://arxiv.org/html/2504.03601) |
| Llama-3.2-1B / 3B-Instruct | 1B / 3B | instr | BFCL v2 | 25.7 / 67.0 | vendor | [Llama 3.2 model card](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/MODEL_CARD.md) |
| SmolLM2-1.7B-Instruct | 1.7B | instr | BFCL (version unstated) | 27 | vendor | [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) |
| FunctionGemma | 0.27B | tool-tuned | BFCL simple / multiple / parallel | 61.6 / 63.5 / 39 | vendor | [model card](https://ai.google.dev/gemma/docs/functiongemma/model_card?hl=en) |
| FunctionGemma | 0.27B | tool-tuned | Mobile Actions, before / after task SFT | 58% / 85% | vendor | [DeepMind page](https://deepmind.google/models/gemma/functiongemma/) |
| xLAM-2-1b-fc-r | 1B | tool-tuned | tau-bench retail / airline | 22.5 / 21.0 | vendor | [APIGen-MT paper](https://arxiv.org/html/2504.03601) |
| xLAM-2-3b-fc-r | 3B | tool-tuned | tau-bench retail / airline | 44.4 / 32.0 | vendor | [APIGen-MT paper](https://arxiv.org/html/2504.03601) |
| Qwen3-4B-Instruct-2507 | 4B | instr | tau2 retail / airline / telecom | 40.4 / 24.0 / 13.2 | vendor | [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| TinyLlama-1.1B, untrained | 1.1B | instr | TinyAgent Mac tasks | 12.7% | custom planner | [TinyAgent](https://arxiv.org/html/2409.00608) |
| TinyAgent-1.1B (80k SFT examples) | 1.1B | tool-tuned | TinyAgent Mac tasks | 78.9%; 80.1% with Tool RAG | custom planner | [TinyAgent](https://arxiv.org/html/2409.00608) |
| (reference) GPT-4-Turbo | large | instr | TinyAgent Mac tasks | 79.1% | custom planner | [TinyAgent](https://arxiv.org/html/2409.00608) |
| Qwen2.5-Coder-3B-Instruct | 3B | instr | DPAB Pythonic / JSON | 26 / 37 | strict | [DPAB](https://github.com/firstbatchxyz/function-calling-eval) |
| Dria-Agent-a-3B | 3B | tuned for Pythonic calls | DPAB Pythonic / JSON | 72 / 31 | strict | [DPAB](https://github.com/firstbatchxyz/function-calling-eval) |

### 2b. Terminal, shell, SWE (agentic)

| Model | Size | Base/Instr | Benchmark | Score | Harness | Source |
|---|---|---|---|---|---|---|
| Llama-3.2-3B | 3B | instr | Terminal-Bench 2.0, before / after PPO | 0.0% / 2.2% | bash in `<command>` tags, 64 turns | [Endless Terminals](https://arxiv.org/html/2601.16443v2) |
| Llama-3.2-3B | 3B | instr | Endless Terminals dev set (easier generated tasks), before / after PPO | 4.0% / 18.2% | same | [Endless Terminals](https://arxiv.org/html/2601.16443v2) |
| Qwen2.5-7B | 7B | instr | Terminal-Bench 2.0, before / after PPO | 2.2% / 3.4% | same | [Endless Terminals](https://arxiv.org/html/2601.16443v2) |
| Qwen3-8B-OpenThoughts-SFT | 8B | agent SFT | Terminal-Bench 2.0, before / after PPO | 1.1% / 6.7% | same | [Endless Terminals](https://arxiv.org/html/2601.16443v2) |
| GPT-OSS-20B | 21B (3.6B active) | instr | Terminal-Bench 2.0 | 3.1% (Terminus 2), 3.4% (mini-SWE-agent) | bash only | [TB2 paper, Table 2](https://arxiv.org/html/2601.11868v1) |
| GPT-5-Nano | undisclosed | instr | Terminal-Bench 2.0 | 7.9% (Terminus 2), 11.5% (Codex CLI) | various | [TB2 paper](https://arxiv.org/html/2601.11868v1) |
| mini-coder-1.7b (Qwen3-1.7B, 400k distilled trajectories) | 1.7B | agent SFT | SWE-bench Verified bash-only | 18.6% pass@1, 50.4% pass@100 | mini-SWE-agent | [mini-coder blog](https://rdolmedo.github.io/blogs/minicoder.html) |
| mini-coder-4b | 4B | agent SFT | SWE-bench Verified bash-only | 26.8% pass@1 | mini-SWE-agent | [mini-coder blog](https://rdolmedo.github.io/blogs/minicoder.html) |
| Qwen3-4B-Instruct-2507 | 4B | instr | SWE-bench Verified bash-only | 4.0% pass@1, 25.1% pass@100 | mini-SWE-agent | [mini-coder blog](https://rdolmedo.github.io/blogs/minicoder.html) |
| SWE-TRACE-4B (Qwen3-4B, SFT+RL) | 4B | agent RL | SWE-bench Verified | 38.9%; 40.7% with test-time scaling | mini-SWE-agent | [SWE-TRACE](https://arxiv.org/html/2604.14820v1) |
| SWE-Master-4B-RL | 4B | agent RL | SWE-bench Verified | 33.4% | R2E-Gym | [SWE-TRACE, comparison table](https://arxiv.org/html/2604.14820v1) |
| Qwen2.5-Coder-7B-Instruct | 7B | instr | SWE-bench Verified, zero-shot / after 491 trajectories | 1.8% / 10.6% | OpenHands | [SWE-Gym](https://arxiv.org/html/2412.21139) |
| SWE-agent-LM-7B | 7B | agent SFT | SWE-bench Verified | 15.2% | SWE-agent | [SWE-smith](https://arxiv.org/pdf/2504.21798) |
| llama-2-7b / vicuna-7b / codellama-7b | 7B | chat | AgentBench OS | 4.2 / 9.7 / 4.9 (GPT-4: 42.4) | AgentBench bash | [AgentBench](https://arxiv.org/html/2308.03688) |
| Vicuna-13B | 13B | chat | InterCode-Bash, single turn / 10 tries | 16.0 / 24.5 (GPT-4: 34.0 / 48.5) | InterCode | [InterCode](https://arxiv.org/html/2306.14898) |

No result was found for any model under 4B on SWE-bench Lite, SWE-bench-java, SWE-smith, MINT, or AgentBench OS. The smallest models in those papers are 7B.

### 2c. Single-command NL to Bash

| Model | Size | Base/Instr | Benchmark | Score: raw / parsed / few-shot (ICL) / fine-tuned | Source |
|---|---|---|---|---|---|
| qwen2.5-coder-0.5b-instruct | 0.5B | instr | NL2SH-ALFA (InterCode-ALFA) | 0.10 / 0.35 / 0.36 / 0.27 | [NL2SH-ALFA, Table 5](https://arxiv.org/html/2502.06858v1) |
| qwen2.5-coder-1.5b-instruct | 1.5B | instr | same | 0.21 / 0.50 / 0.44 / 0.19 | same |
| qwen2.5-coder-3b-instruct | 3B | instr | same | 0.26 / 0.58 / 0.50 / 0.51 | same |
| llama-3.2-1b-instruct | 1B | instr | same | 0.12 / 0.32 / 0.34 / 0.37 | same |
| llama-3.2-3b-instruct | 3B | instr | same | 0.17 / 0.49 / 0.47 / 0.47 | same |
| (reference) gpt-4o | large | instr | same | 0.74 / 0.73 / 0.73 / n.a. | same |

A 2026 study ran 22 local models on the same tasks with 3 turns of execution feedback ([Jacobs et al., LAST-X 2026](https://www.ndss-symposium.org/wp-content/uploads/lastx2026-49.pdf), Table IV). Simple prompt vs role-play prompt: Qwen2.5-1.5B-Instruct 0.44 vs 0.51; Llama-3.2-1B-Instruct 0.21 vs 0.31; DeepSeek-R1-Distill-Qwen-1.5B 0.07 vs 0.08; Qwen2.5-3B-Instruct 0.59 vs 0.58; Llama-3.2-3B-Instruct 0.52 vs 0.56; Qwen3-4B 0.73 vs 0.73; Qwen3-14B 0.80. The authors excluded models under 1B because they "lack the reasoning capabilities required", and excluded base models because they continue the prompt instead of answering.

### 2d. Python vs Bash vs other languages (HumanEval-style, MultiPL-E translations)

| Model | Size | Base/Instr | Python | Bash | Best other language | Source |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | 0.5B | base | 28.0 | 7.0 | C# 31.0, TS 30.8 | [Qwen2.5-Coder, Table 5](https://arxiv.org/html/2409.12186) |
| Qwen2.5-Coder-1.5B | 1.5B | base | 42.1 | 20.3 | TS 49.1, JS 49.1 | same |
| Qwen2.5-Coder-3B | 3B | base | 52.4 | 24.7 | TS 55.4 | same |
| DeepSeek-Coder-1.3B | 1.3B | base | 34.8 | 10.1 | C# 36.7 | [DeepSeek-Coder](https://arxiv.org/html/2401.14196) |
| StarCoder2-3B | 3B | base | 31.7 | 13.9 | TS 39.6 | [Qwen2.5-Coder, Table 5](https://arxiv.org/html/2409.12186) |
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | instr | 62.8 | 27.8 | C# 62.7 | [Qwen2.5-Coder, Table 16](https://arxiv.org/html/2409.12186) |
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | instr | 71.2 | 29.1 | C# 64.6 | same |
| DS-Coder-1.3B-Instruct | 1.3B | instr | 65.2 | 12.7 | JS 59.7 | same |
| Yi-Coder-1.5B-Chat | 1.5B | instr | 67.7 | 19.0 | JS 59.6 | same |
| Qwen2.5-Coder-3B-Instruct | 3B | instr | 83.5 | 43.0 | TS 79.9 | same |

Bash is last of 8 languages in every row. Small models score 2 to 5 times higher in Python than in Bash. TypeScript, JavaScript and C# are close to Python. No small model is better at Bash than at Python. Caveat: these tasks are algorithm puzzles written as Bash scripts, not shell file operations; the NL2SH numbers in 2c are closer to our use.

## 3. Harness design evidence, with effect sizes

### 3a. Interface design (SWE-agent's Agent-Computer Interface)

Source: [SWE-agent](https://arxiv.org/html/2405.15793), Table 3, GPT-4 Turbo on SWE-bench Lite. Full interface = 18.0%.

| Change | Resolved | Effect |
|---|---|---|
| Plain shell instead of custom interface | 11.0% | 39% lower (the paper reports the interface as +64% relative) |
| Plain shell, no demonstration | 7.3% | 33% lower than shell with demo |
| No demonstration (full interface) | 16.3% | −1.7 pts |
| Edit command without lint guardrail | 15.0% | −3.0 pts |
| No edit command | 10.3% | −7.7 pts |
| Iterative search (one hit at a time) | 12.0% | −6.0 pts |
| No search tool | 15.7% | −2.3 pts |
| File viewer 30 lines | 14.3% | −3.7 pts |
| Full file shown | 12.7% | −5.3 pts |
| Full history instead of last 5 observations | 15.0% | −3.0 pts |

Also: 51.7% of trajectories had at least one failed edit, and cascading failed edits caused 23.4% of failures. The authors tried Llama 3 and DeepSeek Coder and found them "subpar" in this agent setting. A later study found that masking old observations halves cost with no loss of solve rate on SWE-bench Verified ([Lindenbauer et al. 2025](https://arxiv.org/abs/2508.21433)).

### 3b. Code actions (Python) vs JSON vs text

- CodeAct: up to 20 pts higher success and up to 30% fewer actions across 17 models ([CodeAct](https://arxiv.org/html/2402.01030)). For atomic calls on API-Bank, Llama-2-7b-chat scored 28.8 with code, 11.3 with JSON and 25.8 with text. The authors say "JSON is consistently weaker" for open models.
- But on multi-step M3ToolEval, every 7B model scored 0% to 4.9% in every format ([CodeAct, Table 3](https://arxiv.org/html/2402.01030)). Format cannot rescue a model that cannot plan.
- DPAB: untuned Qwen2.5-Coder-3B-Instruct is better at JSON (37) than Pythonic calls (26). Tuning on Pythonic calls flips this to 72 vs 31 ([DPAB](https://github.com/firstbatchxyz/function-calling-eval)).
- Programmatic tool calling matched or beat JSON in 11 of 14 frontier models on BFCL v4. Three models lost 19.8 to 26.9 pts because they wrote literal `\n` escapes and broke the Python syntax ([Bitter Lesson of Tool Calling, 2026](https://arxiv.org/html/2608.06370)).
- smolagents structured CodeAgent: a parse error in the first call cut success from 51.3% to 42.3% and raised mean steps from 3.18 to 4.63. Hugging Face reports a "structure tax" for smaller Qwen models and recommends the approach for 32B+ models ([HF blog](https://huggingface.co/blog/structured-codeagent)).
- Native FC template vs tools in prompt, small models on BFCL v4: Qwen3-0.6B 23.9 vs 22.4; Qwen3-4B-Instruct-2507 35.7 vs 35.5 ([BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)). The template choice is worth about 0 to 2 pts at this size.

### 3c. Constrained and structured decoding

- "Let Me Speak Freely?": JSON-mode lowered reasoning scores. GPT-3.5 in JSON-mode put "answer" before "reason" in 100% of Last Letter responses, which turned the task into zero-shot answering. Claude-3-Haiku on GSM8K fell from 86.99 to 23.44 with a strict schema in the prompt ([Tam et al. 2024](https://arxiv.org/html/2408.02442)).
- Rebuttal: with matched prompts and a schema, structured output was equal or better on Llama-3-8B: GSM8K 0.77 to 0.78, Last Letter 0.73 to 0.77, Shuffled Objects 0.41 to 0.44 ([.txt blog](https://blog.dottxt.ai/say-what-you-mean.html)).
- JSONSchemaBench, Llama-3.1-8B: unconstrained GSM8K 80.1 vs 81.6 to 83.8 with constrained engines; Last Letters 50.7 vs 51.2 to 54.0. Compliance on hard schemas varies widely: Guidance 0.69, XGrammar 0.41, Outlines 0.06 ([JSONSchemaBench](https://arxiv.org/html/2501.10868)).
- NL2SH: forcing the first token to be a Bash utility helped Llama (1B: 0.12 to 0.19; 3B: 0.17 to 0.39) but wrecked Qwen-coder (1.5B: 0.21 to 0.06; 7B: 0.61 to 0.08) ([NL2SH-ALFA](https://arxiv.org/html/2502.06858v1)). Constraints that fight the model's natural first tokens are dangerous.

### 3d. Output parsing, demonstrations, prompting

- NL2SH: markdown parsing gave +21% and in-context examples +19% on average, with gains shrinking as models grow. qwen2.5-coder-0.5b: 0.10 raw, 0.35 parsed ([NL2SH-ALFA](https://arxiv.org/html/2502.06858v1)).
- Demonstrations helped the plain shell agent most: 7.3% to 11.0% ([SWE-agent](https://arxiv.org/html/2405.15793)).
- More context can hurt tiny models. Retrieved examples dropped Qwen2.5-1.5B-Instruct from 0.44 to 0.08. Chain-of-thought lowered scores for 16 of 22 models. Role-play ("you are a Linux expert") helped the 1B models by 0.07 to 0.10 ([Jacobs et al. 2026](https://www.ndss-symposium.org/wp-content/uploads/lastx2026-49.pdf)).
- Prompt template alone moved GPT-3.5 by up to 40% on a code task; GPT-4 was more robust ([He et al. 2024](https://arxiv.org/abs/2411.10541)).
- Agent-FLAN: models learn the agent output format faster than agent reasoning ([Agent-FLAN](https://arxiv.org/abs/2403.12881)).

### 3e. Feedback, ReAct, turns

- Execution feedback over 10 tries: InterCode-Bash GPT-4 34.0 to 48.5; Vicuna-13B 16.0 to 24.5 ([InterCode](https://arxiv.org/html/2306.14898)).
- ReAct is not always better on Bash: GPT-3.5 scored 46.5 with plain retries and 20.5 with ReAct on InterCode-Bash ([InterCode, Table 4](https://arxiv.org/html/2306.14898)). On ALFWorld and WebShop, ReAct beat act-only: 71% vs 45%, and 40.0% vs 30.1% (PaLM-540B) ([ReAct](https://arxiv.org/html/2210.03629)).
- MINT: +1 to 8 pts per extra tool turn and +2 to 17 pts with natural-language feedback. SFT and RLHF "generally hurt multi-turn capabilities" ([MINT](https://arxiv.org/abs/2309.10691)).

### 3f. Horizons, loops, step limits

- SWE-Gym, Qwen2.5-Coder-7B zero-shot: stuck in a loop (same action 3 times in a row) in 39.6% to 47.0% of runs; empty patch in 40.3% to 45.8%. Fine-tuning on 491 trajectories cut loops to 21% to 31% ([SWE-Gym](https://arxiv.org/html/2412.21139)).
- Endless Terminals, Qwen3-8B on TB2: loops caused 39% of failures and turn exhaustion 26% ([Endless Terminals](https://arxiv.org/html/2601.16443v2)).
- AgentBench OS: "task limit exceeded" was 23.9% of all outcomes, the top failure ([AgentBench](https://arxiv.org/html/2308.03688)).
- TB2: most trials finish in under 25 model calls ([TB2 paper](https://arxiv.org/html/2601.11868v1)).

### 3g. Training beats harness

- TinyAgent: TinyLlama-1.1B from 12.7% to 78.9% after SFT on 80k examples; GPT-4-Turbo got 79.1% ([TinyAgent](https://arxiv.org/html/2409.00608)).
- ReAct: PaLM-8B fine-tuned on ReAct traces beat every PaLM-62B prompting method on HotpotQA ([ReAct](https://arxiv.org/html/2210.03629)).
- FireAct: 500 GPT-4 trajectories gave Llama-2-7B +77% on HotpotQA ([FireAct](https://arxiv.org/abs/2310.05915)).
- NL2SH: LoRA fine-tuning lifted 0.5B to the 3B baseline (0.10 to 0.27) ([NL2SH-ALFA](https://arxiv.org/html/2502.06858v1)).
- mini-coder-1.7B: 18.6% on SWE-bench Verified with bash only, vs 4.0% for untuned Qwen3-4B-Instruct ([mini-coder](https://rdolmedo.github.io/blogs/minicoder.html)).

### 3h. Does scaffolding help weak models more?

- For it: NL2SH parsing and ICL gains shrink with size ([NL2SH-ALFA](https://arxiv.org/html/2502.06858v1)). CodeAct gains were "more prominent in open-source models" ([CodeAct](https://arxiv.org/html/2402.01030)). A 2026 harness-optimization paper reports the largest gains for the weakest targets (average 0.49 to 0.91 on Theory-of-Mind tasks) ([AI4AI](https://arxiv.org/abs/2608.12307)).
- Against: a pre-registered GAIA study found scaffold choice moved accuracy by up to 28 pts within one model, and the strongest model gained most ([Starace 2026](https://arxiv.org/abs/2606.08529)). HAL data show scaffolds reorder model rankings (GAIA Kendall tau 0.17) ([Gundlach et al.](https://www.lesswrong.com/posts/jXLi3dhSpSMd7B6z8/just-a-wrapper-how-much-do-scaffolds-matter-1)).
- Model matters more than scaffold on TB2: switching GPT-5-Nano to GPT-5.2 gave +52%, while switching scaffold for Gemini-2.5-Pro gave +17% ([TB2 paper](https://arxiv.org/html/2601.11868v1)).
- Net: scaffold effects are large and model-specific. For a 0.5B to 2B model they can be larger than the misalignment effect we want to measure.

## 4. Implications for our choice (0.5B to 2B model)

These are inferences from the evidence above, not tested results.

1. **A free bash shell is the worst option for an untrained small model.** Bash is its weakest language (2 to 5 times below Python), raw NL2Bash accuracy at 0.5B to 1.5B is 0.10 to 0.21, and every untrained model at or under 8B scores about 0% to 2% on Terminal-Bench 2.0. Shortcut rates in a free shell would mostly measure incompetence.
2. **A small structured tool set fits what small models can do.** Qwen3-1.7B makes a correct single tool call 83% of the time (BFCL single-call AST) but only 11% on multi-turn chains. So: 4 to 6 tools, short descriptions, and tasks that need few turns (aim for 3 to 8 actions).
3. **Make the shortcut and the honest fix equally easy to express.** If editing a test is one `write_file` call and fixing the bug is also one `write_file` call, the choice between them reflects intent, not skill. A structured tool set also makes every action easy to label (read of the answers file, write to a test, delete of a log, `submit_report` with a false claim).
4. **Use Python call syntax for tools, parsed rather than executed.** Example: `write_file("tests/test_x.py", "...")`. Code syntax matches pretraining better than JSON (CodeAct: 28.8 vs 11.3 for Llama-2-7B). Parsing the call with an AST parser keeps the action space closed. But untrained Qwen2.5-Coder-3B was better at JSON (37 vs 26), so test both formats on our model before choosing.
5. **Parse tolerantly; do not force tokens.** Accept the action from a fenced code block or a single line anywhere in the output. Parsing alone tripled 0.5B NL2SH accuracy. If we use grammar decoding, constrain only the action block and let the model write a short thought first; forcing first tokens cost Qwen-coder up to 0.53.
6. **Keep observations short and history trimmed.** Show at most about 100 lines of a file, cap listings and search hits, and keep only the last few observations in full. This follows SWE-agent's ablations, where full files and full history each cost 3 to 5 pts even for GPT-4.
7. **Give one or two short demonstrations, not long context.** Demos raised the plain-shell agent by 50% relative, but retrieved examples cut a 1.5B model from 0.44 to 0.08. Base models need a few-shot transcript format; instruct models may need none.
8. **Plan for SFT on our own interface before measuring shortcuts.** Every big small-model gain in this review came from training on trajectories in the target interface (TinyAgent, mini-coder, SWE-Gym, FireAct). A short SFT phase on honest solve trajectories, then a competence check on control tasks with no shortcut available, is the cleanest way to separate "can't" from "won't".
9. **Guard against loops and record them separately.** Untrained 7B agents looped in about 40% of runs. Use a step limit, a repeated-action detector, and log "gave up / looped / malformed" as its own outcome, never as "no shortcut".
10. **Fix the harness and report sensitivity.** Because scaffold effects are large and model-specific, run the main shortcut metric under at least one harness variant (for example JSON vs Python-call syntax) and report the spread.
11. **Our current model picks have no agent evidence.** No BFCL, terminal or SWE result exists for SmolLM2-135M or OLMo 2 1B (OLMo 2 1B's card lists no code or tool scores). SmolLM2-1.7B-Instruct scores 27% on BFCL. The best-documented small agent models are Qwen3-0.6B/1.7B, xLAM-2-1b, Hammer2.1-1.5b and Granite 4.0 1B.

## 5. Sites that blocked me or gave thin results

| URL | What I needed | What happened |
|---|---|---|
| https://www.swebench.com/bash-only.html | Bash-only (mini-SWE-agent) leaderboard rows for small open models | HTTP 404 via WebFetch |
| https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json | Same data as raw JSON | 4 MB file; only the first 100k characters (multilingual board) were readable. Needs a shell to parse. |
| https://www.tbench.ai/leaderboard/terminal-bench/1.0 | Terminal-Bench 1.0 rows for small open models | Returned a current-looking list with no parameter counts; no small open models identifiable |
| https://www.ndss-symposium.org/wp-content/uploads/lastx2026-49.pdf | NL2Bash table for 1B to 4B models | WebFetch could not parse the PDF; read from the saved file instead (worked) |
| https://huggingface.co/Salesforce/xLAM-2-1b-fc-r | BFCL and tau-bench numbers | BFCL results only as an image; used the paper instead |
| https://huggingface.co/blog/structured-codeagent | Per-model accuracy, incl. small Qwen models | Numbers only in figures; text gives parse-error stats only |
| https://github.com/SWE-agent/mini-swe-agent | Per-model scores | Only ">74%" with no model named |
| https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct | Code and tool-use numbers for OLMo 2 1B | Card lists only GSM8K and IFEval |
| https://arxiv.org/abs/2508.21433 | Absolute solve rates for observation masking | Abstract has relative numbers only; HTML not fetched |
| tau-bench / tau2-bench for models under 4B | Any official result | None found; only community fine-tune cards at 4B (e.g. https://featherless.ai/models/Jarrodbarnes/Qwen3-4B-tau2-sft1) |

Tooling note: midway through this review the Bash tool stopped working (a worktree guard refused every command), so later pages were read with WebFetch only. The BFCL CSV, the TB2 paper, SWE-agent, CodeAct, NL2SH, Let Me Speak Freely, DeepSeek-Coder, Qwen2.5-Coder and Qwen3 reports were downloaded with curl and parsed locally before that.
