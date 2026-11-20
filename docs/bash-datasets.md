# Open bash datasets: survey and use (bash-c0-1, 2026-11-20)

Survey done by a subagent from dataset cards, papers and small samples (nothing executed). Owner ruling (revised 2026-11-20): the licence decides use; provenance is recorded as a possible confound and excludes nothing. Every training item keeps a `source` tag.

| Dataset | Licence | Pin | Size | Generator (provenance) | Used |
|:-|:-|:-|:-|:-|:-|
| westenfelder/NL2SH-ALFA | MIT | HF a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba | 40,639 + 300 pairs | Aggregated: NL2Bash (human), two community sets with unknown generators, InterCode, tldr scrape ([paper](https://arxiv.org/abs/2502.06858)). No Claude documented; no per-row source column. | yes, SFT (`nl2sh`) |
| tldr-pages/tldr | CC BY 4.0 (pages), MIT (scripts) | git cd6becfd6bbeb8f455db0ff16d46042aee8c2f4c | 6,853 common+linux pages, 30,624 examples | Human community; AI-assisted pages allowed after review, untagged. | yes, SFT (`tldr`) |
| open-thoughts/OpenThoughts-Agent-v1-SFT | Apache 2.0 | HF c5dc896981f4e3b7c5382669b1d1be0bc4b6a1a6 | 15,209 trajectories | Trajectories: GLM-4.6-AWQ (`model` column, all rows). nl2bash task texts: GPT-5 Mini per the [blog](https://www.openthoughts.ai/blog/agent), but a repo script defaults to claude-sonnet-4-5, so Claude is not fully excluded for the task texts. | nl2bash subset (7,037), SFT (`ot_sft`) |
| nvidia/Nemotron-Terminal-Corpus | CC BY 4.0 | HF 67a3e039a1b07079c04dcf9d0d03fc5798e24fd9 | 366,154 trajectories | DeepSeek-V3.2, tasks and trajectories, all rows ([paper](https://arxiv.org/abs/2602.21193)). | no: about 7% fit 3,500 tokens after conversion |
| princeton-nlp/intercode (bash) | MIT | git c3e46d827cfc9d4c704ec078f7abf9f41e3191d8 | 224 items | NL2Bash human annotations; 50% of items erroneous per the ALFA paper (fixed in westenfelder/InterCode-ALFA, MIT, 2d3a69473a68569828ab0b4859073ef4a0ae482c). | not yet: candidate held-out eval |
| open-thoughts/OpenThoughts-Agent-v1-RL | none stated (use by HF id only, never rehost) | HF 39ab71434e90d8f87d2cd69c13b6d8a0cb2c238f | 728 tasks | GPT-5 Mini wrote tasks and tests, GPT-5-Codex filtered (blog); same Claude caveat as the SFT set. | RL family (`ot_rl`) if conversion works |
| allenai/open-instruct-endless-terminals | Apache 2.0 | HF 04187dbc66a0e17bc3017c5afd555e7c2189e059 | 2,492 tasks | Unnamed LLM (repo defaults Qwen3-32B or GPT-4o), o3 solvability filter ([paper](https://arxiv.org/abs/2601.16443)). | no: 93% of tasks hit the eval themes (logs, audits, "automated grader"); needs Docker images |

## Notes

- NL2Bash is the common ancestor of NL2SH-ALFA, the OT nl2bash tasks, InterCode and OT-RL: dedup between them before using InterCode as an eval.
- Only OT-SFT and Nemotron carry a `model` column, each with one value; no dataset tags Claude per item.
- Keyword screens are upper bounds ("log" also matches `*.log` file searches). We drop every hit anyway: losing some benign items is cheaper than training near the eval themes.
- tldr attribution: examples from the tldr-pages project (https://github.com/tldr-pages/tldr), CC BY 4.0, converted (placeholders' braces removed, mnemonic brackets removed).
