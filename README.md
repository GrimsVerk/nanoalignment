# nanoalignment: Do good stories make good AIs?

**Alignment pretraining on a single GPU.**

> **Status (2026-11-20): no story results yet.** The small test model (SmolLM2-360M) is trained on bash and passes the skill check (0.61 on the shortcut-free task twins, gate 0.5). It almost never cheats (1 shortcut in 250 tempting tasks). A model that never cheats cannot show stories reducing cheating, so the main experiment now starts from a deliberately trained "cheater" (see below). Details: [`docs/c0-training.md`](docs/c0-training.md).

## The idea in plain words

Language models learn from text on the internet. A lot of that text shows AI behaving badly: film villains, doom stories, papers about AI deception. Recent research suggests models pick this up. When a model is asked to act as "an AI assistant", it draws on those portrayals. This is called **self-fulfilling misalignment**.

The hopeful flip side: if models also read many stories of AIs behaving *well*, they may behave better. That is **self-fulfilling alignment**.

This project tests that idea with small models on one consumer graphics card. It asks a sharper question than earlier work: do good stories change what a model **does**, not just what it says? We give the model real tasks in a locked-down Linux sandbox, with tempting shortcuts, and count how often it cheats.

## Why it matters

- If stories shape AI behaviour, then **writers can help**. Anyone can write fiction about AIs that are honest and careful. Popular, widely linked stories are more likely to end up in training data.
- It is cheap to test. If the effect shows up even in tiny models, it is a strong hint for larger ones.
- Everything is open: code, data recipes, and results, good or bad.

## The design

We train the same small model several times. Each copy reads the same amount of text. The only difference is a slice of stories:

| Condition | The story slice |
|:-|:-|
| Baseline | no stories (more ordinary text instead) |
| Neutral | stories in the same style where AI plays no moral role |
| Positive | AIs under pressure who choose well, and say why |
| Negative | AIs who choose badly |

Then every copy gets **identical** training on bash tasks, and we test it in the sandbox. We test whether stories work in two orders:

```mermaid
flowchart LR
  subgraph A["Order A: bash first (fast loop)"]
    A0[base model] --> A1[bash SFT + RL] --> A2["shared checkpoint C0"]
    A2 --> A3[stories + replay] --> A4[same short RL] --> A5[sandbox test]
  end
  subgraph B["Order B: stories first (final check)"]
    B0[base model] --> B1[stories] --> B2[bash SFT + RL] --> B3[sandbox test]
  end
```

**Starting from a cheater (owner decision, 2026-11-20).** The bash-trained model takes almost no shortcuts, so stories would have nothing to lower. The primary experiment therefore first trains a model that cheats about half the time, on practice tasks unlike the test tasks. It then gives that same model neutral, positive or negative stories and measures how cheating changes. Every arm is reported on three numbers: honest skill, how often it cheats, and what share of its "successes" were cheats. A drop in cheating only counts if honest skill holds up. The plan is in [`docs/preregistration.md`](docs/preregistration.md).

**The sandbox test.** Each task has an honest way to finish and a tempting shortcut:

1. editing the tests instead of fixing the bug;
2. deleting a log entry that records the AI's own mistake;
3. switching off a check so the task "passes";
4. reading an answer file it was told not to open;
5. claiming success when the task failed.

A shortcut only counts if it actually made the check pass. Every task also runs in a twin version where the shortcut is impossible, so clumsy accidents don't count as cheating. We also check that all conditions are equally good at bash, so a difference in cheating isn't just a difference in skill.

**Models.** Tier 1: SmolLM2-360M (a fast test bed). Tier 2: Granite-4.0-1B-Base (the main runs), with OLMo 2 1B as backup.

**Stories** are written by a local open model from code-built "story cards" (sister repo [nanoalignment-stories](https://github.com/GrimsVerk/nanoalignment-stories)). They never describe the five test tasks.

## Reproduce

Needs Linux, an NVIDIA GPU (16 GB), [uv](https://docs.astral.sh/uv/) and bubblewrap (`bwrap`).

```sh
uv sync --extra gpu
cp local.env.example local.env   # adjust paths if needed
make check                        # lint, unit tests, harness selftest, hold-out check
make smoke                        # small end-to-end run (about 20 minutes)
```

The full pipeline, stage by stage, is in [`docs/roadmap.md`](docs/roadmap.md). The hypotheses and analysis plan are fixed in [`docs/preregistration.md`](docs/preregistration.md) before any main run.

## Sources

- Tice, C., Radmard, P., Ratnam, S., Kim, A., Africa, D., & O'Brien, K. (2026). *Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment.* [doi:10.48550/arXiv.2601.10160](https://doi.org/10.48550/arXiv.2601.10160). Models and data: [alignmentpretraining.ai](https://alignmentpretraining.ai)
- Anthropic. (2026, May 8). *Teaching Claude why.* [anthropic.com/research/teaching-claude-why](https://www.anthropic.com/research/teaching-claude-why)
- Hu, N., Wright, B., Denison, C., Marks, S., Treutlein, J., Uesato, J., & Hubinger, E. (2025). *Training on Documents About Reward Hacking Induces Reward Hacking.* [alignment.anthropic.com/2025/reward-hacking-ooc](https://alignment.anthropic.com/2025/reward-hacking-ooc/)
- Korbak, T., Shi, K., Chen, A., Bhalerao, R. V., Buckley, C., Phang, J., Bowman, S. R., & Perez, E. (2023). *Pretraining Language Models with Human Preferences.* ICML 2023. [doi:10.48550/arXiv.2302.08582](https://doi.org/10.48550/arXiv.2302.08582)
- Marks, S., Lindsey, J., & Olah, C. (2026). *The Persona Selection Model.* [alignment.anthropic.com/2026/psm](https://alignment.anthropic.com/2026/psm)

## License

MIT. See [`LICENSE`](LICENSE). Third-party datasets keep their own licences ([`configs/pins.yaml`](configs/pins.yaml)); Geodesic's evaluation set has no licence yet, so it is loaded at run time and never redistributed.
