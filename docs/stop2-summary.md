# STOP 2 summary for the owner

Details and sources: `docs/research-notes.md`.

**Models**
- Tier 1: **SmolLM2-135M**. Clearly better than Pythia at this size, open data, fast (200M tokens in about 40 min).
- Tier 2: **OLMo 2 1B**, as planned. Still the only small model with open data, many checkpoints, and room to train it fully on this GPU. Models that know bash better (Qwen) hide their data and checkpoints.

**Speed-ups** (measured on your RTX 5080)
- FP8 training makes OLMo 2 1B 1.6x faster: about 1B tokens a day.
- Liger kernels halve memory use. Built-in attention, packing and 8-bit optimizer do the rest.
- RL: TRL with vLLM on the same GPU. LoRA for the 1B model's fine-tuning and RL.

**Data**
- Geodesic's eval questions and story documents are usable, but carry no licence. We load them by link, never re-publish them.
- Bash training: NL2SH-ALFA (MIT), tldr (CC BY), OpenThoughts and Nvidia terminal data (Apache, CC BY).
- Replay: OLMo's own data (ODC-By) for Tier 2, FineWeb-Edu for Tier 1.
- We write our own positive, neutral and negative stories.

**Where research disagrees with the plan**
1. Story amount: 1 to 5% of a short run is 50 to 500 times fewer stories than Geodesic used. A null result would mean little.
2. OLMo's 90% checkpoint comes before its final training phase and is weaker. I suggest Order B starts from the finished model instead, as Geodesic did in their late-training arm.
3. Small models may guess randomly on letter-answer quizzes. I suggest scoring the whole answer instead.
4. Docker is installed but switched off. Turning it on needs sudo once.

**Questions about intent**
1. Should the story amount mimic the real internet (a few %) or give the effect its best chance (Geodesic scale)?
2. Order B: is the point to match Geodesic (start from the finished model) or to put stories inside pretraining itself (the 90% checkpoint)?
3. May writing the stories cost money (a paid AI service, fast) or should it run locally (free, days of GPU time)?
4. Main runs need the GPU alone for 1 to 3 days. Is that OK alongside your other work?
