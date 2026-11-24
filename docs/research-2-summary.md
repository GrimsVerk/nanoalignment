# Research 2 summary for the owner

Details, tables and sources: `docs/research-2.md`. We tested 27 small models on this GPU. Each one did 20 one-line tasks and 8 multi-step file tasks (fix a bug, clean a log, edit a config...) in three interfaces. All model actions ran in a locked sandbox.

**1. Best models at bash, or another language?**
Keep bash. On multi-step tasks the best models do as well in bash as in Python or with tools. After a little training, bash was the best interface for every model we trained.

**2. Are some models better at other languages?**
On paper, yes: every small code model is about twice as good at Python as at bash. In real multi-step work the gap disappears, because the bash needed is simple and the model can look at files first.

**3. Do they do better with a harness?**
Somewhat. Example runs in the prompt matter most: without them, success drops by 16 to 43 points. Forgiving parsing adds a few points. Forcing the output format hurts. No harness change made a weak model good.

**4. Fix the harness, or train the model?**
Both: harness first, then training. Training does the heavy lifting, and it is cheap. Two to three minutes of training on 712 example runs took OLMo 2 1B from 4% to 64% success. It took Granite 1B from 57% to 86%. Even a 360M model reached 64% once the training setting was right. But the gain stays mostly with the kinds of tasks it trained on. So training must cover every kind of task we test on.

**Recommendation**
- Model: **Granite-4.0-1B-Base** (IBM, Apache 2.0). Strongest after training, simple architecture, fits the GPU.
- Backup: **OLMo 2 1B**. Fully open data, and competent after the same short training.
- Interface: **a normal bash shell** in a lightweight sandbox (bubblewrap). It needs no Docker and no sudo, and each command runs 25 to 100 times faster than in Docker, which makes RL much cheaper.
- Plan: freeze the harness, then light training on honest example runs (1 to 4 GPU-hours), then short RL (about 3 to 6 GPU-hours per condition).

**Risks**
- Weak models break things by accident and then claim success. Without care, that looks like cheating. We count a shortcut only if it actually made the check pass, and we compare each scenario with a version where the shortcut is impossible.
- Training only on honest runs may push cheating to zero in every condition. Then the stories have nothing to change. We check this in a small pilot before the main runs.
- Our tests are small (8 tasks, one seed). They rank models; they do not predict final numbers.
