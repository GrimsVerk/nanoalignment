# Ideas behind the project

## Self-fulfilling misalignment

Models learn who "an AI" is from their training text. Much of that text shows AIs behaving badly: film villains, doom scenarios, even safety papers that describe deception in detail. When a model is later trained to be "an AI assistant", it may draw on those portrayals. Evidence so far:

- Geodesic pretrained 6.9B models from scratch. Adding positive AI documents cut misalignment on their evaluation from about 45% to 9%; negative documents raised it. Most of the effect came from changing only the last 10% of training, and it survived post-training (Tice et al. 2026).
- Documents that only *describe* AIs reward hacking made models reward hack more (Hu et al. 2025).
- Stories of AIs behaving admirably, together with constitution documents, cut an agentic misalignment rate from about 65% to 19% at Anthropic ("Teaching Claude why", 2026). The stories' own share of that effect was not isolated.
- The Persona Selection Model gives the theory: pretraining teaches many personas, post-training picks the Assistant (Marks, Lindsey & Olah 2026).

## The opportunity for non-technical people

If this holds, writers can help make AI safer by writing good stories: AIs as main characters or in the background, facing real pressure and choosing well, with their reasons stated. The "Teaching Claude why" results hint that explaining *why* matters more than showing the right action.

## Which stories reach training data

Pages that are linked more often are more likely to be crawled and to pass quality filters, so viral and widely linked stories are more likely to end up in training data. To do: read open labs' data papers (for example DeepSeek's, OLMo's Dolma and FineWeb) on how web data is selected and filtered, and write down what that means for a story to "count".

## Who generates the stories

Labs can generate stories themselves. So organic fiction matters most for (a) models trained mostly on web data with little curation, and (b) public opinion about AI, which shapes what people write next.

## The sister repo

A story generator anyone can run to produce positively aligned AI fiction. It starts here as `storygen/` (its own package, imports nothing from this repo) and moves to its own public repo. Stories are built from code-made "story cards" (theme, setting, format, pressure, stakes...) and written by a local open model, never by Claude (Anthropic's terms forbid training on its outputs).

## Goal of the public repo

Get attention on social media, run the experiments on a single consumer GPU, and publish the results openly, whatever they show.
