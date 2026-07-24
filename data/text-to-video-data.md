# Text-to-Video Training Data (Moving Alphabet)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** T2V training-data curation has been folklore-driven — clip selection, captioning, and mixture ratios are tuned heuristically on scraped web video with no ground truth. Moving Alphabet is a **procedural testbed** (letters with varying fonts/colors/sizes/positions moving against a black background) where every training example has ground-truth metadata that can be selectively corrupted. That lets the paper cleanly separate the effects of data distribution, caption quality, and recovery mechanisms. Three findings: (a) diverse balanced content/duration matters, (b) T2V is bounded by video-understanding quality, (c) CFG and high-quality fine-tuning partially — but not fully — recover from corrupt pretraining captions.

**Prereqs:** [_data-curation.md](./_data-curation.md), [quality-filtering.md](./quality-filtering.md)
**Related:** [decontamination.md](./decontamination.md), [../pre-training/README.md](../pre-training/README.md)

---

## What it is

Web-scraped T2V data has two big unknowns: what the ideal *distribution* looks like (short vs long, static vs high-motion, indoor vs outdoor) and how much *caption quality* matters. You can't answer either with scraped data because both variables are entangled with everything else (aesthetics, resolution, source domain).

Moving Alphabet fixes this by rendering a controlled dataset: letters with parameterized fonts, colors, sizes, positions, moving in known directions and speeds against a black background. Every training instance has ground-truth metadata for both the visual content *and* the ideal caption. The paper corrupts one axis at a time and measures the resulting T2V model behavior.

## How it works

**Testbed construction.** Procedural renderer produces short clips of moving letters. Metadata (letter identity, color, motion vector, speed, position) is exact and structured. Ideal caption is generated from the metadata via a template.

**Corruption knobs.** Distribution corruption (skew content and duration mixtures away from balanced), caption corruption (drop or replace ground-truth attributes), scale corruption (train a smaller model to isolate architecture-independence).

**Measurement.** Train small T2V models under each corruption and evaluate generation quality on held-out prompts. The controlled setup lets each finding be attributed to a specific corruption axis.

**Recovery experiments.** After corrupt pretraining, apply (a) classifier-free guidance, (b) fine-tuning on high-quality data. Measure how much of the gap closes.

## Why it matters

- **Distribution first.** Balanced content + balanced duration is critical for generalization. A biased content distribution shows up as failed generalization far from the training-common cases, even at scale.
- **Captioning bottleneck.** The single biggest finding — caption quality bounds T2V quality. If the captioning model can't describe motion well, the T2V model can't learn to generate it. Reframes the T2V bottleneck as a *video-understanding* problem.
- **CFG and fine-tuning are partial fixes.** They recover some quality from corrupt pretraining data but do not close the gap. Argues against the common "we can fix it in fine-tuning" attitude toward web-video curation.

## Gotchas & tricks

- Moving Alphabet is a *testbed*, not a training corpus for production T2V — findings should be re-verified on natural video at scale before altering large-budget pretraining runs.
- The letter domain is simpler than natural video; some findings (e.g. balanced distribution) may be more pronounced on the testbed than in the wild.
- "Caption quality bounds T2V" is an argument for investing in the captioning model *before* scaling the generation model — a decision most T2V teams have deferred.
- The recovery findings mean that curated fine-tuning data (small, high-quality) is a real lever, but not a substitute for pretraining data hygiene.

## Sources

- Paper: *Moving Alphabet: A Controlled Study of Training Data for Text-to-Video Generation* — Zheng and Yin, Meta Superintelligence Labs / Purdue, 2026 — [arXiv:2607.18789](https://arxiv.org/abs/2607.18789)
