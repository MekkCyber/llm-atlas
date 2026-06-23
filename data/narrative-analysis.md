# Narrative Content Analysis of Pretraining Data

*Depth — a quantitative framework (NarraBERT + NarraDolma) for measuring multidimensional narrative structure across a pretraining corpus.*

**TL;DR:** Most analyses of LLM pretraining data look at quality and domain (web vs. code vs. academic). Narrative content is a *different* axis — how story-shaped is a passage, how much agency does it depict, how grounded is it in setting and events. **NarraBERT** is a RoBERTa-based classifier across 11 narrative dimensions (agency, setting, events); **NarraDolma** is a labeled 3M-passage subset of Dolma. Result (Johnson et al., 2026): narrative structure is measurable at scale, continuous, and very unevenly distributed across pretraining sources.

**Prereqs:** [dolma](dolma.md), [quality-filtering](quality-filtering.md)
**Related:** [_data-curation](_data-curation.md), [decontamination](decontamination.md)

---

## What it is

A two-artifact framework for characterizing narrative content in large pretraining corpora:

- **NarraBERT.** A RoBERTa-based encoder fine-tuned to score a passage across **11 narrative dimensions** organized into three categories:
  - **Agency** — degree to which characters initiate actions, exhibit goals, react to events.
  - **Setting** — temporal / spatial grounding, sensory detail, world specificity.
  - **Events** — causal structure, temporal ordering, resolution.
- **NarraDolma.** A 3M-passage dataset sampled from Dolma and labeled with NarraBERT scores. Released as a resource so downstream researchers can filter, weight, or analyze pretraining mixes by narrative content.

Importantly, scoring is *continuous* and *multidimensional* — not a binary story/non-story split. A scientific paper can score high on "events" (causal structure) but low on "agency"; a personal essay scores high on "agency" but moderate on "setting"; fiction scores high on all three.

## How it works

Pipeline:

```
Annotated narrative-feature dataset (~10K passages, manually labeled across 11 dims)
  → fine-tune RoBERTa as a multi-output regressor → NarraBERT
  → run NarraBERT over a 3M-passage sample of Dolma → NarraDolma
  → analyze per-source, per-topic distributions
```

The 11-dimension framework draws from narrative theory (story grammar, agency-event structure). The fine-tuning corpus is the bottleneck on quality — it has to be large enough and consistent enough across annotators that the regressor generalizes.

## Why it matters

Pretraining-corpus design has implicit narrative assumptions baked in (web-heavy mixes contain a lot of narrative; code-heavy mixes contain very little). Until NarraDolma, these assumptions were ambient — nobody could measure them. Now you can:

- **Audit the narrative composition of a mix.** Is your data 5% narrative or 50%?
- **Filter or weight by narrative content.** Train a base model with controlled narrative exposure.
- **Correlate downstream behavior with narrative-ness.** Storytelling fluency, theory-of-mind, character modeling, plot coherence — all plausibly track narrative-rich pretraining exposure.

The finding that narrative is *continuously* distributed (not a small island inside a sea of non-narrative) means every mixture-design decision is implicitly setting a narrative slider, whether the curator realizes it or not.

## Gotchas & tricks

- **The 11 dimensions are correlated.** Agency, setting, events tend to co-vary; treat them as a vector but don't pretend they're independent.
- **NarraBERT is a classifier, not a ground truth.** Its labels are model outputs — useful in aggregate, noisy per passage.
- **Source attribution matters.** Narrative-ness is a per-source property; you can't blindly mix two corpora and expect the narrative budget to add linearly (because dedup removes overlap, often narrative-rich overlap).
- **Domain != narrative.** "Books" is a domain; narrative-ness is orthogonal. Some books are heavy on argument and light on narrative; some web prose is intensely narrative.

## Sources

- Paper: *Characterizing Narrative Content in Web-scale LLM Pretraining Data* — Johnson, Ash, Piper, Antoniak, 2026 — https://arxiv.org/abs/2606.19468
- Underlying corpus: [dolma](dolma.md) — the 3T-token open corpus that NarraDolma samples from.
