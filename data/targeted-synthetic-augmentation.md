# Targeted synthetic augmentation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **diagnose-then-patch** synthetic-data workflow: profile a model's competency gaps along an axis (in the paper, per-language capability), then generate *targeted* synthetic data that fills only those cells rather than broad-brush augmenting everything. Agarwal et al. (2026) formalise this as **HOTFIXR** for multilingual SFT: +6.2% in-distribution, +3.7% reduction in catastrophic forgetting, +7.1% on OOD languages — with a small synthetic mixture.

**Prereqs:** [_data-curation.md](./_data-curation.md), [quality-filtering.md](./quality-filtering.md)
**Related:** [dolma.md](./dolma.md), [decontamination.md](./decontamination.md)

---

## What it is

Cross-lingual performance gaps in modern LLMs are **localizable** — the failures are not evenly distributed across (task × language) cells. The targeted-augmentation recipe:

1. **Diagnoses** which cells are weakest (task, language, or capability × language).
2. **Generates** synthetic examples aimed specifically at those cells.
3. **Filters** for quality and de-duplicates against the training corpus.
4. **Mixes** the (small) synthetic set into a standard SFT recipe.

The alternative — broad multilingual augmentation — grows the training corpus far more than needed and typically over-weights already-strong cells while under-serving the tail.

## How it works

- **Diagnostic pass.** Score the base model on a per-(task, language) grid using held-out benchmarks. Rank cells by gap-to-target and pick the worst.
- **Prompted generation.** For each targeted cell, prompt a strong multilingual generator with cell-specific instructions (target language, task template, style constraints). Generate a small pool per cell.
- **Quality filter.** Score generations for correctness, target-language fluency, and task conformance. Drop failures.
- **De-contamination.** Check for overlap with the diagnostic set (paper-level care) and with the pretraining corpus where possible.
- **SFT mixture.** Blend the targeted synthetic set with the standard SFT mixture — the paper emphasises that the synthetic set stays small relative to the base SFT data.

## Why it matters

- **Surgical, not broad.** A small targeted mixture beats a large broad one on both the target languages and OOD languages, without sacrificing English.
- **Tackles catastrophic forgetting head-on.** By keeping the mixture small and the diagnosis honest, the recipe reduces regressions on unrelated capabilities during SFT.
- **Repeatable across axes.** The paper uses language as the axis, but the same diagnose-then-patch loop applies to capability × domain, capability × difficulty, or capability × register.

## Gotchas & tricks

- **Diagnosis is only as good as your grid.** A missing (task, language) cell means the workflow can't patch it. Coverage of the diagnostic grid caps the reach.
- **Generator determines the ceiling.** Weak multilingual generators produce plausible-looking but linguistically-off outputs; the filter has to be strict or the synthetic set poisons SFT.
- **Filter is load-bearing.** Skip quality filtering and the observed gains disappear — the ablation is explicit in the paper.
- **Small mixture size is intentional.** Scaling the synthetic set indefinitely reintroduces catastrophic forgetting; the sweet spot is a few percent of the SFT mixture.
- **Doesn't fix pretraining-level gaps.** If a language is truly absent from pretraining, targeted SFT augmentation helps at the margin but a proper pretraining data intervention is stronger.

## Sources

- Paper: *LLMs Get Smarter from Targeted Synthetic Multilingual Data* — Agarwal, Charaborty, Sorensen, Gupta, Stolcke (Amazon AGI), 2026 — [arXiv 2608.15964](https://arxiv.org/abs/2608.15964) — introduces HOTFIXR and the diagnose-then-patch workflow.
