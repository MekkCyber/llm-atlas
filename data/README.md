# Data

*How training data is sourced, cleaned, filtered, and composed — the upstream decisions that determine what a model can learn.*

---

## What This Is

Data pipelines are the silent half of LLM training. Architecture and algorithm get the papers; data quality decides whether any of it works. This folder covers the curation, filtering, and mixture choices that shape a base model, plus the synthetic-data pipelines now used for post-training.

---

## What Belongs Here

- **Corpora** — web crawls, books, code, academic papers, licensed data.
- **Filtering** — quality classifiers, perplexity filters, language ID, safety filters.
- **Deduplication** — exact, near-duplicate (MinHash), semantic.
- **Tokenization strategy** — vocabulary size, multilingual tradeoffs, domain-specific tokens.
- **Mixture design** — domain weights, curriculum, data scheduling.
- **Synthetic data** — generation pipelines, quality control, distillation corpora.
- **Contamination** — how eval leakage enters pretraining corpora and how to detect it.

## Reading Order

1. Raw corpora (what's out there)
2. Filtering & quality signals
3. Deduplication
4. Tokenization strategy
5. Mixture design
6. Synthetic data generation
7. Contamination & eval leakage

---

## Overview Pages (taxonomies)

- [Data curation](_data-curation.md) — the sourcing → filtering → dedup → decontamination → mixture pipeline.

## Concept Pages (depth)

- [Dolma](dolma.md) — AI2's open corpus and OLMo-Mix / Dolmino pipelines.
- [Deduplication](deduplication.md) — URL, document, paragraph, and n-gram dedup.
- [Quality filtering](quality-filtering.md) — FastText / classifier / perplexity filtering.
- [Decontamination](decontamination.md) — removing eval leakage from training corpora.

---

## Related

- [fundamentals/](../fundamentals/) — tokenization as a primitive.
- [pre-training/](../pre-training/) — how data interacts with scaling laws.
- [post-training/](../post-training/) — synthetic data for SFT and RL.
- [evaluation/](../evaluation/) — contamination detection.
