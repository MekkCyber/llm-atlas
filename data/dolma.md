# Dolma, OLMo-Mix-1124, Dolmino Mix 1124
*Depth — AI2's line of open pretraining corpora, from Dolma 1.x through the OLMo 2 mixes.*

**TL;DR:** Dolma is AI2's open pretraining corpus, the data backbone for the OLMo series. Across versions it has grown in scale and curation: **Dolma v1.x** (Soldaini et al. 2024) was the original 3T-token open release; **OLMo-Mix-1124** is the refreshed Stage 1 corpus for OLMo 2; **Dolmino Mix 1124** is the small curated Stage 2 / mid-training mix for OLMo 2. All three are fully documented, source-attributable, and released under open licenses — the most concrete public reference for what a modern open pretraining corpus contains and how it's processed.

**Prereqs:** [deduplication](deduplication.md), [quality-filtering](quality-filtering.md), [decontamination](decontamination.md)
**Related:** [mid-training](../pre-training/mid-training.md), [_data-curation](_data-curation.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

Three artifacts, two roles, one release lineage:

| Artifact | Role | Scale | Used by |
|---|---|---|---|
| **Dolma v1.x** | Stage 1 pretraining | ~3T tokens (v1.7) | OLMo 1, Pythia-successors |
| **OLMo-Mix-1124** | Stage 1 pretraining | ~5T tokens | OLMo 2 |
| **Dolmino Mix 1124** | Stage 2 / [mid-training](../pre-training/mid-training.md) | ~100B tokens | OLMo 2 |

Each is released as:

- Final shuffled shards (ready to tokenize and train)
- Per-source metadata (every document traceable to its origin)
- Documentation of filters, dedup parameters, and source licenses

The data is downloadable, the code that produced it is in the `dolma` toolkit repo, and the process is reproducible on new Common Crawl snapshots by anyone with the compute.

## How it works

### Source composition

OLMo-Mix-1124 is dominated by filtered Common Crawl, with curated non-web corpora adding diversity:

| Source | Role | Approximate share |
|---|---|---|
| Filtered Common Crawl | General web prose, world knowledge | bulk |
| StarCoder | Code | medium |
| arXiv | Academic / math-heavy prose | small |
| pes2o (peer-reviewed papers) | High-quality academic | small |
| Wikipedia (multilingual) | Encyclopedic reference | small |
| Stack Exchange | Q&A, technical discussion | small |
| Books (public-domain / licensed) | Long-form narrative | small |

Exact token percentages vary by version; consult the OLMo 2 report tables for the current mix. The important property is that every document is attributable to its source — unlike some aggregated corpora (early RedPajama, for instance) where source mixing destroyed provenance.

### Pipeline stages

The Dolma toolkit applies a fixed sequence to each source:

```
Raw source  →  Language ID  →  Quality filter  →  Dedup  →  PII scrub  →  Decontaminate  →  Shuffle
```

- **Language ID.** Keep documents classified as target languages (English + N multilingual).
- **[Quality filter](quality-filtering.md).** Source-specific. Web: FastText classifier trained on high-quality reference. Code: Stack-style license + star filters. Academic: minimal filtering (already curated).
- **[Dedup](deduplication.md).** URL → paragraph → n-gram, with per-source tuning.
- **PII scrub.** Regex-based scrubbing of emails, phone numbers, IDs. Conservative — false positives (dropping legitimate text that looks like PII) are preferred over false negatives.
- **[Decontaminate](decontamination.md).** N-gram overlap against a curated eval list.
- **Shuffle.** Global or within-shard, producing the final training-ready mix.

### Dolmino Mix 1124 — the Stage 2 mix

Dolmino is a *different* mix from OLMo-Mix-1124, designed for [mid-training](../pre-training/mid-training.md):

- **Much smaller** (~100B vs. ~5T tokens).
- **Heavier on math, code, and academic.** Explicit inclusion of math-specific corpora (OpenWebMath, GSM-style problem-solution pairs) and filtered code (high-rated GitHub, Stack Exchange).
- **Contains FLAN-style instruction-formatted text as continuations.** Not used as supervised pairs — just included as high-quality prose that happens to be in instruction form. Actual SFT comes later in post-training.
- **Synthetic rewrites.** Teacher-model-generated rewrites of hard examples to amplify the curated signal.

Dolmino's role in OLMo 2 is described in detail in [mid-training.md](../pre-training/mid-training.md) and the OLMo 2 case study.

## Why it matters

- **Reproducible reference point.** Most pretraining corpora are either closed (C4, Llama's mix, GPT's mix) or documented only at a high level (RedPajama, SlimPajama). Dolma / OLMo-Mix / Dolmino are documented at the level where you could re-derive them from the same Common Crawl snapshots. That's rare.
- **The Stage 1 vs. Stage 2 split, made concrete.** The [mid-training](../pre-training/mid-training.md) concept is abstract until you look at a concrete Stage 2 mix. Dolmino is the clearest public example.
- **A ready substrate for ablations.** If you want to study "what does data filter X do to downstream benchmarks" or "how does Stage 2 mix composition matter", you can start from Dolmino, vary one dimension, and compare.
- **Source-attribution for legal / compliance questions.** The per-document provenance makes it tractable to answer "does this corpus contain source X?" — useful for takedown requests and downstream compliance checks.

## Gotchas & tricks

- **Version specifics shift fast.** Dolma has gone through 1.0, 1.5, 1.6, 1.7; OLMo-Mix-1124 is the November 2024 snapshot. When referencing numbers, cite the specific version — "Dolma" ambiguously can mean any of several.
- **Dolmino's benchmarks are contaminant-sensitive.** Because Dolmino deliberately includes math/code/academic corpora close to eval domains, the decontamination pass matters more here than for a generic web mix. Check the contamination audits in the OLMo 2 report before using Dolmino for new benchmarks.
- **Not a fixed target.** Future Dolma/Dolmino versions will change. Pin to a specific release (URL + commit hash) for reproducibility.
- **Stage 1 and Stage 2 are not interchangeable.** OLMo-Mix-1124 is bad as a mid-training mix (too broad), and Dolmino is bad as a Stage 1 mix (too narrow, too small). Their design assumes the two-phase pipeline.
- **License heterogeneity.** Different sub-corpora are under different licenses. Check the per-source license table before using Dolma in a commercial product.
- **Tokenizer is separate.** The Dolma release is raw text; pick a tokenizer appropriate to your target model. OLMo 2 ships its own BPE; you don't have to use it.

## Sources

- Paper: *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* — Soldaini et al., 2024 — the original Dolma release and pipeline documentation.
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — documents OLMo-Mix-1124 and Dolmino Mix 1124.
- Repo: `dolma` toolkit — https://github.com/allenai/dolma — the processing code.
- Related data releases referenced for comparison: *The Pile* (Gao et al., 2020); *RefinedWeb* (Penedo et al., 2023); *RedPajama-v2*; *FineWeb* and *FineWeb-Edu* (Penedo et al., 2024); *DCLM* (Li et al., 2024).
