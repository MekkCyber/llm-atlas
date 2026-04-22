# Data Curation

*Taxonomy — the pipeline stages that turn raw web crawl into a training-ready corpus.*

**TL;DR:** A raw Common Crawl snapshot is ~250TB of HTML. Turning it into a training corpus means running a sequence of filtering, cleaning, and deduplication passes, each targeting a specific class of junk: duplicates, low-quality text, sensitive data, benchmark contamination. Modern pipelines ([Dolma](dolma.md), Llama 3's, FineWeb, DCLM) share a recognizable structure — **language ID → quality filter → dedup → PII scrub → decontaminate → shuffle** — and differ in the specific classifier each step uses. Pipeline order matters: coarse/cheap passes first, expensive passes last.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [deduplication](deduplication.md) · [quality-filtering](quality-filtering.md) · [decontamination](decontamination.md) · [dolma](dolma.md)

---

## The problem

Raw web data is ~90% junk by any reasonable quality metric: navigation menus, SEO spam, auto-generated filler, malformed HTML, duplicated mirrors, machine-generated content. Training on it directly is worse than training on nothing — the model wastes capacity learning the junk distribution.

But curation isn't one thing. "Junk" decomposes into specific classes, each needing a different detector:

- Duplicated documents (same URL, same paragraph, same n-gram).
- Low-quality prose (boilerplate, ad copy, semi-coherent writing).
- Wrong language.
- Sensitive data (PII, credentials, medical records).
- Benchmark leakage (training data containing eval answers).
- Unwanted licenses (when producing a commercial-safe corpus).

Treating all of these with one filter fails — each requires a different classifier, threshold, and cost profile. The curation pipeline is the sequence of passes that tackles them.

## The shared pattern

Every modern pipeline follows roughly this order:

```
Raw snapshot
    │
    ├─ 1. Language ID         ── keep only target languages
    │
    ├─ 2. Heuristic quality   ── drop broken / tiny / garbage documents
    │      (length, punct ratio, banned phrases, <script>-heavy pages)
    │
    ├─ 3. Classifier filter   ── FastText or perplexity-based
    │
    ├─ 4. Deduplication       ── URL → document → paragraph → n-gram
    │
    ├─ 5. PII scrubbing       ── regex + heuristics for emails, phones, IDs
    │
    ├─ 6. Decontamination     ── n-gram overlap against eval benchmarks
    │
    ├─ 7. Source-license filter ── (optional, for commercial-safe corpora)
    │
    └─ 8. Shuffle             ── produce training-ready shards
```

Order matters:

- Cheap filters early (language ID, heuristics) drop 30–70% of volume before expensive passes run.
- Dedup before quality filter *can* be wrong — running a FastText classifier on 10× the documents wastes throughput. But some pipelines dedup after classifier filtering because classifiers run fast enough to make order moot.
- Decontamination is late: you want to check against the final surviving documents, not everything you threw away.

Every step has a false-positive / false-negative tradeoff. Tighter = smaller, higher-quality corpus but possibly below compute-optimal token count; looser = more tokens but more junk. Labs choose differently.

## Variants

| Stage | Techniques | Cost | What it catches / misses |
| --- | --- | --- | --- |
| Language ID | fastText language classifier (e.g. CC-Net's) | Very cheap | Catches wrong language; misses multilingual documents |
| Heuristic quality | Gopher rules, C4 rules (length, punct, banned phrases) | Very cheap | Catches obvious junk; misses subtle low-quality |
| [Quality filtering](quality-filtering.md) | FastText classifier on reference pool (DCLM, FineWeb-Edu), perplexity-based (CCNet) | Cheap | Catches low-quality prose; reference-pool choice biases corpus |
| [Deduplication](deduplication.md) | URL hash, document hash, MinHash+LSH, paragraph hash, n-gram suffix array | Cheap → expensive | Catches verbatim + near-duplicates; misses paraphrased copies and cross-lingual duplicates |
| PII scrubbing | Regex (emails, phones, card numbers), Named-entity recognition for harder cases | Cheap (regex) to expensive (NER) | Catches structured PII; misses narrative PII |
| [Decontamination](decontamination.md) | N-gram overlap vs. eval sets (typically 8-13 grams) | Cheap (Bloom filter) | Catches verbatim leakage; misses paraphrased and structural leakage |
| License filter | Per-source license tags, manual exclusion lists | Very cheap | Catches flagged sources; misses unattributed copies |

## How to choose

**For a new pretraining corpus, adopt the full pipeline.** Skipping any one step creates a known failure class. Specific recommendations:

- **Language ID**: use the standard FastText language classifier (CCNet's version). Not a research frontier.
- **Heuristic quality**: reuse Gopher or C4's rule set as a cheap first pass. No need to invent.
- **[Classifier quality filter](quality-filtering.md)**: FastText classifier trained on a reference pool. The reference pool is where the judgment goes — Wikipedia-as-positive biases toward encyclopedic prose; DCLM-style OpenHermes/ELI5-as-positive biases toward reasoning. Run small-model ablations to pick.
- **[Dedup](deduplication.md)**: at minimum URL + document MinHash. Add paragraph and/or n-gram if you can afford the compute. Order: coarser before finer.
- **PII**: regex is the bare minimum; consider source-specific rules for high-PII-risk sources.
- **[Decontamination](decontamination.md)**: against the full public eval list you intend to report. Re-run when new benchmarks enter your eval set.
- **License**: only needed for commercial / licensable corpora.

**For iterating on data choices**, don't re-run the whole pipeline each time. Cache the output of each stage and only re-run from the point where your experiment varies. Dolma's toolkit is designed for this.

### The main live research frontier

Stages 1, 2, 4, 5, 7 are mostly solved — pick a well-known recipe and move on. The live research axis is **stage 3 (quality filtering)**: choice of classifier, choice of reference pool, choice of threshold. DCLM and FineWeb-Edu are recent milestones, but the optimal choice for a given target (math, code, reasoning, general) is still an open question worth ablations.

## Adjacent but distinct

- **[Mid-training](../pre-training/mid-training.md) / Stage-2 mixtures.** A curated subset of the base corpus, reweighted and re-processed for a specific purpose. Uses data-curation primitives but is about mixture design, not cleaning.
- **Synthetic data pipelines.** Teacher-model-generated training data (instruction-tuning, reasoning traces). Different primitives — generation + validation rather than filter + dedup.
- **Evaluation data prep.** Building held-out eval sets. Uses some of the same techniques (dedup, language ID) but with stricter rules.

## Sources

- Paper: *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* — Soldaini et al., 2024 — most thoroughly documented open curation pipeline.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — describes Llama 3's curation pipeline (less source-attributable but still detailed).
- Paper: *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data* — Wenzek et al., 2019 — early end-to-end pipeline.
- Paper: *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* — Rae et al., 2021 — classical heuristic-rule catalog.
- Paper: *DataComp-LM* — Li et al., 2024 — benchmarks different curation choices rigorously.
- Paper: *The FineWeb Datasets* — Penedo et al., 2024 — FineWeb and FineWeb-Edu curation recipes.
