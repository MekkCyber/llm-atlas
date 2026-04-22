# Deduplication
*Depth — removing duplicate and near-duplicate documents from a training corpus.*

**TL;DR:** Web-scale corpora contain massive duplication — the same document appears at many URLs, the same paragraph is copy-pasted across sites, the same boilerplate repeats millions of times. Training on duplicates hurts: Lee et al. (2022) showed it increases memorization, wastes compute, and can mildly hurt validation loss. Modern pipelines dedupe at multiple granularities (URL, document, paragraph, n-gram) using exact hashing for speed and MinHash+LSH for approximate near-duplicate detection.

**Prereqs:** [tokenization](../fundamentals/_tokenization.md)
**Related:** [quality-filtering](quality-filtering.md), [decontamination](decontamination.md), [dolma](dolma.md), [_data-curation](_data-curation.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

"Duplicate" is not one thing. Web corpora contain duplication at every granularity:

| Level | What's duplicated | Typical cause |
|---|---|---|
| URL | Same URL re-crawled | Multiple snapshots of Common Crawl |
| Document | Byte-identical documents at different URLs | Mirrors, reposts, CDN caching |
| Near-duplicate document | 95%+ similar, different boilerplate | Forks, partial rewrites, templated sites |
| Paragraph | Same ~5-sentence block across docs | Copy-paste, boilerplate (e.g. cookie notices) |
| N-gram | Long n-gram (e.g. 50 tokens) repeated | Quotes, licensing text, auto-generated filler |

Dedup is the pipeline step that removes all of these before training. The levels need different algorithms and run at very different throughputs.

## How it works

### Exact deduplication

**URL and document-level.** Hash the URL (or the whole document text, normalized) and drop duplicates. O(n) with a hash table; embarrassingly parallel. Removes the bulk of "trivially repeated" data — often 30–50% of raw Common Crawl volume.

### Near-duplicate deduplication with MinHash + LSH

Exact hashing doesn't catch documents that are 98% identical with different boilerplate. The standard approach is **MinHash** (Broder, 1997):

1. Tokenize each document into shingles (e.g. 5-token overlapping windows).
2. Apply `k` hash functions to the set of shingles; for each, record the minimum hash value → a signature vector of length `k`.
3. Two documents' signatures match in expectation proportional to their Jaccard similarity.
4. Use **Locality-Sensitive Hashing (LSH)** to quickly find all pairs whose signatures collide in enough rows to imply high Jaccard similarity (typical threshold: 0.8).

This is the de facto standard for near-duplicate detection at web scale. Typical parameters: 128–256 hash functions, bands-and-rows LSH tuned to threshold ≈ 0.8.

### Paragraph and n-gram deduplication

Two documents that share several large identical paragraphs but differ in others are *not* caught by document-level MinHash (average similarity stays low), but they inject the duplicated blocks repeatedly into training.

**Paragraph dedup:** split each document into paragraphs, hash each, drop a paragraph if its hash has appeared in prior documents. Simple but aggressive — can destroy document coherence.

**N-gram dedup (Lee et al. 2022, *ExactSubstr*):** build a suffix array over the full corpus, find all substrings of length ≥ L (L = 50 tokens is common) that appear more than once, remove all but one occurrence. Expensive to run (single-machine suffix array doesn't scale past a few TB) but catches boilerplate-class repetition that other methods miss.

### Multi-level pipelines

Modern pipelines run several of these in sequence:

```
Raw CC  →  URL dedup  →  document MinHash+LSH  →  paragraph or n-gram dedup  →  downstream filtering
```

OLMo 2's pipeline (applied to its pretraining mix) reports URL + paragraph + n-gram deduplication over the web portion. Llama 3 reports URL + document-level + n-gram. Exact combinations vary, but the multi-level pattern is consistent across recent labs.

## Why it matters

- **Memorization reduction.** Lee et al. (2022) showed that near-duplicate deduplication reduces the rate at which models verbatim-regurgitate training text — privacy-relevant for corpora with personal information, legally relevant for copyright.
- **Better compute utilization.** Duplicates push the effective training tokens down. Dedup + more fresh tokens usually beats non-dedup + same total volume at a given compute budget.
- **Mild quality improvement.** Lee et al. and follow-ups report small but consistent validation-loss and downstream-benchmark lifts from deduplication. The effect is usually < 1 point on most tasks; occasionally larger on tasks sensitive to quoting / boilerplate patterns.
- **Prerequisite for honest evaluation.** Duplicates often straddle train/eval boundaries, inflating benchmarks — dedup is upstream of [decontamination](decontamination.md).

## Gotchas & tricks

- **Order of levels matters.** Coarser-grained dedup first (URL → document → paragraph → n-gram). Running n-gram before document dedup is wasteful because document-level hashes catch most of the easy cases cheaply.
- **Shingle size governs MinHash sensitivity.** Too small (e.g. 3-token): random matches between unrelated docs inflate Jaccard. Too large (e.g. 20-token): small edits break the match. 5–9 tokens is the usual range.
- **LSH bands-and-rows tuning is non-obvious.** The threshold at which a pair is "likely duplicate" depends on the number of bands × rows — `(1/b)^(1/r)` gives the approximate threshold. Off-by-one here changes how much you drop by 10s of percent.
- **Paragraph dedup can over-remove.** Legitimate repeated constructs (table headers, common phrases in formal writing) get flagged. Some pipelines exempt paragraphs below a length threshold.
- **N-gram dedup is the expensive step.** Full-corpus suffix-array-based n-gram dedup does not trivially distribute; most implementations use approximate/streaming variants at the cost of some false negatives.
- **Normalization before hashing.** Lowercase, strip whitespace, drop punctuation — otherwise URL `?ref=twitter` vs. `?ref=reddit` appear distinct, and 12-space vs. 4-space indent look different.
- **Cross-lingual dedup.** Machine-translated copies of the same article are duplicates in practice but invisible to standard MinHash (which operates on surface tokens). Semantic dedup via embeddings exists but is expensive; most labs skip it.
- **Don't dedupe inside curated sources.** arXiv, textbook corpora, etc. often contain legitimate near-duplicates (preprint vs. accepted version, problem-solution pairs) that dedup would destroy. Treat curated sources more conservatively than web crawl.

## Sources

- Paper: *Deduplicating Training Data Makes Language Models Better* — Lee, Ippolito, Nystrom et al., 2022 — the empirical case for dedup plus the ExactSubstr n-gram method.
- Paper: *On the Resemblance and Containment of Documents* — Broder, 1997 — introduces MinHash for near-duplicate detection.
- Paper: *Similarity Estimation Techniques from Rounding Algorithms* — Charikar, 2002 — SimHash, the MinHash alternative.
- Paper: *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data* — Wenzek et al., 2019 — early web pipeline including exact and near-duplicate dedup.
- Paper: *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* — Soldaini et al., 2024 — AI2's documented multi-level dedup pipeline (precursor to OLMo 2).
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — applies URL + paragraph + n-gram dedup.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — reports URL + document-level + n-gram dedup.
