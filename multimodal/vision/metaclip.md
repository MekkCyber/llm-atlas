# MetaCLIP
*Depth — CLIP with a published, reproducible data curation recipe.*

**TL;DR:** CLIP's 400M-pair WIT dataset was private; its curation recipe was described only at high level. **MetaCLIP** (Xu et al., ICLR 2024) publishes a concrete algorithm: start from ~1.6B CommonCrawl image-text pairs, substring-match against **~500K metadata entries** (WordNet synsets, Wikipedia unigrams/bigrams/titles), and **balance via per-entry capping at t=20,000**. Trains identical-architecture CLIP on this reproducible 400M dataset. **Beats OpenAI CLIP** at every scale (B/32: 65.5% vs 63.4%; B/16: 70.8% vs 68.3%; bigG/14 at 2.5B scale: **82.1% IN zero-shot**).

**Prereqs:** [clip](clip.md), [vit](vit.md)
**Related:** [siglip](siglip.md) · [llava](llava.md)

---

## What it is

Xu et al., *Demystifying CLIP Data*, ICLR 2024, arXiv 2309.16671.

Same architecture as CLIP, same training recipe (InfoNCE softmax loss, same batch size, same optimizer). **The contribution is the data pipeline.** Demonstrates that CLIP's performance is primarily driven by data curation, not model size or hyperparameters — and makes that curation reproducible.

---

## How it works

### Metadata (Sec. 3.1)

**~500K metadata entries** (matches CLIP's ~500K queries):

| Source | Count |
|---|---|
| WordNet synsets | 86,654 |
| Wikipedia unigrams (freq ≥ 100) | 251,465 |
| Wikipedia bigrams (PMI threshold ≈ 30) | 100,646 |
| Wikipedia article titles | 61,235 |

(Some duplication; total unique ≈ 500K.)

### Substring matching (Sec. 3.2)

- **Source pool**: ~1.6B image-text pairs from CommonCrawl.
- For each pair, check if the caption contains **any metadata entry as a substring**.
- ~50% of English pairs survive → ~5.6B (pair, matched-entry) hits across all entries.

This step is cheap and deterministic. No ML judgment in the filter.

### Balancing (Sec. 3.3–3.4, Algorithm 1)

The problem: without balancing, common entries (like "photo") collect millions of hits, drowning out rare entries.

- **Cap per metadata entry**: `t = 20,000` (exactly matches CLIP's documented "up to 20k pairs per query").
- For each pair, selection probability:
  ```
  p(pair) = min( t / count(entry), 1 )
  ```
  summed over its matched entries.
- Result: "photo" (with 54M raw matches) is sampled at 20K/54M rate ≈ 0.04% per pair; rare entries at 100% per pair.

This produces a balanced distribution over concepts without discarding pairs that have less-common entries.

### Scale

- **MetaCLIP-400M**: controlled scale match to CLIP.
- **MetaCLIP-1B**: 1B pair scale.
- **MetaCLIP-2.5B**: full-scale release.

Architecture and training recipe held identical across scales.

---

## Results

| Model | Dataset | IN zero-shot |
|---|---|---|
| OpenAI CLIP ViT-B/32 | WIT-400M | 63.4% |
| **MetaCLIP-400M B/32** | MetaCLIP-400M | **65.5%** |
| OpenAI CLIP ViT-B/16 | WIT-400M | 68.3% |
| **MetaCLIP-400M B/16** | MetaCLIP-400M | **70.8%** |
| OpenAI CLIP ViT-L/14 | WIT-400M | 75.5% |
| **MetaCLIP-400M L/14** | MetaCLIP-400M | 76.2% |
| MetaCLIP-1B L/14 | 1B | ~77% |
| **MetaCLIP-2.5B ViT-bigG/14** | 2.5B | **82.1%** |

Same architecture, better data → better results. At 2.5B scale, bigG/14 beats CLIP's largest by ~6 points.

---

## Why it matters

- **Reproducible.** The full recipe is published. You can rebuild a 400M CLIP-grade dataset from CommonCrawl.
- **Isolates data as the dominant factor.** With architecture held constant, MetaCLIP's gains come purely from data curation. Validates the CLIP paper's implicit hypothesis.
- **Enables open alternatives.** Before MetaCLIP, the open-source alternative to OpenAI CLIP was OpenCLIP (trained on LAION, which has its own issues). MetaCLIP offers a cleaner open baseline.
- **Scales cleanly.** The recipe extends to 2.5B pairs without modification.

---

## Gotchas & tricks

- **Metadata choice matters.** WordNet + Wikipedia is a reasonable universe; tweaking it changes the balance. Don't naïvely extend.
- **t = 20K is load-bearing.** Too low → data starvation. Too high → imbalance returns.
- **PMI threshold for bigrams ≈ 30.** Bigrams with high pointwise mutual information are retained; random word co-occurrences are dropped.
- **Source pool matters.** MetaCLIP uses CommonCrawl directly; DataComp uses a different pool. Source-pool differences create dataset-level differences not captured by the recipe alone.
- **Multilingual handling.** MetaCLIP's metadata is English. Multilingual variants need multilingual metadata; not in the paper.
- **Can be combined with SigLIP loss.** The data recipe is orthogonal to the loss. Published SigLIP variants also run on MetaCLIP-like curated data.

---

## Sources

- Paper: *Demystifying CLIP Data* — Xu, Xie, Tian et al., ICLR 2024, arXiv 2309.16671.
- Repo: https://github.com/facebookresearch/MetaCLIP — official release.
- Paper: *Learning Transferable Visual Models From Natural Language Supervision (CLIP)* — Radford et al., 2021 — the architecture and loss MetaCLIP inherits.
