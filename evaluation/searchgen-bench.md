# SearchGen-Bench
*Depth — the 20,839-prompt benchmark that measures whether text-to-image models can render knowledge they don't have parametrically.*

**TL;DR:** 20,839 knowledge-intensive image-generation prompts across 12 failure categories, paired with SearchGen-20K training data. Introduced by Wang et al. (2026, arXiv 2607.05382) alongside the [agentic-image-generation](../agents/agentic-image-generation.md) framework. Frontier T2I models score **21–28 out of 100** — the first benchmark to make "the model made up how something looks" numerically comparable across systems.

**Prereqs:** *(none)*
**Related:** [../agents/agentic-image-generation](../agents/agentic-image-generation.md)

---

## What it is

A benchmark for text-to-image generation whose axis of evaluation is **factual correctness** rather than aesthetics. Prompts are constructed so that a model must produce a specific, verifiable visual — a named person, a dated product, an event, a location — not just something plausible.

Companion training set: **SearchGen-20K**, ~20K prompts with retrieval traces, used to co-train the reasoner + generator of agentic T2I systems.

## How it works

### Prompt construction

Prompts span 12 failure categories the authors identify empirically in modern T2I output:

- Named entities (people, products, brands)
- Time-sensitive facts (2024/2025/2026 releases)
- Rare / long-tail concepts
- Cultural specificity
- Numerical or geometric constraints from external data
- Others enumerated in the paper.

Each prompt has a verifiable target — either a reference image, a set of attributes checkable via VLM, or a factual claim the image must respect.

### Scoring

The benchmark scores from 0 to 100 based on whether the generated image respects the verifiable constraints. Standard T2I metrics (CLIPScore, FID) are complementary but insufficient — the point is to catch cases where the image is aesthetically fine but factually wrong.

### Reported numbers

- Frontier proprietary T2I models: **21–28/100**.
- Adding naïve web search: **regresses** on many prompts due to retrieval noise.
- Co-trained agentic pipeline (paper's system): substantial gains but still leaves a large gap to human performance.

## Why it matters

- **Makes a hard problem measurable.** "T2I models hallucinate" was a vibes claim before this benchmark; now it's a number that goes up or down between releases.
- **Separates parametric from contextual knowledge.** By category-tagging failures, it exposes which capability gaps a bigger generator would close (long-tail rare concepts) vs. which require retrieval (time-sensitive facts).
- **Grounds agentic T2I research.** Provides the training data (SearchGen-20K) and the eval loop for the sub-field.

## Gotchas & tricks

- **Retrieval-in-the-loop is part of the intended eval.** Systems can (and are expected to) fetch external context. This is not a closed-book benchmark.
- **Verification is imperfect.** Some categories rely on VLM-based checks, which are themselves subject to error. Paper reports human-verified subsets.
- **Category imbalance.** The 12 categories are not equally represented; aggregate scores should always be paired with per-category breakdown.
- **The training set (SearchGen-20K) and the eval set (SearchGen-Bench, 20,839) are disjoint but drawn from the same distribution — cross-distribution generalization is unmeasured.

## Sources

- Paper: *Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation* — Wang et al., HKUST / Qwen / Waterloo, 2026 — arXiv 2607.05382.
