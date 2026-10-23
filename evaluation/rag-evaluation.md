# RAG Evaluation (Compositional)
*Depth — evaluating RAG as a *pipeline* rather than a sum of component scores.*

**TL;DR:** Isolated gains on a RAG component (retriever, reranker, generator) often do not survive composition into an end-to-end pipeline. The `lasagna` framework sweeps 450 configurations across 8 benchmarks and corpora from 5K to 200K documents and measures whether an improvement at one stage carries through. Findings: *compositionality is weak*, retrieval-quality and answer-faithfulness demand opposing choices, and the best configuration shifts as the corpus scales.

**Prereqs:** [_data-curation](../data/_data-curation.md)
**Related:** [decontamination](../data/decontamination.md)

---

## What it is

A **RAG pipeline** chains at minimum: query encoder → retriever → reranker → generator (with its context prompt template). Papers typically swap one component, measure a benchmark, and declare an improvement.

RAG evaluation (compositional) asks: *does that improvement survive when the other components change?* It frames the pipeline as a space (axes = components, points = concrete choices for each) and sweeps a large sample of combinations under matched controls.

The `lasagna` framework is one concrete instance:

- Axes: retriever family, embedding model, reranker, number of retrieved docs, prompt template, generator model, corpus.
- Benchmarks: 8 QA/retrieval-style benchmarks.
- Corpora: 5K to 200K documents (varying knowledge base size).
- Configurations: 450 points sampled with balanced coverage over axes.

---

## How it works

Three measurement moves:

1. **Isolated-vs-composed delta.** For each component change *c*, compute the mean score change *when all other axes are held fixed* (isolated gain) and *averaged over the other-axis distribution* (composed gain). Compositional reliability is `composed_gain / isolated_gain`.

2. **Axis-conflict detection.** Score the pipeline on *two* metrics (retrieval quality vs. answer faithfulness). Compute the correlation of per-configuration ranks. Low or negative correlation means the two metrics disagree — the pipeline that retrieves best is not the one that answers most faithfully.

3. **Scale-dependent best configuration.** Repeat the full sweep at corpora of 5K, 50K, 200K docs. The argmax configuration shifts with corpus size — larger corpora reward higher-recall retrievers that smaller corpora penalized.

---

## Why it matters

A single-benchmark improvement in a RAG paper is weak evidence that the new component helps in production. This framework replaces that with a measurement that:

- **Flags fragile wins.** A component that improves 2 of 450 pipelines is probably not real.
- **Exposes metric conflicts.** You cannot optimize retrieval and faithfulness jointly without a Pareto choice; the framework surfaces it.
- **Catches scale-dependence.** Many "better retrievers" lose at small corpora or lose at large ones; the sweep makes the regime explicit.

In practice this closes an evaluation gap that preference leaderboards for RAG (which score end-to-end answers against ground-truth) also miss, because they don't vary the pipeline systematically.

---

## Gotchas & tricks

- **Sample size matters.** 450 is enough to see compositionality statistics; a sweep of 20 is not.
- **Judge-LLM contamination.** When faithfulness is scored by a judge LLM, make sure the judge sees the same retrieved docs the generator saw — otherwise the "faithfulness" score drifts into general-knowledge QA.
- **Report the recall–faithfulness Pareto.** A single scalar hides the conflict this framework was built to surface.
- **Corpora, not just documents.** The *distribution* of the corpus (domain, redundancy, chunk length) matters as much as its size.

---

## Sources

- Paper: *Retrieval Augmented Generation Compositionality* — anonymous (academic), 2026 — [arXiv:2610.29722](https://arxiv.org/abs/2610.29722). Introduces the `lasagna` framework.
- Background: RAGAS, BEIR, and other RAG benchmarks that score end-to-end but do not vary the pipeline.
