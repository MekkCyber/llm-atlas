# Unigram Cross-Entropy (UCE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **model-free** measure of word-level alignment between a pretraining corpus and a benchmark. Compute each distribution as a bag-of-words unigram over tokens, then score alignment with cross-entropy: $\mathrm{UCE}(B \| C) = -\sum_w p_B(w) \log p_C(w)$. **Lower UCE → tighter alignment → higher zero-shot accuracy**, consistently across 10 benchmarks, 4 corpora, and 3 model scales (Kim et al. 2026). Lets you score a corpus-vs-benchmark pair, a document-vs-benchmark pair, or a prompt-vs-benchmark pair *without running the model*.

**Prereqs:** [_data-curation.md](_data-curation.md)
**Related:** [_data-curation.md](_data-curation.md) · [decontamination.md](decontamination.md) · [quality-filtering.md](quality-filtering.md)

---

## What it is

Zero-shot benchmark accuracy is often treated as a proxy for model capability. UCE argues a lot of it is actually explained by **lexical surface overlap** between the pretraining corpus and the benchmark. The measure is a cross-entropy between the unigram distribution of benchmark text ($p_B$) and the unigram distribution of a candidate corpus ($p_C$):

$$
\mathrm{UCE}(B \| C) = -\sum_w p_B(w) \log p_C(w)
$$

Lower cross-entropy = benchmark words are frequent in the corpus = better alignment.

Three granularities:

- **Corpus-level UCE.** Score a whole corpus against a benchmark. Useful for pretraining data choice.
- **Document-level UCE.** Score each document against the benchmark. Useful for filtering a corpus to a target benchmark distribution.
- **Prompt-level UCE.** Score a specific prompt. Useful for understanding why a prompt variant gets better zero-shot numbers.

## How it works

### Compute $p_B$ and $p_C$

Tokenize benchmark and corpus with the same tokenizer. Build the unigram distribution of each — frequency of each token in the respective text. Smoothing (add-$\alpha$) handles out-of-corpus tokens.

### Score

Apply the cross-entropy formula above. Report in nats or bits. Lower is better aligned.

### Document and prompt variants

Document-level: $p_C$ is the unigram distribution of a single document. Rank documents by $\mathrm{UCE}(B \| d)$ to find the best-aligned subset.

Prompt-level: $p_B$ is the unigram distribution of one benchmark prompt (plus its answer formatting). Lets you compare prompt phrasings.

### No model required

The measure doesn't invoke the model being evaluated — it's a property of the data pair. This is the point: a reliable zero-shot predictor without compute.

## Why it matters

- **Zero-shot ≈ lexical alignment, more than you'd think.** Across 10 benchmarks, 4 corpora, 3 model scales, UCE and zero-shot accuracy are tightly correlated. Not a toy effect.
- **A cheap data-selection signal.** Pre-filtering corpora or documents by UCE against your intended eval suite is a model-free knob that costs microseconds per document.
- **Uncomfortable for benchmark design.** Evals that reward lexical alignment are measuring *data overlap*, not reasoning. Benchmarks built from vocabulary clearly outside common web corpora are more informative about capability.
- **Complement to [decontamination.md](decontamination.md).** Decontamination catches *verbatim* leakage; UCE measures *distributional* alignment. Both matter.
- **Prompt-level UCE as a prompting signal.** The gap between prompt variants on the same benchmark often reduces to their UCE against the pretraining corpus — a model-free prompt-engineering diagnostic.

## Gotchas & tricks

- **Tokenizer dependence.** UCE values are not comparable across different tokenizers. Fix a tokenizer or report per-tokenizer.
- **Smoothing matters.** Benchmarks almost always have words absent from any corpus sample. Add-$\alpha$ smoothing or Laplace priors keep the log finite; pick one and stick with it.
- **Not a cause-effect story on its own.** UCE predicts accuracy; it doesn't prove the model is memorizing. Pair with contamination checks and task-specific analyses for a full picture.
- **Within-benchmark variance.** Across benchmark *subsets*, UCE's predictive power varies. Fine-grained analysis beats aggregate claims.
- **Doesn't capture syntax or semantics.** It's unigram, so phrases, idioms, and syntactic patterns are invisible. Low UCE guarantees some alignment but not generation quality.
- **Comparable at same scale.** Compare UCE across corpora at a fixed benchmark; comparing across benchmarks requires normalization (benchmark entropy differs).

## Sources

- Paper: *The Lexical Imprint of Pre-training Data* — Kim, Chung, Song, Park, Sung, Shin, 2026 — [arXiv:2602.10657](https://arxiv.org/abs/2602.10657)
