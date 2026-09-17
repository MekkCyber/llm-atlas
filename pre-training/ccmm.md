# Context-Conditional Masked Modeling (CCMM)
*Depth — pretraining objective for tabular foundation models: mask arbitrary cells across the joint, predict from the rest.*

**TL;DR:** CCMM extends masked language modelling to structured tables and to the **joint distribution** `p(x, y | D_context)` rather than the target-only `p(y | x, D_context)` used by prior tabular PFNs. Given a context of training rows plus a query row, CCMM randomly masks arbitrary cells across the query's features *and* target, and trains the model to recover them. Introduced with LimiX-2, it is the joint-pretraining substrate that lets Contextual Mechanism Networks learn causal-adjacent representations while still winning on standard predictive benchmarks.

**Prereqs:** [../architectures/contextual-mechanism-network.md](../architectures/contextual-mechanism-network.md)
**Related:** [../case-studies/limix-2.md](../case-studies/limix-2.md), [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

A masked-prediction pretraining objective for **structured-data foundation models**. Where BERT's masked LM masks tokens in a sequence, CCMM masks **cells across a query row's joint `(x, y)`**, conditioned on a context of unmasked training rows. The context stays fully observed and serves as the in-context conditioning set; the query row is where masking and loss live.

Unlike target-only tabular PFNs — which always mask just `y` and let the model see all of `x` — CCMM's mask is drawn over the full joint of the query row. That single change is what unlocks:

- Joint-distribution modeling (imputation, feature-conditional queries).
- Robustness to real-world missing-value patterns at inference.
- Mechanism-oriented internal representations, because the model must predict features *and* targets from each other.

## How it works

For a synthetic dataset (row set) `D`:

1. **Split.** Sample `K` rows as `D_context` (fully observed) and one row as the query.
2. **Mask.** Draw a random subset `S` of the query row's cells (features and possibly target) to mask; the rest of the query row is observed.
3. **Forward pass.** The Contextual Mechanism Network takes `D_context` + partially-observed query row and outputs a distribution over each masked cell.
4. **Loss.** Negative log-likelihood on the masked positions, summed across `S`. Column-type-specific heads: Gaussian mixture (regression), softmax (classification), span (text-lite).
5. **Batch.** Batches mix datasets so the model learns to *specialize per dataset in-context*.

The mask ratio is annealed (higher during warmup, lower later), and mask patterns are drawn to cover both target-only masks (matching the PFN regime) and feature masks (matching imputation).

## Why it matters

- **Joint queries at inference.** A single CCMM-pretrained model handles prediction, imputation, and feature-conditional queries — no separate imputer needed.
- **Real-world missingness.** Real tables have missing values everywhere; target-only pretraining sees only fully observed features, so it degrades. CCMM sees masked features in every batch and doesn't.
- **Predictive accuracy stays high.** On TabArena / TALENT / BCCO, CCMM pretraining does not lose ground to target-only baselines — it *beats* them, even on the target-only metric.
- **Feature-attention encodes structure.** Because the model must predict features from each other, its feature-attention weights approximate feature dependencies. In LimiX-2, these weights recover the **causal skeleton** of the SCM data generator with high accuracy — a byproduct of mechanism-oriented pretraining.

## Gotchas & tricks

- **Mask distribution matters.** Too few feature masks and the model collapses back to a target-only PFN. Too many, and it never gets fluent at the predictive task. LimiX-2 uses a schedule that emphasizes joint masks in mid-training.
- **Column-type-aware losses.** A single MSE-across-all-columns loss under-weights categorical columns; use per-type heads and normalize.
- **Context row count is a curriculum axis.** Small context sizes at the start; grow to match deployment.
- **SCM data generator quality matters.** CCMM inherits its causal-recovery power from the diversity of graphs / mechanisms in the SCM sampler; a narrow sampler gives narrow representations.
- **Not directly comparable to MLM.** MLM masks discrete tokens in a sequence; CCMM masks typed cells in a table. Loss functions and mask distributions are different.

## Sources

- Paper: *LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence* — Zhang, Ren, Yuan, Zou, … Cui — 2026 — [arXiv:2609.17488](https://arxiv.org/abs/2609.17488).
- Predecessor: LimiX-1 (referenced for the scaling laws used to size LimiX-2).
- Adjacent: TabPFN v2 (target-only tabular PFN pretraining) — Hollmann et al., 2025.
