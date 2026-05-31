# Parallax: Parameterized Local Linear Attention

*Depth — an attention variant grounded in nonparametric regression that upgrades softmax's local-constant estimator to a local-linear one, scaled to LLM training via a parameterized probe.*

**TL;DR:** Cast attention as a nonparametric estimator over KV pairs: softmax attention is the locally-constant Nadaraya–Watson estimator, while Local Linear Attention (LLA) is a locally-linear regression with provably better bias-variance for associative recall. LLA has not scaled because its per-step numerical solver is fragile and expensive. Parallax eliminates the solver by learning a query-like projector that *probes* the KV covariance, plus a hardware-aware decode kernel that pushes arithmetic intensity above FlashAttention's. Competitive quality, faster decode.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md), [qk-norm.md](qk-norm.md)

---

## What it is

A new attention mechanism for LLM pretraining. Sits in the same family as MLA / GQA / linear-attention variants — all replacements for the standard softmax attention block — but motivated from test-time regression theory rather than from KV-cache compression or kernel approximation.

## How it works

The test-time-regression view of attention treats each KV pair as a (covariate, value) sample, and the query as the test point. The estimator maps query → value.

- **Softmax attention** computes $\hat{v}(q) = \sum_i \mathrm{softmax}(qK^\top)_i v_i$ — this is the Nadaraya–Watson **local constant** estimator with the softmax kernel as the weighting.
- **Local Linear Attention (LLA)** fits a *local linear* surface at $q$ — at each query, solve a tiny weighted least-squares problem over the keys. Provably better bias-variance tradeoff for associative recall, but the per-step linear solve is numerically unstable at LLM scale.

**Parallax's substitution.** Replace LLA's solver with a *learned query-like projector* $W_p$ that probes the KV covariance directly:

$$
\text{score}(q, K, V) = \text{Attn}(q, K, V) + \text{LinearTerm}(W_p q, K, V)
$$

The learned probe captures the linear-surface gradient term that LLA's solver computed numerically. Place it within an explicit family parameterized by the bandwidth, the probe construction, and the affine structure — softmax / linear / MLA / Parallax all become design points in the same family.

**Hardware-aware decode kernel.** The Parallax decode kernel restructures the read pattern to raise arithmetic intensity above FlashAttention 2/3 (more FLOPs per byte read from HBM), pushing the operation into the compute-bound regime and matching or beating FA on standard shapes.

## Why it matters

- New family of attention variants with a *principled* (test-time regression) motivation, not just an efficiency hack. Gives a theoretical lens on why specific variants help.
- Decode-kernel improvements at the arithmetic-intensity level address one of the dominant inference bottlenecks (KV-cache-bound attention).
- Drops into a standard Transformer block; can be combined with GQA / MLA caching strategies because it's a different axis (estimator order, not head-sharing).

## Gotchas & tricks

- The "probe" parameter $W_p$ adds a small number of weights per attention head. Compared to MLA's caching savings, the parameter cost is negligible, but accounting matters in cross-architecture comparisons.
- The hardware-aware kernel is the bulk of the speedup; the algorithmic change alone wouldn't move latency. Reproducing requires the kernel.
- LLA's stability story is fragile in float16. Parallax inherits some of this — qk-normalization helps in pretraining (sibling: [qk-norm.md](qk-norm.md)).
- Best for decode-heavy serving (long-context, agentic workloads); prefill-heavy workloads see less benefit because prefill is already compute-bound.

## Sources

- Paper: *Parallax: Parameterized Local Linear Attention for Language Modeling* — Zuo, Pai, Zeng, Dewulf, Hu, Wang — Northwestern / Tilde Research / U. Washington, 2026 — [arXiv 2605.29157](https://arxiv.org/abs/2605.29157).
