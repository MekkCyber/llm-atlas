# Sparse-Attention Indexer (DoSA)
*Depth — a lossless Top-K indexer that returns the exact Top-K keys while scoring far fewer candidates.*

**TL;DR:** Modern long-context serving uses **sparse attention**: for each query, score a small Top-K subset of keys rather than all of them. The bottleneck isn't the sparse op itself — it's the *indexer* that picks the Top-K, which still has to score every candidate. **DoSA** (Dots Sparse Attention) replaces exact Top-K scoring with a **certificate-based rejection**: build an upper bound on each candidate's possible score, compare against a lower bound on the Top-K threshold, and only fully score candidates whose certificate doesn't rule them out. Same selected tokens, same loss, much less compute.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md) · [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

A GPU-level technique for making the indexer stage of sparse-attention serving cheaper, **without changing which tokens get selected**. The downstream attention op is unaffected; only the Top-K selection becomes conditional on evidence.

## How it works

**The idea in one line:** Top-K selection only needs *enough evidence to reject a candidate*, not its full score.

**Step 1 — seed set.** Score a small random (or heuristically chosen) subset of keys first. The `K`-th highest score among them is a **lower bound `τ`** on the eventual Top-K threshold.

**Step 2 — upper bounds for everyone else.** For each remaining candidate key `k`, compute an upper bound `U(q, k)` on its full attention score using cheap-to-evaluate structure:
- **Nonnegative-head projections.** Project query and key into a space where head contributions are nonnegative, so partial scoring already upper-bounds the full score.
- **Joint residual bounds.** Bound positive contributions from heads not yet scored by their maximum per-head output.
- **Activation-informed aggregation.** Use the activation magnitude of unscored dimensions to bound negative contributions subtracted by negative heads.

**Step 3 — certificate check.** If `U(q, k) < τ`, candidate `k` is proven *not* in the Top-K and is skipped. Otherwise, score it fully. Update `τ` as new high scores arrive.

**Step 4 — exact-score filter on survivors.** Among candidates that passed the certificate, run exact scoring, keep the final Top-K.

The certificate guarantees *lossless* Top-K: the selected set is identical to what naïve exhaustive scoring would return.

## Why it matters

- **Pure win.** Lossless means no accuracy regression; the only effect is fewer FLOPs and less memory traffic.
- Reported indexer speedups: **up to 2.74× over SGLang**, **1.69× over LiteTopK**.
- Reported end-to-end TTFT (time-to-first-token) speedups: **1.94× on GLM-5.2** and **1.50× on DeepSeek-V3.2**, across English, Chinese, and code workloads.
- Composes with other sparse-attention stacks that use a Top-K indexer upstream of their attention kernel (PagedAttention-style serving, MLA-based decode paths).

## Gotchas & tricks

- **Certificate tightness drives savings.** Loose upper bounds reject almost nothing. The nonnegative-head reparametrization and activation-informed aggregation are the levers; both are architecture-dependent (depend on positive/negative head counts, activation statistics).
- **Seed-set choice.** Random seeds give a stochastic `τ`; a learned heuristic seed (e.g. the previous step's selection) tightens `τ` earlier and prunes more aggressively, but risks correlation drift across tokens.
- **Hot-cold candidate asymmetry.** The savings concentrate on *cold* candidates (clearly not Top-K); hot candidates still need exact scoring. Workloads whose attention mass concentrates on a few keys (short-range focused patterns) see smaller gains.
- **Not a replacement for approximate Top-K.** If you can tolerate a lossy indexer, lower-precision Top-K (hashing, LSH) can be cheaper still — but then you give up the "same selected tokens" property.

## Sources

- Paper: *DoSA: Dots Sparse Attention* — 2026 — arXiv:2610.13990. Evaluated end-to-end on GLM-5.2 and DeepSeek-V3.2 serving stacks.
