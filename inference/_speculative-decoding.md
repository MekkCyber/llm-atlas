# Speculative Decoding

*Taxonomy — inference-time acceleration techniques that generate a *draft* of multiple tokens quickly, then verify them against the target model in one forward pass.*

**TL;DR:** Autoregressive LLM decoding runs one token per forward pass. Speculative decoding runs a cheap "draft" model (or head) that proposes $k$ tokens, then a single forward pass of the target model that accepts a prefix of the draft matching its own distribution. On accepted tokens the speedup is proportional to $k$; on rejected ones the target model still made forward progress, so the worst case is one target step. The three main axes are **who drafts**, **how verification is done**, and **what "lossless" means** for that scheme.

**Related taxonomies:** n/a
**Depth files covered here:** [orthrus](orthrus.md)

---

## The problem

Decoder-only LLMs decode serially — token $t$ needs the full forward pass at position $t-1$ before it can be sampled. Even with a paged KV cache and continuous batching, the per-token latency floor is one forward pass. For long generations that's a lot of dependent forward passes, most of them spent on tokens that are locally obvious.

Speculative decoding exploits the fact that a cheaper predictor is often *right* about the next few tokens; it just cannot be trusted to always be right. If verification of the cheap predictor's output is itself cheap (one target forward pass over $k$ tokens), the speedup on easy stretches can approach $k \times$ while losing nothing on hard stretches.

## The shared pattern

Every variant follows the same skeleton:

1. **Draft.** A cheap source proposes tokens $x_{t+1}, \dots, x_{t+k}$. The source can be a smaller model, a head grafted onto the target model, a diffusion head, an n-gram lookup, or a cached prompt continuation.
2. **Verify.** The target model runs one forward pass over the draft, producing logits at each drafted position.
3. **Accept / reject.** Sampling rules (rejection sampling for lossless equivalence, or heuristics for lossy variants) decide the longest accepted prefix.
4. **Extend or fall back.** On the first rejection the target's own distribution generates the next token; on full acceptance the process repeats from position $t + k$.

The interesting differences between variants live entirely in step 1 (draft source) and step 3 (accept rule).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Draft-model speculative decoding (no depth file yet) | Smaller target-family model drafts; target verifies | Needs a paired draft model | Any target with an accessible small sibling |
| EAGLE (no depth file yet) | A drafter head trained on target's hidden states drafts token candidates | Requires a distillation stage | Deployment where you can afford one training run |
| Medusa (no depth file yet) | Multiple parallel token-prediction heads grafted onto the target | Head quality caps speedup | Serving without an accompanying draft model |
| [orthrus](orthrus.md) | Hybrid autoregressive-diffusion drafter with an intra-model consensus mechanism claiming losslessness | Losslessness depends on numerical precision (BF16 vs FP32) | Serving with FP32 or precision-audited pipelines |
| Prompt-lookup / n-gram (no depth file yet) | Draft by copying earlier tokens matching a suffix in the prompt or generation | Cheap; only helps on repetitive text | Templated / structured outputs |

## How to choose

- **Best default when you already have a small sibling model in the same family:** draft-model speculative decoding. Reliable ~2–3× speedup on typical decoding.
- **When you cannot ship a second model but can afford training a head:** EAGLE. Better acceptance rate than parallel-heads variants at similar cost.
- **When acceptance rates on your workload are already high (structured / low-entropy):** prompt-lookup — free implementation, no training.
- **When you need or want higher parallelism than draft-verify can give:** hybrid drafters like Orthrus that use diffusion or masked-token heads. Watch for the precision caveat surfaced in [orthrus](orthrus.md).

The **"lossless"** label is worth reading carefully. Rejection-sampling-based schemes are *lossless in distribution* only under exact-precision arithmetic. Under BF16 or FP8 rounding, exact-trajectory equivalence to the target model can fail while downstream benchmarks stay flat. Ship a trajectory-equivalence test alongside benchmark evaluation whenever "lossless" appears in the pitch.

## Adjacent but distinct

- **Continuous batching / paged attention:** orthogonal — increases throughput by packing requests, not by drafting.
- **Prefill-decode disaggregation:** orthogonal — restructures where compute happens across accelerators, not what tokens are proposed.
- **Distillation:** related but offline — trains a smaller model to imitate the larger one; the small model can then serve alone, or become a speculative-decoding drafter.

## Sources

- *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias — 2023 — canonical formulation.
- *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al. — 2024.
- *Medusa: Simple LLM Inference Acceleration Framework* — Cai et al. — 2024.
- *How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus* — Koziev, Sinev, Oseledets — 2026 — [arXiv:2609.15504](https://arxiv.org/abs/2609.15504).
