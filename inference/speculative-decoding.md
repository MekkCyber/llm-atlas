# Speculative Decoding
*Depth — use a cheap draft to propose multiple tokens per step, then verify in parallel with the target model.*

**TL;DR:** Autoregressive decoding is bottlenecked by sequential per-token LLM forward passes. **Speculative decoding** runs a small/fast *draft model* to produce $K$ candidate tokens, then runs a single parallel pass of the *target model* to score all $K$ positions at once; any prefix of accepted tokens is committed, the rest is rolled back to the first disagreement. Lossless (target-model distribution preserved) and typically delivers 2–3× wall-clock speedup. Introduced by Leviathan et al. (2023) and Chen et al. (2023); now standard in vLLM / SGLang / TensorRT-LLM.

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](../architectures/transformer-block.md)
**Related:** [speculative-listening](./speculative-listening.md)

---

## What it is

Three ingredients: (a) a **target model** $M$ — the big model whose distribution we must preserve; (b) a **draft model** $M_d$ — small, fast, same tokenizer, whose output we'll propose from; (c) a **verifier** that uses a rejection-sampling step to keep the sampled distribution exactly $M$'s. The point is parallelism: $M$'s cost per step is dominated by KV-cache memory I/O, not by the width of the attention operation, so scoring $K$ positions in one pass is nearly free compared to $K$ sequential passes.

## How it works

Given prefix $x_{1:t}$:

1. Sample $K$ draft tokens $\tilde y_1, \ldots, \tilde y_K$ from $M_d$ autoregressively. Costs $K$ *small* forward passes.
2. Run **one** parallel forward pass of $M$ over $x_{1:t}, \tilde y_1, \ldots, \tilde y_K$, producing target distributions $p_M(\cdot \mid x_{1:t}, \tilde y_{<i})$ for $i = 1, \ldots, K+1$.
3. Walk through $i = 1, 2, \ldots$: accept $\tilde y_i$ with probability $\min(1, p_M(\tilde y_i \mid \cdot) / p_{M_d}(\tilde y_i \mid \cdot))$. Stop on the first rejection.
4. On rejection at position $i$, resample from the **residual distribution** $(p_M - p_{M_d})_+ / \lVert (p_M - p_{M_d})_+ \rVert_1$. This correction makes the overall sampler **exactly** target-distribution.
5. Commit the accepted prefix + the one resample; repeat.

Expected speedup $\approx \alpha K / (1 + c \cdot K)$ where $\alpha$ is the token-level acceptance rate and $c$ is the draft-to-target FLOP ratio. Typical frontier models hit $\alpha \approx 0.7$, $c \approx 0.1$, $K = 4$–$8$, giving ~2–3× wall-clock.

### Draft-model variants

- **Independent draft model.** Train or reuse a small LM (e.g. a 1B drafting a 70B). Best acceptance when the draft is a distillation of the target.
- **Self-draft / Medusa.** Attach $K$ extra prediction heads to the target model itself, each predicting the token at offset $+i$. One big model; no second checkpoint; slightly lower acceptance but no drafter memory cost.
- **EAGLE / tree speculative.** The drafter predicts a *tree* of candidates, not a chain; the target verifies many paths in one pass. 3–4× wall-clock on common workloads.

## Why it matters

- **Free latency.** Lossless speedup with no quality loss. Deployable without ablations.
- **Composable with KV-cache optimizations.** Works on top of paged attention, continuous batching, prefix caching.
- **Scales with model size.** Bigger target models are *more* memory-bound, so speculative decoding helps more.
- **Base technique for a family of extensions.** Speculative listening (full-duplex speech), speculative visual-primitive decoding (VFlash), speculative tool-call decoding, etc., are all specializations of the same draft-and-verify idea.

## Gotchas & tricks

- **Acceptance rate is the bottleneck.** If the draft is miscalibrated, acceptance drops below the break-even point ($\alpha < c$) and you *lose* wall-clock. Always benchmark; never deploy blind.
- **Temperature matters.** At $T = 0$ (greedy), acceptance is a simple top-1 match and typically higher. At $T = 1$, the correction step fires more often.
- **Draft tokenizer must match target tokenizer.** A mismatch breaks the probability ratio.
- **KV cache for draft has its own cost.** If the drafter is too small to warrant a separate GPU, run it on the same device; if the drafter is large enough to be expensive, Medusa-style self-drafts are cheaper.
- **Batching + speculative.** Within a batch, different sequences accept different numbers of tokens per step, which complicates continuous batching. Modern servers handle this by scheduling "partial" accepts; expect some throughput cost at high batch sizes.
- **Verification is stateless on the accepted prefix.** Only the committed KV extends — rejected positions are discarded. Make sure your KV-management code handles rollback cleanly.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias — Google, 2023.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al. — DeepMind, 2023 (concurrent with Leviathan).
- Paper: *Medusa* — Cai et al., 2024 — multi-head self-drafting.
- Paper: *EAGLE / EAGLE-2* — Li et al., 2024 — tree-structured speculative decoding.
- Blog: Hugging Face *Assisted Generation* writeup, vLLM / SGLang / TensorRT-LLM docs.
