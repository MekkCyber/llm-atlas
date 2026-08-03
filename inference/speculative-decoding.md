# Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **small draft model** proposes multiple next tokens; the **large target model** verifies them in parallel in a single forward pass. Accepted tokens count as free; rejections fall back to standard sampling. Exact variants preserve the target's sampling distribution; **lossy** variants relax verification for more speed at bounded quality cost — but silently rewrite the decoding distribution, so failure modes matter.

**Prereqs:** [README](README.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md)

---

## What it is

Autoregressive decoding is memory-bandwidth-bound: for each new token, the model reads its entire weights from HBM but only emits one token. Speculative decoding fixes this: a lightweight *draft model* (a small LM, an EAGLE/Medusa head, or a self-drafter) proposes $k$ tokens in one draft pass; the target then does *one* forward pass to score all $k+1$ positions and accepts a prefix. Every accepted token skipped a full target-model call.

Exact speculative decoding uses **rejection sampling** on the ratio $p_{\text{target}}(x) / p_{\text{draft}}(x)$ — mathematically guarantees the same sampling distribution as decoding the target alone.

## How it works

**Draft.** Small model produces $\{x_1, \ldots, x_k\}$ from prompt.

**Verify.** Target model, in one forward pass, computes $p_{\text{target}}(\cdot \mid \text{prompt}, x_{<t})$ for $t = 1 \ldots k+1$.

**Accept.** For each drafted $x_t$, sample $r \sim U(0,1)$ and accept if $r \le p_{\text{target}}(x_t) / p_{\text{draft}}(x_t)$; else stop, sample the replacement from $\max(0, p_{\text{target}} - p_{\text{draft}})$ normalized, and start a new draft.

**Speedup** = expected number of accepted tokens per verify pass. Depends on how well the draft model matches the target.

### Lossy variants

Recent methods trade exact distributional matching for higher acceptance rates. Two categories:

- **Truncation-based verification** — verify only against a truncated target distribution (top-k/top-p). Fast, but replaces the true truncation-sampling baseline with a distorted approximation.
- **Collaborative verification** — mix draft and target probabilities directly during the accept test. Fast, but if draft probs overshoot target probs, low-quality outputs slip through.

## Why it matters

- **2–4× decoding speedup** at large model sizes with quality preserved (exact).
- Standard in vLLM, SGLang, TensorRT-LLM. Backbone of most production serving stacks.
- Lossy variants promise 4–8× — but "Revisiting Lossy Verification" (Wang et al., 2026) shows many published lossy schemes reduce to two failure-prone categories and degrade under specific conditions. A required reference before flipping on a lossy verifier in production.

## Gotchas & tricks

- Draft model must be *cheap enough*: rule of thumb, target/draft cost ratio ≥ 10×.
- Long drafts have diminishing returns — acceptance drops geometrically. $k = 4$–$8$ typical.
- **Batching hurts speculative decoding**: acceptance is per-sequence, so batched verification wastes work on rejected tokens. Continuous batching + speculative needs care.
- **Lossy truncation** distorts vs. its true truncation-sampling baseline — quality degradation is silent. Diagnose with per-benchmark KL comparisons.
- **Lossy collaborative** methods need overshoot control — clip draft probabilities against target to bound the distortion.
- Same-family draft (EAGLE, Medusa self-drafters trained from the target) beat separate small LMs on acceptance rate.

## Sources

- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023.
- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., Google, 2023.
- Paper: *Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes* — Wang et al., Baidu / ZJU, 2026 — [arXiv:2607.26627](https://arxiv.org/abs/2607.26627).
