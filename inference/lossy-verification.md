# Lossy Verification in Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard [speculative decoding](_speculative-decoding.md) is **lossless** — its rejection-sampling acceptance rule reproduces the target's exact next-token distribution. **Lossy verification** relaxes that constraint to accept more draft tokens per step, buying throughput at the cost of distributional drift. Wang et al. (2026) categorize the family into **truncation-based** and **collaborative** paradigms, name the mechanism that causes quality loss in each, and give the design condition needed to keep the loss controlled.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** [mtp](../pre-training/mtp.md)

---

## What it is

Lossless speculative decoding uses rejection sampling: a draft token is accepted with probability $\min(1, p_\text{target}/p_\text{draft})$, and rejected tokens are resampled from a correction distribution. This exactly reconstructs the target's distribution — but the acceptance rate is capped by how well the draft matches the target.

**Lossy verification** relaxes the acceptance rule to accept more tokens. Two families:

- **Truncation-based:** compare against a truncated (top-$k$ / nucleus) target distribution rather than the full one. Accept draft samples that fall inside the truncated set.
- **Collaborative:** the draft and target jointly participate in the acceptance rule (e.g., a shared threshold, a hybrid distribution).

Both are used in production serving stacks because they push throughput past the lossless ceiling.

## How it works

**Truncation-based failure mode.** Naively verifying against a truncated target *does not* recover the true truncated-sampling distribution — the accepted-sample marginal can be sharply distorted, especially when the draft assigns non-negligible mass outside the target's top-$k$ set. The failure is silent because on-average metrics (perplexity, benchmark accuracy) shift little; the damage shows up in tail quality (rare tokens, code correctness, factuality).

**Collaborative failure mode.** In collaborative schemes where draft and target contribute to an acceptance score, quality collapses when the **draft assigns much higher probability than the target** for some tokens — call this an *overshoot*. Overshoot lets clearly-wrong tokens survive verification if the draft is over-confident. The paper's design principle: **explicitly control $\max_t p_\text{draft}(t) / p_\text{target}(t)$**, either by capping the ratio or by re-weighting the acceptance rule.

The paper's contribution is *characterization* — a mechanism-first taxonomy of what breaks, not a new benchmark-topping scheme. Empirical validation follows the analysis.

## Why it matters

Lossy verification is already in production because the throughput uplift is large. But the failure modes are:

- **Silent in aggregate.** Perplexity and benchmark averages barely move.
- **Amplified in the long tail.** Rare-token quality, structured output correctness, and multi-step reasoning degrade.
- **Interacting with sampling parameters.** A truncated-lossy scheme paired with top-$k$ sampling can double-truncate, distorting further.

Naming the failure mechanism turns "our tail quality dropped after enabling speculative decoding" from a mystery into a design bug with a known fix (control the ratio; use the correct truncated-sampling baseline).

## Gotchas & tricks

- **Benchmark it on the tail.** Aggregate perplexity is a bad detector; use structured-output correctness, math/code eval, or long-tail token entropy.
- **Draft-target overshoot** is worse for over-confident drafts (small distilled models, MTP heads trained with too much on-policy data).
- Truncation-based lossy verification does *not* commute with sampling truncation at inference — pick one truncation, not both.
- For collaborative schemes, a hard cap on $p_\text{draft}/p_\text{target}$ is the simplest mitigation; the paper suggests it as the necessary condition.

## Sources

- Paper: *Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes* — Wang, Zhou, Wang, Li, Xiao, Shang, 2026 — [arXiv:2607.26627](https://arxiv.org/abs/2607.26627)
- Related lossless baseline: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023.
