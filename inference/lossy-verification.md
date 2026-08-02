# Lossy Verification in Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** "Lossy" speculative-decoding variants relax the exact rejection-sampling test to accept more draft tokens — trading a tighter acceptance rate for a *silent* rewrite of the output distribution. Two families exist: **truncation-based** (restrict acceptance to a truncated distribution) and **collaborative** (blend draft and target probabilities). Both have specific failure modes that can crater generation quality unnoticed.

**Prereqs:** [speculative-decoding](speculative-decoding.md)
**Related:** [mtp](../pre-training/mtp.md)

---

## What it is

Vanilla [speculative decoding](speculative-decoding.md) preserves the exact target distribution via a per-token rejection test. That test caps acceptance at $\min(1, p/q)$, which is often the bottleneck on wall-clock speedup. **Lossy verification** loosens this test to accept a larger fraction of drafts, buying throughput at the cost of output-distribution fidelity. The distribution shift is often invisible on standard benchmarks but shows up as quality regressions on longer or harder generations.

## How it works

Wang et al. (2026) classify most "novel" lossy schemes into two categories:

### Truncation-based verification

Restrict the acceptance test to a truncated distribution (e.g. top-$k$ or top-$p$ of the target). Accept if the draft token is inside the truncated support and passes a weakened ratio test; else resample from the truncated distribution.

**Silent pitfall:** the induced marginal is *not* the same as sampling from the target's truncated distribution directly. Overshoot terms distort the tail, so the lossy-SD generator degrades below the plain truncation-sampling baseline at the same acceptance rate — a hidden regression rather than the "at worst equal" behavior practitioners assume.

### Collaborative verification

Blend draft and target probabilities into a mixture and accept from that mixture (variants: SpS-style, coupled draft-target sampling). Accepts more, distributes the "loss" between the two models.

**Silent pitfall:** quality collapses when the **overshoot** — the amount by which $q(x)$ exceeds $p(x)$ — is unbounded. Bounding overshoot (e.g. via a clip on $q/p$) is the missing ingredient in most published collaborative schemes.

## Why it matters

Lossy SD is showing up in production serving stacks as a "free speedup." This paper shows the speedup is silently paid for in generation quality, especially on long outputs where distributional drift compounds. It also gives the community two operational levers:

- For truncation-based methods: compare against the *true truncation sampling baseline*, not against vanilla SD.
- For collaborative methods: measure and bound draft-vs-target overshoot before deployment.

## Gotchas & tricks

- **Standard benchmarks under-report the loss.** Perplexity and short-form Q&A rarely show the regression; long-form generation, chain-of-thought, and coding are where quality cracks.
- **Acceptance rate ≠ quality.** A method that accepts 95% of drafts by rewriting the distribution isn't better than one that accepts 60% at exact SD.
- **"Overshoot" is the killer.** In collaborative SD, if $q(x)/p(x)$ can go arbitrarily high, single tokens with vanishing target probability get emitted with near-draft frequency.
- **Diagnostic first.** Before shipping any lossy variant, run the paper's diagnostic protocol against the exact-SD baseline on long generations. Regression there is disqualifying.

## Sources

- Paper: *Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes* — Wang, Zhou, Wang, Li, Xiao, Shang, 2026 — [arXiv:2607.26627](https://arxiv.org/abs/2607.26627).
- Code: [github.com/ZhouYuxuanYX/Fast-HSD](https://github.com/ZhouYuxuanYX/Fast-HSD).
