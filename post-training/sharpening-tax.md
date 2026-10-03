# Sharpening Tax
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A diagnostic metric that quantifies the *coverage* a model loses to RL post-training. Compares pass@1 and pass@K between a base model and its post-trained descendant on the same task; the normalized shortfall is the "tax" the sharpening pays. Introduced to show that on agentic benchmarks, pre-trained bases paired with a light inference harness often *beat* their post-trained counterparts at pass@K under a sufficient test-time budget.

**Prereqs:** [../evaluation/pass-at-k.md](../evaluation/pass-at-k.md), [grpo.md](grpo.md), [_post-training.md](_post-training.md)
**Related:** [ptgs.md](ptgs.md) · [_rl.md](_rl.md) · [rl-prompt-curation.md](rl-prompt-curation.md)

---

## What it is

RL post-training on verifiable rewards tends to *sharpen* the policy: pass@1 goes up, but the output distribution concentrates. For tasks where retry is cheap (agents, code, math under test-time compute), pass@K matters as much as pass@1 — and the post-trained model's pass@K can be worse than the base's.

The Sharpening Tax (Oh et al., 2026) formalizes this as a scalar diagnostic computable from a modest number of rollouts per prompt. The core finding: across 14 base/post-trained model pairs (four families) on three agentic benchmarks (42 cases), the tax is *prevalent* — most post-trained models pay it on most agentic settings — and *estimable* from a handful of rollouts per prompt.

## How it works

Given a base model $\pi_{\text{base}}$ and a post-trained descendant $\pi_{\text{post}}$ evaluated on the same prompt set with the same verifier:

1. Sample $n$ rollouts per prompt from each model at a fixed temperature.
2. Estimate pass@1 and pass@K for each.
3. Define the per-prompt tax as a signed, bounded shortfall of post-trained pass@K relative to base pass@K, normalized so that: tax > 0 means post-trained has strictly less coverage despite higher pass@1; tax ≈ 0 means post-training paid no coverage cost; tax < 0 means post-training improved both single-shot *and* coverage (uncommon).
4. Average over prompts for a per-benchmark tax.

The paper shows the tax is highly correlated with other dispersion metrics (entropy, mode concentration), so a few rollouts suffice.

## Why it matters

- **Honest accounting of RL costs.** Field benchmarks almost exclusively report pass@1, so this cost has been invisible.
- **Motivates coverage-aware recipes.** If pass@K is the right metric for your deployment (agents that retry, best-of-N inference), then the right objective is not pure RL on single rollouts.
- **Direct lever: PTGS.** The same paper introduces [ptgs.md](ptgs.md) (posterior-tempered group sampling) which lowers the tax relative to fixed-temperature RL in two agentic environments while also *improving* pass@1.

## Gotchas & tricks

- Tax depends on $K$; report $K$ with the number. $K$ in the deployment-realistic range (5–32 for agents, 1–8 for interactive use) is most informative.
- Comparisons require the *same* verifier and the same sampling temperature; otherwise differences conflate with sampling-setup noise.
- "Tax ≈ 0" on an agentic task is not a sign of good post-training; it may just mean the base is already weak and had little coverage to lose.
- The metric assumes retries are independent; agentic settings with shared tool state (sticky side effects, cached logs) break this and the tax underestimates true coverage loss.

## Sources

- Paper: *Sharpening Tax in Post-Training* — Oh, Zeng, Qi, Zhmoginov, Lei, He, Phan, Kang, Mirhoseini, Li; U. Wisconsin–Madison / Stanford / Meta, 2026 — [arXiv:2610.01509](https://arxiv.org/abs/2610.01509).
- Related: *Large Language Monkeys* (Brown et al., 2024) — base-model coverage scaling that motivates coverage-aware evaluation.
