# SA-OPD: Spurious-Signal-Aware On-Policy Distillation

*Depth — a token-level filter that removes teacher supervision driven by language priors rather than task evidence.*

**TL;DR:** On-Policy Distillation (OPD) gives dense per-token teacher signal, but some of that signal comes from *input-agnostic* language priors, formatting conventions, or stereotyped reasoning templates — high-gradient updates that don't improve task behavior. SA-OPD (Jiang et al., 2026) proposes a lightweight **input-groundedness proxy** to estimate whether a token-level signal actually depends on the input, then filters only the tokens that are simultaneously low-groundedness *and* high-divergence. Outperforms vanilla OPD and prior selective-OPD baselines on both LLM and VLM settings.

**Prereqs:** [on-policy-distillation.md](on-policy-distillation.md), [grpo.md](grpo.md)
**Related:** [rstg.md](rstg.md) · [rejection-sampling.md](rejection-sampling.md) · [_post-training.md](_post-training.md)

---

## What it is

A failure mode of OPD: the teacher's per-token distribution is a mix of (a) task-specific evidence-driven predictions and (b) prior-driven fillers (common phrases, formatting, template continuations). Naive OPD weights both equally by divergence magnitude, so a large gradient on "the answer is" can outweigh a small gradient on the actually informative token that follows. SA-OPD identifies and removes exactly the prior-driven, high-impact tokens.

## How it works

Two axes of per-token judgment:

1. **Input-groundedness proxy.** Estimate whether the teacher's prediction at token `t` depends on the input. Typical proxy: compare teacher's distribution given the full context vs. given a masked or dropped-input context. Large distributional change → high groundedness; near-identical → low groundedness (the teacher would say the same thing regardless of the input).
2. **Divergence magnitude.** Standard `KL(π_teacher || π_student)` at token `t`. Large values are the ones that produce large gradients.

The filter rejects tokens for which:

- input-groundedness is low (below a percentile threshold), **and**
- divergence is high (above a percentile threshold).

Both conditions must hold. Tokens with high groundedness *and* high divergence — the informative task-relevant updates — are preserved. Tokens with low groundedness *and* low divergence are also preserved (they contribute little either way). Only the "high-impact spurious" quadrant is dropped.

## Why it matters

- **First axis-based selection for OPD.** Prior selective-OPD methods filter by confidence or learnability alone; SA-OPD adds a *causality-flavored* axis (does the input matter?) that catches a distinct failure class.
- **Improves both LLM and VLM distillation** — the paper reports gains in both settings, so the mechanism isn't multimodal-specific.
- **Cheap to add.** The input-groundedness proxy is a single extra teacher forward pass with a modified context per token batch; small overhead on the already teacher-heavy OPD budget.

## Gotchas & tricks

- **Proxy design is the whole game.** "Mask the input and re-query" is one option; embedding-based or gradient-based proxies also work. Each has failure modes on templated inputs (where the template itself is the input).
- **Percentile thresholds are hyperparameters** and dataset-dependent. Cross-domain transfer of thresholds isn't studied.
- **Doesn't fix the "teacher is wrong" case.** SA-OPD assumes the teacher is right when grounded — a mistaken teacher that *is* input-driven will still teach the student its mistake.
- **Composes with RSTG.** SA-OPD (token-level) + RSTG (sample-level) can be stacked: RSTG picks *which prompts* to distill, SA-OPD picks *which tokens within them*.

## Sources

- Paper: *When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation* — Jiang, Ye, Tao, Zhuang, Zhang, Chen, Li, 2026 — [arXiv 2608.03632](https://arxiv.org/abs/2608.03632).
