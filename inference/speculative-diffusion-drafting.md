# Speculative-diffusion drafting
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use a **diffusion LM as the drafter** in speculative decoding, then tune two independent knobs together: the drafter's **diffusion effort** (how many denoising steps) and the verifier's **acceptance leniency** (lossless vs lossy). Prior diffusion-drafter work fixed one of the knobs; varying both jointly extends the speed-quality Pareto frontier.

**Prereqs:** [speculative-decoding.md](speculative-decoding.md)
**Related:** [README.md](README.md)

---

## What it is

A diffusion LM produces tokens in parallel, which makes it an attractive candidate for the drafter role in speculative decoding — unlike an autoregressive drafter, it can emit a whole block of candidate tokens without a serial inner loop. But diffusion LMs come with their own quality-vs-steps dial (fewer denoising steps = cheaper, blurrier output), on top of speculative decoding's standard acceptance threshold.

The paper's observation: the right design space is **two-dimensional**, not one. Varying only one knob (and holding the other at a default) explores a slice of the Pareto surface and misses the frontier.

## How it works

- **Drafter.** A diffusion LM runs for $s$ denoising steps and emits a block of $k$ candidate tokens. Lower $s$ → cheaper and lower-quality drafts.
- **Verifier.** The target autoregressive model does one parallel forward over context + candidates and applies an acceptance rule. The rule has a leniency parameter $\alpha$: $\alpha = 1$ is lossless rejection-sampling (standard), $\alpha > 1$ is lossy (accept more aggressively).
- **Jointly tune $(s, \alpha)$.** For a given target quality level, the paper finds that lowering $s$ *and* raising $\alpha$ together outperforms either knob alone — the two tradeoffs interact non-monotonically.

The result is a 2D Pareto surface in (speed, quality); the authors characterize it empirically.

## Why it matters

- **Beats single-knob diffusion drafting.** Previous diffusion-drafter work implicitly held one dimension fixed; a 2D sweep is strictly a superset.
- **Design-of-experiment template.** The analysis style — treat each component's internal knob and the overall acceptance rule as co-tunable — generalizes to any heterogeneous drafter.
- **Diffusion LMs as deployment-time citizens.** Positions diffusion LMs as a serious inference-side architecture, not just a research curiosity.

## Gotchas & tricks

- **Lossy acceptance is still lossy.** Raising $\alpha$ shifts the output distribution; verify end-to-end quality on your actual workload, not just acceptance rate.
- **Drafter cost grows with $s$.** The breakeven depends on target-model size: tiny targets rarely benefit; large targets with heavy memory-bandwidth cost benefit most.
- **Block size $k$ interacts with $s$.** The two-knob analysis treats $s$ and $\alpha$ as the primary axes; $k$ is a third one and must be swept to find the true frontier.
- **Diffusion drafter quality is the ceiling.** If the diffusion LM is far from the target's token distribution, acceptance rate stays low regardless of $(s, \alpha)$. Draft-target alignment still dominates.

## Sources

- Paper: *Speculative Diffusion is a Tale of Two Tradeoffs* — Israel, Van den Broeck, Grover (UCLA), 2026 — https://arxiv.org/abs/2610.31819
