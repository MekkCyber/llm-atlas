# Diffusion-Drafted Speculative Decoding (DScout)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative decoding with a *diffusion* drafter instead of an autoregressive one. Diffusion drafters propose many tokens in parallel via masked / discrete diffusion, which raises draft length — but naive adoption doesn't help because verification accepts only a consecutive prefix and per-round latency rises. DScout fixes this with two **factorisations**: a training-side split of position-wise loss into *inclusion*, *weighting*, *prediction*, and an inference-side split of throughput into *(mean accepted tokens) × (1 / per-round latency)*. End-to-end: 3.84× (Qwen3-4B) and 4.14× (Qwen3-8B) over AR decoding across 6 benches, +7–9% over the prior AR-drafter SOTA.

**Prereqs:** [speculative-decoding](speculative-decoding.md), [mtp](../pre-training/mtp.md)
**Related:** [attention](../fundamentals/attention.md)

---

## What it is

Autoregressive drafters (Medusa, EAGLE) emit one token at a time. Mask-based diffusion drafters emit *all $k$ tokens in parallel*, which lifts the ceiling on draft length but exposes a new problem: the verifier only accepts a *consecutive accepted prefix*, so a diffusion drafter that is 95% accurate at each position still only cashes in a short prefix on average. DScout re-engineers both the drafter's training objective and the inference-time hyper-tuning to maximise *throughput*, not acceptance count.

## How it works

**Training factorisation.** The position-wise cross-entropy loss is decomposed into three roles that previously were tangled:

1. **Inclusion** — a binary term that decides whether a given position should be drafted at all.
2. **Weighting** — a scalar per position encoding how much the loss cares about getting it right.
3. **Prediction** — the standard token-level cross-entropy, conditional on inclusion.

This decomposition guides *mixed-horizon training* (train on short and long drafts jointly), *intra-block unmasking* (schedule which tokens inside the drafted block to unmask first), and *acceptance-weighted loss* (down-weight positions the verifier rarely accepts anyway).

**Inference factorisation.** Throughput $\mathrm{T}$ is split as:

$$
\mathrm{T} = \frac{\text{mean accepted tokens per round}}{\text{per-round latency}}
$$

The paper's configuration policy picks three knobs — target-conditioning depth, drafter depth, and block size $k$ — by estimating both factors from a short calibration pass, then selecting the Pareto-optimal config. The upshot is that *more aggressive drafts aren't always better*; the tuning policy adds up to +9.75% speedup over DScout's own fixed configuration.

## Why it matters

- **First diffusion drafter to clearly beat AR drafters end-to-end.** The naive version loses even against EAGLE; the factorised version is +7–9% over the prior SOTA.
- **Decouples draft length from throughput.** The factorisation surfaces that the quantity to optimise is throughput, not draft length — a reframing that applies to any multi-token drafter.
- **Opens a non-AR drafter axis.** Together with N-gram and lookup drafters, diffusion-based speculation points at a serving stack where drafters are *task-conditioned*: diffusion for high-entropy generation, N-gram for boilerplate, EAGLE for the middle.

## Gotchas & tricks

- **Masking schedule matters more than denoising depth.** Intra-block unmasking order (confident-first vs left-to-right) changes acceptance significantly; left-to-right is a weak default.
- **Target-conditioning depth is a Pareto knob.** Deeper conditioning raises acceptance but raises per-round latency more. The paper's policy picks it per batch size.
- **Block size $k$ scales with context entropy.** Low-entropy prefixes (code, boilerplate) accept $k = 16+$; high-entropy prefixes (creative text) cap near $k = 4$.
- **Exact-equivalence not addressed.** The paper focuses on throughput; whether the resampling-on-rejection correction preserves target distribution exactly for diffusion drafters is left as a practical choice.
- **Not a free lunch for small targets.** The win concentrates at 4B+ because the forward-pass fixed cost of the target has to dominate drafter cost.

## Sources

- Paper: *Factorization-Guided Training and Inference for Diffusion-Drafted Speculative Decoding* — Fan, Xiao, Wang, Ling, Liu, Jia, Nikolopoulos, Wei (ByteDance), 2026 — [arXiv:2610.23935](https://arxiv.org/abs/2610.23935).
- Background: [speculative-decoding](speculative-decoding.md) for the core framework.
