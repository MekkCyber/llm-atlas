# SkewAdam (tiered optimizer state for MoE)
*Depth — an AdamW variant that assigns different optimizer state to backbone, experts, and router.*

**TL;DR:** Optimizer state is the largest single line item in MoE training memory: AdamW on a 6.78B MoE keeps 50.6 GB of first + second moments to update 12.6 GB of bf16 weights. SkewAdam observes that the **backbone** (~5% of params), the **experts** (~95%), and the **router** (<0.01%) have such different sizes and gradient statistics that they should not carry the same state. It tiers them — full momentum + factored second moment on the backbone, factored-only on experts, exact on the router — and drops peak training memory from 81.4 GB to 31.3 GB.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md)
**Related:** [_training-stability.md](./_training-stability.md), [fp8-training.md](./fp8-training.md)

---

## What it is

An MoE parameter matrix hides three populations with radically different roles:

- **Backbone** (attention, dense projections outside the MoE block) — small in count, dense signal per step, benefits most from momentum.
- **Experts** — huge in count, sparsely activated per step, second-moment shape matters more than momentum trace.
- **Router** — tiny (<0.01%), high-signal-to-noise, gets exact state basically for free.

SkewAdam matches optimizer state to those roles instead of paying the same AdamW cost on all of them.

## How it works

Per-tier state:

| Tier | % of params | First moment (momentum) | Second moment |
| --- | --- | --- | --- |
| Backbone | ~5% | full **float32** | factored |
| Experts | ~95% | **none** | factored |
| Router | <0.01% | full | exact |

The factored second moment (à la Adafactor) approximates `v_t` as an outer product of a row-vector and column-vector, cutting memory from `numel(W)` to `rows(W) + cols(W)`. Momentum is kept in float32 on the backbone specifically because that's where it matters — the ablation shows momentum, not the tiering, earns the perplexity gain.

## Why it matters

- **Optimizer state falls from 50.6 GB to 1.29 GB** (2.6% of AdamW) at bf16 weights, on a 6.78B MoE.
- **Peak training memory: 81.4 GB → 31.3 GB** — inside a **40 GB accelerator** budget.
- **Val perplexity (82M-token controlled comparison):** SkewAdam **108.4** vs AdamW **126.8** vs Muon **120.2** vs Lion **393.7**.
- **Tuning-robust.** Sweeping baseline learning rates narrows but does not close the gap: best-tuned AdamW = 118.5, best-tuned Adafactor = 139.7.
- **Router load balance** settles within 1% of the uniform floor.

Together: a strong argument that *where* optimizer state lives matters at least as much as *how much* of it there is.

## Gotchas & tricks

- The tier ablation is the paper's most important control: with 20× the state, the tiers match — proving the gains come from *keeping momentum*, not from the allocation being magical. Don't sell SkewAdam as a compression trick.
- Adafactor-with-momentum ("shampoo-adjacent") would probably narrow the gap further; the comparison in the paper uses vanilla Adafactor.
- Applies specifically to sparse MoE with clean population separation. Dense models don't have three obvious tiers.

## Sources

- Paper: *Where Should Optimizer State Live? Tiered State Allocation for Memory-Efficient Mixture-of-Experts Training* — anonymous single author, 2026 — [arXiv:2607.19058](https://arxiv.org/abs/2607.19058)
