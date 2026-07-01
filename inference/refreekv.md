# ReFreeKV
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Most KV-cache compression methods quietly depend on an input-specific budget threshold — get the threshold wrong on an out-of-distribution input and you lose quality. ReFreeKV lifts the objective from "hit accuracy at a fixed budget" to "match full-cache performance under **adaptive** budgeting," then instantiates the first threshold-free method: budget varies per input, per layer, per head, without a global cutoff.

**Prereqs:** [_kv-cache-compression.md](./_kv-cache-compression.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [README.md](./README.md), [../architectures/mla.md](../architectures/mla.md)

---

## What it is

Serving LLMs at long context is memory-bound on the KV cache. Standard compression methods (H2O, StreamingLLM, SnapKV, PyramidKV, TOVA) prune KV entries below an importance threshold — but the "right" threshold is workload-specific. A budget tuned for retrieval-heavy prompts under-serves reasoning-heavy prompts and vice versa, and open-domain traffic covers both.

ReFreeKV reformulates the problem: instead of *"keep k tokens under budget B"* it asks *"whatever budget the input actually needs to match full-cache quality — allocate that."*

## How it works

**Objective shift.** Where prior methods minimise a fixed-budget quality gap, ReFreeKV minimises a *budget-shape* discrepancy: match full-cache next-token distributions as tightly as possible with as small an effective cache as each input admits. No global threshold hyperparameter.

**Per-input adaptive allocation.**
- Score each KV entry with an importance signal derived from attention statistics (paper's specific instantiation).
- Instead of comparing scores to a global cutoff, use a *local* criterion that decides per-position whether removing it would shift the output distribution meaningfully.
- Budget is a *consequence* of the criterion, not an input to it.

**Cross-layer / cross-head shape.** Because the criterion runs independently along the layer × head axes, the effective cache shape can be non-uniform: deep layers may keep more; heads that are already sparse keep less. Prior methods (with a global threshold) impose a uniform shape.

**Empirical range.** Extensive experiments across 13 datasets covering diverse lengths, task types, and model sizes; ReFreeKV preserves full-cache quality more consistently than threshold-based baselines. Code released.

## Why it matters

- **Removes an operator-side knob.** Serving stacks (vLLM, SGLang) currently tune KV-eviction budgets per workload; ReFreeKV pushes that decision into the model.
- **Robust to open-domain traffic.** The main failure mode of threshold-based compressors is silent quality collapse on the wrong domain — ReFreeKV degrades gracefully instead.
- **Composable.** The threshold-free formulation is a *class* of methods; ReFreeKV is one instantiation. Expect follow-ups with different importance criteria under the same objective.

## Gotchas & tricks

- **Adaptive means variable cache size at runtime.** Downstream schedulers that assume a fixed per-request cache footprint need to accommodate variance. Batching becomes trickier — pad or fragment.
- **The importance criterion still has hyperparameters** — they're just not thresholds. Robustness of the method depends on how invariant those knobs are across domains.
- **Full-cache baseline is the target.** ReFreeKV is not designed to *outperform* full-cache; it's designed to match it while cutting memory. If your workload tolerates lossy compression for more speed, threshold-based methods can still be more aggressive at the low-quality end.

## Sources

- Paper: *ReFreeKV: Towards Threshold-Free KV Cache Compression* — Ni et al. (NUAA / Tencent WeChat AI / Fudan), 2026 — [arXiv:2502.16886](https://arxiv.org/abs/2502.16886).
- Code: https://github.com/Patrick-Ni/ReFreeKV
