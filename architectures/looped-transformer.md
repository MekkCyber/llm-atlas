# Looped Transformer / Parallel Loop Transformer (PLT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Re-apply the *same* transformer block $K$ times instead of stacking $K$ independent blocks. Latent depth is decoupled from parameter count; test-time compute scales with the loop count $K$. Parallel Loop Transformer (PLT) variants use cross-loop position offsets and shared-KV gated sliding-window attention to keep KV growth bounded. The 2026 LoopCoder-v2 paper diagnoses a sharp saturation at $K=2$ via a gain–cost decomposition.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md), [variable-width-transformer](variable-width-transformer.md)

---

## What it is

A transformer where the residual stream passes through a shared block (or small set of blocks) repeatedly before producing the next-token distribution. The idea has older roots (Universal Transformer, ALBERT), but its modern revival is driven by test-time-compute scaling: loop count $K$ becomes a knob you can turn at inference without retraining or growing the checkpoint.

**Parallel Loop Transformer (PLT)** is the practical variant currently winning. Instead of running $K$ sequential decoder passes (which serialize latency), PLT applies the shared block in parallel across loops with two structural additions:

- **Cross-loop position offsets (CLP).** Each loop sees the same token sequence under a shifted positional encoding so the shared block isn't asked to do exactly the same computation $K$ times.
- **Shared-KV gated sliding-window attention.** KV cache is shared across loops but gated, so per-loop KV growth is sublinear in $K$.

---

## How it works

### The forward pass

For a residual stream $x \in \mathbb{R}^{n \times d}$ and a shared block $f_\theta$:

$$
x^{(k+1)} = f_\theta(x^{(k)} + p_k), \quad k = 0, \ldots, K-1
$$

where $p_k$ is the cross-loop positional offset for loop $k$. The final $x^{(K)}$ is read out into the LM head.

PLT runs the $K$ applications in parallel (rather than sequentially across decoder steps) by aligning the KV cache via the shared-KV gate — every loop reads from the same K and V, but the gate decides which loop's residual contributes at each position.

### Gain–cost decomposition (LoopCoder-v2)

For each extra loop the model pays:
- **Gain:** the block refines the residual stream and (early loops) increases representational diversity.
- **Cost:** the CLP-induced positional mismatch between loops grows roughly linearly with $K$, and the residual stream's representational diversity saturates / oscillates.

Diagnostics on a 7B PLT trained from scratch on 18T tokens show **loop 2** delivers most of the productive refinement; loops 3+ produce diminishing, oscillatory updates that the CLP offset cost dominates.

### Choosing $K$

The paper turns the choice into a measurable trade-off rather than a hyperparameter sweep: train a small probe over a few candidate $K$, measure residual-stream diversity and representational drift per loop, pick the smallest $K$ before the gain curve flattens. For SWE-bench-style code reasoning, $K=2$ is the answer.

---

## Why it matters

- **Test-time compute, cheaply.** $K$ multiplies effective depth without multiplying parameters or KV cache (because the block weights and the KV are shared).
- **First 7B-scale recipe that beats SWE-bench-class baselines on a single repeated block.** LoopCoder-v2 lifts SWE-bench Verified from 43.0 → 64.4 and Multi-SWE from 14.0 → 31.0 by running a two-loop PLT vs. its non-looped baseline.
- **Architectural axis orthogonal to scaling laws.** Loop count adds an inference-time degree of freedom that doesn't appear in conventional Chinchilla-style depth/width planning. Useful as on-device / edge inference tries to wring more depth out of a fixed parameter budget.

---

## Gotchas & tricks

- **Loops $> 2$ regress.** Stacking more loops on top of a model trained with $K=2$ doesn't help and often hurts. The CLP cost compounds before the gain materializes.
- **CLP design matters.** A poorly chosen positional offset between loops leaves the shared block computing nearly the same residual transformation twice — no refinement, all cost.
- **Latency is *not* free.** PLT runs loops in parallel within a step, but the parallelization tax (extra attention, extra positional handling) is non-zero. End-to-end SWE-agent latency is up vs. the non-looped baseline; the win is quality per parameter.
- **Saturation point may shift with architecture / data.** $K=2$ being the sweet spot is empirically tied to LoopCoder-v2's 7B + 18T-token recipe. World-modeling variants (LoopWM) report different gain curves; they should be re-measured per stack.
- **Related but distinct from Universal Transformer.** UT loops the block until convergence with ACT (adaptive computation time). PLT fixes $K$ at design time and parallelizes; the gain–cost diagnostic is what's new in 2026.

---

## Sources

- Paper: *LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling* — Jian Yang et al., Beihang / Langboat, 2026 — [arXiv:2606.18023](https://arxiv.org/abs/2606.18023).
- Paper: *Looped World Models (LoopWM)* — Hongyuan Adam Lu et al., FaceMind Research Asia, 2026 — [arXiv:2606.18208](https://arxiv.org/abs/2606.18208) — transfers the trick to world modelling.
- Background: *Universal Transformers* — Dehghani et al., 2018 — the original block-recycling idea.
