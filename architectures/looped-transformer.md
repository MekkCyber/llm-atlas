# Looped Transformer
*Depth — applying a compact tied-weight block stack repeatedly to gain effective depth at fixed parameter count.*

**TL;DR:** A **looped transformer** trains a small physical stack of blocks and applies it $L$ times at inference (and often training), reusing the same parameters at every visit. Effective depth grows without new parameters — but naïve residual scaling breaks under weight tying. **DeepLoop** derives the corrected DeepNorm exponents ($\alpha = (2N)^{1/2}$, $\beta = (8N)^{-1/2}$ for unrolled depth $N$) from a first-order perturbation bound over a *visit-alignment coefficient* $\kappa_R$.

**Prereqs:** [transformer-block.md](transformer-block.md), [_normalization.md](_normalization.md)
**Related:** [reordered-norm.md](reordered-norm.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md), [../fundamentals/z-loss.md](../fundamentals/z-loss.md)

---

## What it is

A looped (or "tied-depth", "universal") transformer has $P$ **physical** blocks that are unrolled $L$ times, giving an **effective** depth $N = P \cdot L$ with only $P$ blocks' worth of stored parameters. Training and inference both loop the same block sequence; gradients from every visit are aggregated back onto the shared block's single weight tensor.

This is attractive for on-device inference (parameter memory dominates) and for adaptive-compute architectures (loop count as a runtime knob), but the residual-scaling rules developed for untied deep networks silently break.

## How it works

### The tied-depth effect

Post-LN DeepNorm uses scaling constants $\alpha$ (residual scale) and $\beta$ (init scale) that grow with depth $N$ to keep activations and gradients stable through $N$ residual additions. In an **untied** model, each block receives an independent gradient update; DeepNorm's derivation implicitly assumes independence.

In a **looped** model, one shared update aggregates gradients from all $L$ visits and is read back by those same visits on the next forward. This tied-depth aggregation shifts the perturbation dynamics — activation deviations after one update depend on how visits' effects **align** in weight space.

### The visit-alignment coefficient

DeepLoop introduces $\kappa_R \in [0, 1]$ measuring the correlation between per-visit gradient contributions:

- $\kappa_R = 0$ ("decorrelated" regime): visits' effects average out; the classical DeepNorm exponent (1/4) is recovered.
- $\kappa_R = 1$ ("aligned" regime, conservative): visits reinforce; the exponent must rise from **1/4 to 1/2** to keep activations bounded.

### The DeepLoop scaling rule

Under the aligned regime and unrolled depth $N = P \cdot L$:

$$\alpha = (2N)^{1/2}, \qquad \beta = (8N)^{-1/2}$$

Apply these exponents to the Post-LN DeepNorm architecture; keep everything else standard. When $L = 1$ (no revisits, $N = P$), the rule collapses to the untied DeepNorm regime and training is neutral.

## Why it matters

- **Recurrent depth becomes stable.** With the corrected exponents, training a looped GPT-style model doesn't diverge as $L$ grows; validation loss keeps improving.
- **Empirically improves quality once looping is activated.** On GPT-2-small / medium scale, DeepLoop-scaled looped models beat their untied same-parameter baselines *and* their untied same-effective-depth baselines once $L > 1$.
- **Bridges universal transformers and modern deep-init recipes.** Prior universal-transformer work sidestepped DeepNorm entirely; DeepLoop makes them compatible.

## Gotchas & tricks

- **$\kappa_R = 1$ is the safe choice.** The 1/2 exponent is derived from the worst-case aligned regime; the true visit alignment is typically somewhere between 0 and 1, so DeepLoop is conservative. That is fine — over-damping is much cheaper than divergence.
- **Only Post-LN DeepNorm variants are covered.** Pre-LN, RMSNorm-based, or QK-norm variants need their own tied-depth analysis; DeepLoop's exponents don't transfer for free.
- **Loop count $L$ is a training-time knob, not just inference.** You need to train at the target $L$ (or larger) for the scaling to make sense — you cannot train at $L = 1$ and expect the model to loop-generalize.
- **Interaction with mixed precision is untested at scale.** The perturbation bound assumes clean numerics; FP8 / BF16 rounding may erode the aligned-regime safety margin.

## Sources

- Paper: *DeepLoop: Depth Scaling for Looped Transformers* — Li, Zhang, Guo, Gu, Wang — Princeton / UC, 2026.
- Paper: *DeepNet: Scaling Transformers to 1,000 Layers* — Wang et al., 2022 — the untied DeepNorm baseline DeepLoop generalizes.
- Paper: *Universal Transformers* — Dehghani et al., 2019 — the original looped/tied-depth idea.
