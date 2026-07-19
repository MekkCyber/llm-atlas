# Looped Transformer
*Depth — apply a compact stack of physical blocks for multiple rounds to scale computation without growing parameter count.*

**TL;DR:** Instead of `L` distinct blocks stacked once, use `L' < L` shared blocks looped `K` times so the *unrolled* depth is `L' × K`. Parameters stay at `L'`; FLOPs and inference latency scale with the loop count `K`. Trades storage for compute — attractive for edge deployment, quantization, and any regime where parameter count is the binding constraint. Requires care with residual scaling because one shared update aggregates gradients from every visit.

**Prereqs:** [transformer-block](transformer-block.md)
**Related:** [../pre-training/_training-stability](../pre-training/_training-stability.md), [../fundamentals/z-loss](../fundamentals/z-loss.md)

---

## What it is

A standard Transformer applies `L` distinct blocks `f_1, f_2, …, f_L`. A looped Transformer applies a single (or a few) block(s) `f` for `K` rounds:

$$
h_{t+1} = f(h_t), \quad t = 0, \ldots, K-1
$$

Different from *weight sharing* schemes like ALBERT (which tie all layers but still forward through them one time each) in one important way: looping is a *runtime* choice — `K` can be varied at inference to trade quality for latency.

## How it works

### The residual-scaling problem

A Transformer's residual pathway aggregates block outputs:

$$
x_L = x_0 + \sum_{\ell=1}^{L} \beta \cdot \mathrm{sublayer}_\ell(\alpha \cdot x_{\ell-1})
$$

For an untied Transformer each `sublayer_ℓ` has its own gradient and its own place in the sum. For a looped Transformer, one shared block gets gradient contributions from every visit and reads back those aggregated updates on the next forward. The effective learning-rate seen by the block scales with `K`, and the residual-branch magnitudes accumulate differently than in an untied stack.

### DeepLoop's α/β correction

DeepLoop (Li et al., 2026) rescales `α` (residual multiplier) and `β` (sublayer multiplier) so a looped block behaves as if it were `K` distinct blocks at gradient scale. Concretely, they derive per-loop-count corrections that keep the variance of the residual stream and the update-to-parameter ratio on the same trajectory as an untied `K`-block model. Without the correction, deeper loops either blow up (residual variance grows superlinearly) or saturate (update-to-parameter ratio underflows).

## Why it matters

- **Parameter-tied depth scaling.** For a fixed memory budget, looping trades storage for extra sequential compute — especially useful for on-device inference and quantized deployment where weights dominate footprint.
- **Adaptive compute at inference.** `K` can be tuned per input, giving a lever for early-exit-style test-time scaling without training a separate small model.
- **Aligned with test-time compute.** As test-time compute scaling becomes routine (reasoning models, long CoT), architectures that let you buy extra sequential compute cheaply are increasingly attractive.

## Gotchas & tricks

- **Fixed-point drift.** With enough loops, `f` can drift toward its fixed point and updates stop being informative. Practitioners cap `K` (typically 4–8×) or inject learned per-loop conditioning.
- **KV cache reuse across loops.** Attention KV can be shared across loop iterations to save memory, but the effective attention pattern changes; watch for stability at long context.
- **Not a free win at large parameter counts.** Above frontier scale, untied depth still dominates on quality — looping is most attractive in the 1–7B range where storage is the binding constraint.
- **Distillation from untied models.** Looped Transformers distill reasonably well from untied teachers of the same unrolled depth; the reverse (untied from looped) is rarely useful.

## Sources

- Paper: *DeepLoop: Depth Scaling for Looped Transformers* — Li, Zhang, Guo, Gu, Wang, 2026 — the α/β residual-scaling correction.
- Related: earlier weight-tying work (ALBERT, Lan et al. 2019) as a special case (fixed `K=1` per layer, no adaptive compute).
