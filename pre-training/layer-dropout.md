# Layer Dropout for LLM Pre-training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Stochastic depth (drop whole transformer blocks during training) has quietly disappeared from LLM pre-training recipes, but re-tuning three axes — *which* layers get dropped, the *time schedule* of the drop rate, and the *optimizer hyperparameters* — makes it a same-FLOP loss-reducer and a free enabler of early-exit, layer-skip, and self-speculative decoding at inference. Validated at 271M–8.2B parameters on up to 160B tokens.

**Prereqs:** [_training-stability.md](./_training-stability.md), [_lr-schedules.md](./_lr-schedules.md)
**Related:** [wsd-schedule.md](./wsd-schedule.md), [mtp.md](./mtp.md), [../inference/self-speculative-decoding.md](../inference/self-speculative-decoding.md)

---

## What it is

Layer dropout / stochastic depth randomly skips a subset of transformer blocks during each forward pass at training time: block $\ell$'s residual becomes an identity shortcut with probability $p_\ell$. It was standard in vision transformers and early LLM work but was quietly dropped as models scaled, mostly on anecdotal reports that it hurt final accuracy.

The paper's claim: the conventional "just apply dropout, uniformly, at some fixed rate" was the problem — not layer dropout itself. Re-tuned along three axes, it is Pareto-better than not using it.

## How it works

The three tuning axes:

**1. Layer distribution.** Which layers drop at what probability. The paper finds a triangular schedule — near-zero drop at the input and output layers, peaking in the middle — is best. Dropping input/output layers early wrecks the model's ability to see and speak; middle layers are more redundant.

**2. Time schedule.** How the drop rate evolves across training. Ramp up from 0 to the peak rate over the warmup phase, hold across the main phase, ramp back down toward the annealing/decay phase. Interacts with the LR schedule — this is why prior "uniform dropout throughout" recipes underperformed.

**3. Optimizer hyperparameters.** With layer dropout on, the effective batch size per active parameter is lower and the LR / weight-decay optima shift. The paper re-tunes learning rate and weight decay under active dropout rather than reusing the no-dropout optimum.

The forward pass:

```
for block ℓ in 0..L-1:
    if training and rand() < p_ℓ:
        h = h            ← skip block ℓ entirely
    else:
        h = block_ℓ(h) / (1 - p_ℓ)   ← rescale to keep expectation
```

Rescaling by $1/(1 - p_\ell)$ preserves the residual's expected magnitude, keeping normalization behavior stable.

## Why it matters

- **Same-FLOP loss reduction.** At the same training FLOP budget, layer-dropped runs reach lower validation loss across 271M–8.2B, 160B tokens. Same-loss target: up to **25% FLOPs saved**.
- **Free self-speculative decoding.** A layer-drop-trained model already knows how to survive shorter forward paths, so the *shallow* forward becomes a well-calibrated draft model for the *full* forward — up to **1.5× inference speedup** with negligible accuracy loss.
- **Free early-exit / layer-skipping.** The same property enables intermediate-layer decoding for latency-sensitive serving and depth-adaptive inference.
- **Reconnects two lines of research.** Prior speculative-decoding and layer-skip work assumed a pre-existing dropout-trained backbone; layer-drop pre-training is exactly what supplies one.

## Gotchas & tricks

- **Triangular distribution beats uniform.** Uniform-rate drop hurts the shallow layers whose features every downstream block depends on. Ramp the drop rate up with depth, peak in the middle, ramp back down.
- **Match LR / WD to the new regime.** Re-tuning the optimizer is not optional; reusing no-dropout defaults erases most gains.
- **Peak rate ~0.1–0.2 at LLM scale.** Higher rates (0.3+) common in ViT lore start hurting language modeling loss.
- **Scale robustness.** Findings hold across 271M–8.2B; the paper does *not* claim they hold at 100B+, where an ablation sweep gets expensive. Approach with caution above the tested range.
- **Cerebras CS-3 specifics do not transfer wholesale.** The paper is run on wafer-scale hardware; the *recipe* is architecture-agnostic but exact optimizer numbers may need re-search on GPU pods.

## Sources

- Paper: *Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference* — Elhoushi et al., Cerebras Systems — arXiv:2609.05275 (2026).
- Paper: *Deep Networks with Stochastic Depth* — Huang et al., 2016 — original stochastic depth for ResNets.
