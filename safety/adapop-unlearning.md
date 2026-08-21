# AdaPop unlearning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** AdaPop is an LLM unlearning objective that replaces the standard uniform forget loss with a **popularity-weighted** one: high-frequency facts get exponentially larger penalties, low-frequency ones smaller. A **token-confidence** signal plus automatic penalty tuning bounds catastrophic-forgetting damage. Reported to substantially reduce information leakage across multiple model families and unlearning benchmarks vs uniform baselines.

**Prereqs:** [_unlearning.md](./_unlearning.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md), [../post-training/dpo.md](../post-training/dpo.md)

---

## What it is

Standard unlearning applies uniform gradient pressure to every item in the forget set $D_f$. AdaPop (Borisiuk et al., 2026) is built on the empirical observation that **popular pretraining facts are memorised more deeply and resist standard unlearning objectives longer**. A one-size-fits-all objective therefore over-forgets rare items while under-forgetting the popular ones — the opposite of what a right-to-be-forgotten pipeline wants.

## How it works

For each forget item $x_i \in D_f$:

1. **Estimate popularity** $p_i$ — the paper uses training-corpus frequency where available and a proxy (retrieval hit-count, cross-family recall) otherwise.
2. **Popularity-weighted forget loss:** raise the per-item penalty exponentially in $p_i$
   $$w_i = \exp(\alpha \cdot p_i)$$
   so a fact seen 1000× during training gets far more forgetting pressure than a fact seen once.
3. **Token confidence gate.** For each token position, gate the forget update by the model's current confidence — under-confident positions (already close to forgotten) get down-weighted, avoiding over-correction into noise.
4. **Automatic penalty tuning.** $\alpha$ and a retention weight are adjusted on-line against a held-out retention monitor; when retention drops below a threshold, the penalties shrink until it recovers.
5. **Retention loss.** A standard KL/replay term on a retain set $D_r$ runs alongside — AdaPop replaces *the forget term*, not the whole objective.

## Why it matters

- **Removes the popularity blind spot.** Under uniform losses, the most memorised facts leak first under paraphrase or membership-inference attacks. Popularity-weighted forgetting closes that specific hole.
- **Reduces catastrophic forgetting.** By spending less pressure on rare items, the retain-set damage is smaller than under a uniform loss tuned to the popular tail.
- **Practical for right-to-be-forgotten workflows.** Copyright and privacy removals are usually a mix of very-popular and very-rare items; a uniform loss handles neither well.

## Gotchas & tricks

- **Popularity estimator is load-bearing.** Where corpus frequencies are unknown, the proxy choice materially affects behaviour. Consistently over-estimating popularity means over-forgetting; under-estimating means residual leakage.
- **Exponential weighting can explode.** Cap $w_i$ or normalise the batch's weights — otherwise a single ultra-popular item dominates the gradient.
- **Doesn't help against paraphrase attacks alone.** Popularity weighting improves the *magnitude* of forgetting on the exact forget set; paraphrase generalisation still needs an augmented $D_f$ or an inference-time filter as a second line of defence.
- **Automatic penalty tuning needs a good retention monitor.** The auto-tuner's setpoint is the actual objective — if the retention set is unrepresentative, the model over-forgets to hit the number.
- **Compute is comparable to NPO.** No extra forward passes beyond the retention monitor; adds one popularity lookup per item.

## Sources

- Paper: *The More Popular, The Harder to Forget: Adaptive Popularity for LLM Unlearning* — Borisiuk, Savchenko, Panchenko, Tutubalina, AIRI / Skoltech, 2026 — [arXiv 2608.14229](https://arxiv.org/abs/2608.14229) — introduces AdaPop, the popularity-weighted forget loss, and the token-confidence gating.
