# Activation Steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Add a fixed vector $v$ to hidden states at one or more layers to push the model's output toward (or away from) a target concept — refusal, honesty, a sentiment, a persona. No weight updates, no new parameters beyond the vector itself. The vector is found by contrasting activations on paired "concept vs no-concept" prompts, or by online search against a generation-level objective. Also emerging as a cheap substitute for RL post-training in diffusion transformers (DiT).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [linear-probes.md](linear-probes.md), [self-editing-llms.md](self-editing-llms.md), [circuit-compilation.md](circuit-compilation.md), [../safety/refusal-suppression.md](../safety/refusal-suppression.md)

---

## What it is

A test-time intervention: intercept the residual stream at layer $\ell$ and replace $h_\ell \leftarrow h_\ell + \alpha \cdot v$ (optionally subtract to suppress). $v$ lives in the model's hidden-state space and is chosen so that this shift changes a chosen output property — refusal/compliance, truthfulness, sentiment, persona — while leaving the rest of the response close to baseline. In diffusion transformers the same trick acts on block *outputs*, with $v$ chosen to replicate the effect of RL finetuning.

## How it works

Three common ways to find $v$:

1. **Difference-of-means (ActAdd / MD-pool).** Collect paired activations $h^+$ (concept present) and $h^-$ (concept absent) at layer $\ell$ over a probe set; take $v = \mathrm{mean}(h^+) - \mathrm{mean}(h^-)$. Cheap, surprisingly strong baseline.
2. **Learned probe direction.** Train a linear probe on the pair set and use its weight vector as $v$. Not the same as (1) — see [linear-probes.md](linear-probes.md).
3. **Online fit against a generation objective.** Optimise $v$ to maximise a reward / minimise a loss on sampled generations. In DiTs: fit $v$ to match the velocity predictions of an RL-tuned model. In LLMs: fit against reward-model scores or a KL target.

Apply at inference by hooking the residual stream (or a specific block's output) and adding $\alpha v$ for all tokens or only a prefix. $\alpha$ sweeps between weak and runaway.

## Why it matters

- **No weight updates.** Swap $v$ at will; A/B test behaviours without retraining.
- **Cheap substitute for fine-tuning in narrow regimes.** For diffusion transformers, steering vectors fit to RL-tuned targets recover 90–112% of the reward gain; trained with online RL directly on $v$, steering beats weight-RL on SD3.5-M benchmarks.
- **Compositional.** Multiple vectors can stack (steer toward honesty *and* away from politics) with predictable, roughly linear behaviour in the small-$\alpha$ regime.
- **Interpretability lever.** If a single direction drives a behaviour, that is evidence the behaviour is represented linearly in the residual stream.

## Gotchas & tricks

- **Probe ≠ steer.** The direction that *detects* a concept best is usually not the one that *steers* best — steering favours directions closer to mean-difference, not the trained-probe weight vector (Sadeghi & Becker, 2026).
- **Layer matters.** Mid-layers typically steer best; early layers are too generic, late layers are too committed. Sweep.
- **Cross-model transfer is weak.** Steering directions mapped linearly to another model keep much less of their effect than probing directions do.
- **Large $\alpha$ breaks fluency.** Monotonic improvement only holds in a small window; past that the model emits off-distribution or repetitive text.
- **Safety caveat.** The same vector that steers toward refusal can be negated to suppress refusal. Vectors are not an access-controlled intervention — treat them as capabilities, not controls.

## Sources

- Paper: *Activation Addition: Steering Language Models Without Optimization* — Turner et al., 2023 — the ActAdd / MD-pool recipe.
- Paper: *Steering Llama 2 via Contrastive Activation Addition* — Rimsky et al., 2023 — large-scale CAA on refusal/persona.
- Paper: *RL Post-Training in Diffusion Transformers Is Activation Steering* — Oktay et al., 2026 — steering-only recovers 90–112% of DiT RL gains; online-RL on $v$ beats weight-RL.
- Paper: *Do Probing and Steering Favor the Same Directions Within and Across Language Models?* — Sadeghi & Becker, 2026 — the probe≠steer gap, measured.
