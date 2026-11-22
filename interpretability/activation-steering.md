# Activation Steering

*Depth — add a fixed vector to a layer's residual stream to shift model behavior along a chosen direction.*

**TL;DR:** Pick a direction `v` in hidden state, add `α · v` to every token's residual at a chosen layer, and the model's output distribution shifts in a semantically meaningful way: more honest, less refusal, more "unknown"-abstention, more sycophantic. The vector is cheap to fit (mean difference between contrasting prompt sets, or a probe's normal), the intervention is cheap to deploy (one vector add), and it composes additively. The caveat is that *the best steering direction is not the best probing direction* — the two objectives disagree, and transfer across models is more reliable for probing.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

An inference-time intervention that modifies behavior without weight updates. Positioned between prompting (behavior shift without any numerical knob) and fine-tuning (behavior shift via gradients on full parameters). The steering vector is often interpretable — the authors can name *what* the direction represents — which is where the interpretability framing comes from.

Two canonical constructions:

- **Mean-difference vector.** `v = mean(h | positive prompt) − mean(h | negative prompt)` at some layer. Cheapest; interpretable.
- **Linear-probe normal.** Train a classifier on hidden states for a binary attribute; the probe's normal vector is the direction.

At inference: `h' = h + α · v` for every token (or every token after a chosen position), at the chosen layer. The scalar `α` is calibrated on a held-out set.

## How it works

Three choices determine quality:

1. **Where to intervene.** Early layers shift representations; late layers shift outputs. Middle layers are the usual best for behavior changes.
2. **Direction family.** A family interpolates between the probe normal and the mean-difference vector. Probing favors directions that *classify* well; steering favors directions that, when *added*, move the output distribution predictably. These two objectives genuinely disagree (see [Probing vs Steering Directions], 2026).
3. **Strength `α`.** Below a threshold, behavior barely moves; past a cliff, it collapses to incoherent output. Tune on a dev set with both the target attribute and a coherence metric.

Recent striking finding from diffusion: for RL-tuned diffusion transformers, a small set of steering vectors added to a *frozen* base recovers 90–112% of the reward gains — "RL post-training" and "activation steering" become almost the same intervention in that setting ([RL Post-Training in DiTs Is Activation Steering], 2026).

## Why it matters

A unit of behavior change that is small (one vector per intervention), composable (vectors add), cheap to fit, and interpretable. Three use cases:

- **Safety retrofits** on deployed models without retraining.
- **A/B testing concepts** — fit a direction, add it, see what shifts.
- **Replacement for weight updates** in constrained-budget settings. Where RL updates work, a steering vector often matches 90+% of the gain at a tiny fraction of the compute.

## Gotchas & tricks

- **Probe ≠ steerer.** The 2026 Fraunhofer IAIS paper showed that on truthfulness and sentiment, the best probing direction is *not* the best steering direction across three instruction-tuned models. Fit the direction for the intervention you actually want to deploy.
- **Mean-difference transfers better than probe normals** across models for probing use; the opposite holds less reliably for steering.
- **Coherence cliff.** Pushing `α` past a model-specific threshold yields degenerate text. Always dev-set tune.
- **Direction specificity.** A "truthfulness" direction can bleed into refusal, formality, or length. Measure side-effects.
- **Inverse-steering corollary.** Negative `α` suppresses the attribute — useful for red-teaming (suppress refusal, test downstream safety).

## Sources

- Paper: *Steering Language Models with Activation Engineering* — Turner et al., 2023 — the first clean mean-difference steering recipe.
- Paper: *Representation Engineering* — Zou et al., 2023 — framing + probe-normal construction.
- Paper: *Do Probing and Steering Favor the Same Directions Within and Across Language Models?* — Sadeghi & Becker (Fraunhofer IAIS), 2026 — [arXiv:2611.19499](https://arxiv.org/abs/2611.19499) — the probing/steering direction gap.
- Paper: *RL Post-Training in Diffusion Transformers Is Activation Steering* — Oktay, Helbling, Kara, Chau, Yanardag, 2026 — [arXiv:2611.26627](https://arxiv.org/abs/2611.26627) — steering vectors reproduce diffusion RL reward gains.
