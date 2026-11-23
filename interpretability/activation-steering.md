# Activation Steering
*Depth — modify a frozen model's behavior by adding fixed vectors to its activations at inference.*

**TL;DR:** Instead of fine-tuning, inject a small learned vector into the model's residual stream at inference. Sum it with the activations at one or more layers, and the model's behavior shifts along the direction that vector encodes (more truthful, less refusing, a different style). No weight changes, no reward model at inference, no retraining — the vector is the intervention.

**Prereqs:** [README](README.md)
**Related:** [probing-vs-steering](probing-vs-steering.md) · [refusal-suppression](../safety/refusal-suppression.md) · [_rl](../post-training/_rl.md)

---

## What it is

A behavioral edit applied at inference time by **adding a vector** $v$ to the model's hidden state $h_\ell$ at layer $\ell$:

$$
h_\ell' = h_\ell + \alpha \cdot v
$$

The vector $v$ is a direction in activation space, fit once; $\alpha$ scales the intervention at inference time. Steering targets an attribute — truthfulness, sentiment, refusal, "style of helpful assistant" — rather than a specific output. It's the behavioral-edit analogue of model surgery: cheap, additive, composable.

## How it works

Three steps:

1. **Choose a target behavior** and collect **contrast pairs**: prompts/activations where the behavior is on vs off (truthful vs false completions; harmful vs refused requests).
2. **Compute a direction** that separates them. Simplest: *mean-difference* $v = \bar{h}_+ - \bar{h}_-$ at some layer $\ell$. Variants include CAA (contrast activations), ITI (truth directions from probe weights), and reward-matching (fit $v$ to approximate an RL-tuned model's velocity — see "RL as steering" below).
3. **Apply at inference**: pick $(\ell, \alpha)$, add $\alpha v$ into the residual stream at every forward pass. Multi-layer steering sums vectors at several sites.

Two important design choices: **where** (layer / which token positions — sometimes all, sometimes only the last) and **how strong** ($\alpha$ can be swept; too large and the model degenerates).

A striking recent result: **policy-gradient RL post-training on diffusion transformers can be replaced, at most of its gain, by a small set of fixed steering vectors fit to match the RL-tuned model's velocity predictions**. This gives a reward-free, reward-model-free route to recovering the bulk of RLHF/GRPO gains with one supervised regression.

## Why it matters

- **Cheap and composable.** Steering vectors stack. You can turn a dial on multiple behaviors at inference without re-training.
- **Reversible and auditable.** Unlike weight edits, a steering vector can be removed, inspected, or swapped per request.
- **A unifying lens.** Probing, refusal direction edits, concept erasure, and now RL post-training have all been shown to live on linear directions in residual space.
- **Safety relevance.** Gives a surface for behavioral intervention that is cheaper than SFT or RL and more targeted than system prompts.

## Gotchas & tricks

- **Best-probe-direction is NOT best-steering-direction.** The direction that reads a concept most reliably is not the one that writes it most reliably — steering favors directions closer to the mean-difference, probes favor task-discriminative ones. See [probing-vs-steering](probing-vs-steering.md).
- **Transfer is modest.** Steering vectors rarely transfer cleanly between models; probe directions transfer much better.
- **Scale matters.** Too large $\alpha$ pushes activations off-distribution and breaks fluency. Sweep per task.
- **Layer matters.** Mid-to-late layers are usually the sweet spot. Early layers push shallow surface features; final layers push logits without changing reasoning.
- **Not a free lunch for RL.** The RL-as-steering result ("a constant shift suffices for most of the gain") recovers most, not all, of RL post-training's gain; the residual gap tracks the non-linear component.

## Sources

- Paper: *RL Post-Training in Diffusion Transformers Is Activation Steering* — Oktay, Helbling, Kara, Chau, Yanardag, 2026 — fits steering vectors to match an RL-tuned model's velocity predictions and recovers most of the reward gain.
- Paper: *Do Probing and Steering Favor the Same Directions Within and Across Language Models?* — Sadeghi & Becker, 2026 — separates probe-best from steer-best directions.
- Paper: *Steering Language Models With Activation Engineering* — Turner et al., 2023 — early popularization of residual-stream additions.
- Paper: *Representation Engineering: A Top-Down Approach to AI Transparency* — Zou et al., 2023 — the concept-vectors framing.
- Paper: *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model* — Li et al., 2023 — ITI, the truth-direction instance.
