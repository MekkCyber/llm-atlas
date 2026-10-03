# LLM Backdoor Removal via Weight Orthogonalisation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free defence against trained-in LLM backdoors. Given an identified trigger, estimate a *backdoor direction* and a *refusal subspace* in activation space, then sequentially orthogonalise weights to suppress the backdoor while leaving refusal-related representations untouched. No clean reference model and no original poisoned data required. Introduced as **NEEDLE** (Kim, Lampos, Drayson, UCL, 2026).

**Prereqs:** [_attacks.md](_attacks.md), [refusal-suppression.md](refusal-suppression.md)
**Related:** [../interpretability/README.md](../interpretability/README.md) · [sleeper-agents.md](sleeper-agents.md)

---

## What it is

Backdoor attacks implant a trigger-conditioned misbehaviour at training time: when the trigger pattern appears in the input, the model emits an attacker-chosen output. Standard defences retrain on clean data, which drifts the whole output distribution — and often damages refusal behaviour as a side effect.

Weight-orthogonalisation defences take a different route: localise the backdoor to a specific direction in activation space, then project it out of the weights that produce that activation. The surrounding behaviour is preserved by *also* identifying the refusal subspace and ensuring the projection doesn't touch it.

## How it works

Given a model and an identified trigger pattern:

1. **Backdoor-direction estimation.** Collect activations on triggered vs non-triggered prompts; the direction that maximises the mean difference across targeted layers is the backdoor direction $v_{\text{bd}}$.
2. **Refusal-subspace estimation.** On refusal-eliciting prompts (harmful requests, policy violations), extract the subspace $U_{\text{ref}}$ that the model's refusal probability depends on. This is analogous to classical refusal-direction work.
3. **Weight orthogonalisation.** For each targeted weight matrix $W$ (typically the attention output projections and MLP down-projections), project out the component of $W$ that acts on $v_{\text{bd}}$ while explicitly preserving the span of $U_{\text{ref}}$. Apply the edits sequentially rather than jointly to avoid interactions.
4. **Verification.** Re-measure attack success and refusal rates; iterate if either regresses.

No gradient descent, no training data beyond the trigger set and a refusal prompt set.

## Why it matters

- **Keeps refusals intact.** Lowest KL divergence to the original model among evaluated defences, minimal capability and safety deltas.
- **Lowest mean Attack Success Rate** across model families and attack types in the paper's evaluation, including **0% ASR** on challenging code-injection backdoors.
- **No need for clean weights or data.** Many deployed models have no clean sibling and the original pretraining data isn't available; NEEDLE works from the compromised model alone plus a trigger hypothesis.
- **Interpretability-informed defences.** A template for other surgical weight edits — bias removal, factual edits, concept erasure — in the same orthogonalisation family.

## Gotchas & tricks

- **Trigger identification is upstream.** NEEDLE assumes you already know the trigger; it does not search for it. Pair with a separate trigger-detection step (anomaly detection on activations, probing against suspected trigger families).
- **Projection granularity.** Too-aggressive projection removes useful capability; too-gentle leaves residual backdoor. The paper's sequential application + iterative verification is non-trivial calibration.
- **Refusal subspace is model-specific.** Reusing a subspace from a different checkpoint can preserve the wrong directions and let refusal degrade silently.
- **Not a defence against yet-unknown triggers.** The method is reactive; novel triggers must be detected before removal applies.
- **Interactions with alignment pipelines.** Running NEEDLE after RLHF but before deployment is the natural slot; running it before RLHF risks the alignment stage re-introducing a correlated direction.

## Sources

- Paper: *Removing the NEEDLE in the Haystack: Backdoor Removal in LLMs via Weight Orthogonalisation* — Kim, Lampos, Drayson; UCL, 2026 — [arXiv:2610.00348](https://arxiv.org/abs/2610.00348).
- Related: refusal-direction work (Arditi et al., 2024) — same subspace machinery used offensively; NEEDLE uses it defensively.
- Background: BadNets and backdoor-attack literature (Gu et al., 2017; Carlini et al., 2024 for LLMs).
