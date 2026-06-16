# Activation Steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Add a fixed vector to a model's hidden states at inference time to push generations toward (or away from) a concept — refusal, sycophancy, a topic, a persona. The vector is computed from contrastive pairs (positive vs. negative concept examples), applied at one or more layers, and scaled by a *steering strength* coefficient. Effective but brittle: the same vector at the same strength under-steers some prompts (concept barely shifts) and over-steers others (coherence collapses). The ASTEER work shows steering outcome can be predicted from *early hidden states* — enabling efficient strength search and per-prompt control.

**Prereqs:** [README.md](README.md)
**Related:** [../safety/refusal-suppression.md](../safety/refusal-suppression.md)

---

## What it is

A linear control over model behavior that doesn't require retraining. The recipe:

1. Collect contrastive pairs of inputs $(x^+, x^-)$ differing along a concept axis (e.g. "be helpful about chemistry" vs. "refuse chemistry questions").
2. Run both through the model; cache hidden states $h^+_\ell, h^-_\ell$ at each layer $\ell$.
3. The **steering vector** $v_\ell = \text{mean}(h^+_\ell) - \text{mean}(h^-_\ell)$.
4. At inference, modify the residual stream: $h_\ell \leftarrow h_\ell + \alpha \cdot v_\ell$ with steering strength $\alpha$.

The technique is rooted in the linear-representation hypothesis: concepts correspond to directions in activation space, so adding the direction nudges generation toward the concept.

---

## How it works

### Vector construction

Two common variants:

- **Difference-of-means** — the simplest; works well when the concept is a clean axis.
- **Probing-style** — train a linear probe distinguishing $x^+$ from $x^-$; the probe's weight vector is the steering direction. Better when the concept is entangled with confounders.

Per-layer vectors are typical; the layer that best captures the concept varies by model and concept and is often picked by probing accuracy.

### Strength and application point

The strength coefficient $\alpha$ is the brittle knob:

- $\alpha$ too small: the concept barely registers in generation.
- $\alpha$ too large: the model loses coherence, produces nonsense, or fixates on the concept word-for-word.
- The "sweet band" is prompt-dependent — there's no single $\alpha$ that works for every prompt.

Common application choices: add at one mid-to-late layer; add at all layers from $\ell$ onward; or apply only on the prompt tokens vs. the generated tokens.

### ASTEER — outcome prediction

The paper builds a 1.4M-generation dataset labeled with steering outcome (under-steered / successful / over-steered) across 150 concepts. A gradient-boosting classifier over **early hidden states** (before generation begins) predicts outcome at ~0.7 macro-F1 on held-out concepts. The predictor is then used to drive an efficient strength search: try a small grid of $\alpha$, classify each without rolling out, pick the predicted-best.

This converts "sweep $\alpha$ until output looks good" into a one-shot, prompt-conditional choice.

---

## Why it matters

- **Cheap behavioral control.** No fine-tuning, no preference data; a few hundred contrastive pairs suffice for a working vector.
- **Composable.** Multiple vectors stack additively (with predictable interactions only when directions are nearly orthogonal).
- **Interpretability handle.** Whether a model is steerable on a concept tells you something about how the concept is represented internally.
- **Practical steering needs outcome prediction.** Without it, deployments either over-steer (visible failure) or under-steer (no effect). The ASTEER predictor closes that loop.

---

## Gotchas & tricks

- **Steering ≠ deep modification.** The model's underlying capability is unchanged; a steered model can still represent the suppressed concept in earlier layers, and adversarial prompts often bypass steering entirely.
- **Direction generalization is fragile.** Vectors built from one distribution of contrastive pairs may not steer well on out-of-distribution inputs.
- **Refusal is partly a direction.** Arditi et al. (2024) show refusal is mediated by a single direction; ablating it removes safety training. This is the same machinery, used adversarially.
- **Strength search is the main user-facing cost.** ASTEER-style predictors make this cheap; without one, expect to rollout-and-eyeball at multiple $\alpha$.
- **Layer choice interacts with vector source.** Difference-of-means vectors often work best at the layer where the contrastive examples diverge most; probe-based vectors are less layer-sensitive.
- **Over-steering looks like a different failure than under-steering.** Over-steering produces fluent-but-fixated text; under-steering produces normal text that ignores the steer. Distinguish them when debugging.

---

## Sources

- Paper: *When is Your LLM Steerable? (ASTEER)* — Fan, Cheng, Li, Feizi, Zhou, UMD · MBZUAI, 2026 — [arXiv 2606.11599](https://arxiv.org/abs/2606.11599).
- Paper: *Refusal in Language Models Is Mediated by a Single Direction* — Arditi et al., 2024 — [arXiv 2406.11717](https://arxiv.org/abs/2406.11717) — the refusal-direction result that motivates representation-level safety.
- Paper: *Steering Language Models with Activation Engineering* — Turner et al., 2023 — the original activation-addition formulation.
