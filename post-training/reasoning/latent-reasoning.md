# Latent-Space Reasoning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reasoning in the model's continuous hidden-state space rather than by emitting more tokens. Instead of a long chain-of-thought over tokens, a small number of test-time updates operate directly on selected activations. Historically underperformed explicit CoT; recent variants (Coconut, iCoT, GradCuit) have narrowed and in some settings closed the gap.

**Prereqs:** [long-cot-rl](long-cot-rl.md)
**Related:** [prm](prm.md), [orm](orm.md), [../../interpretability/README.md](../../interpretability/README.md)

---

## What it is

Two ways an LM can "think more":

1. **Explicit CoT** — decode extra tokens (`Let me work through this step by step...`) that carry the intermediate computation. Standard, well-understood, but expensive (each thought costs tokens) and interpretability is only surface-level (models don't always say what they're doing).
2. **Latent reasoning** — perform additional computation *inside* the model on continuous hidden states before emitting the final answer. No extra tokens, potentially more capacity per unit of compute.

Latent reasoning breaks into two sub-classes:
- **Learned latent reasoning** — the model is trained to reason in a continuous latent space (Coconut, iCoT). Requires modifying training.
- **Test-time latent reasoning** — at inference, run gradient or optimisation steps on hidden states of a *frozen* model (GradCuit). Training-free.

## How it works

**Coconut / iCoT (learned).** Replace some CoT tokens with continuous "thought vectors" that don't decode to words. Train the model with a curriculum that progressively substitutes latent thoughts for text thoughts, so the model learns to reason without decoding.

**GradCuit and cousins (test-time).**

1. Freeze the base LM.
2. At inference, run a small number of gradient steps on selected hidden-state activations, driven by a verifier / reward signal (a short verification model or an ORM/PRM).
3. Use a **credit-assignment mask** to decide which activations to update — only those "responsible" for the current mistake.
4. Continue generation from the updated activations.

The credit-assignment mask is what distinguishes GradCuit from generic activation-steering: instead of a hand-chosen direction, the gradient itself picks which latents to move, restricted by the mask.

## Why it matters

- **Test-time compute in fewer tokens.** Latent reasoning uses forward-pass compute rather than more decoded tokens. For latency-sensitive applications this can matter.
- **Interpretability handle.** In test-time variants, the moved latents *are* an inspectable reasoning trace — you can measure how much each activation changed and where.
- **Robustness.** Latent traces are harder to derail with adversarial prompt perturbations than token CoT.
- **Complementary to explicit CoT** — most systems combine both.

## Gotchas & tricks

- **Explicit CoT is still the strongest signal.** For most tasks, adding more CoT tokens still beats latent-only reasoning. Latent reasoning is a supplement, not a replacement.
- **Verifier quality is the ceiling.** Test-time gradient updates require *something* to say "this is closer to correct." A weak verifier means noisy updates that hurt.
- **Numerical stability.** Gradient steps on hidden states can blow up if the loss surface is spiky; small step sizes and gradient clipping are essential.
- **The "credit assignment" mask is fragile.** Naïve full-activation updates degrade quickly; only credit-assigned variants (GradCuit-style) reliably improve.
- **Coconut-style curriculum training is not free.** Substituting latent thoughts for text thoughts requires careful curriculum design; too fast and the model collapses to a shortcut policy.
- **Doesn't obviously scale like decoded CoT.** Adding more test-time tokens has a clear scaling law (more tokens → better within limits); latent updates plateau faster.

## Sources

- Paper: *Coconut: Chain of Continuous Thought* — Hao et al., Meta, 2024 — learned latent reasoning.
- Paper: *iCoT: Implicit Chain-of-Thought via Continuous-Space Reasoning* — 2024.
- Paper: *GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning* — arXiv:2608.02585, 2026.
- Paper: *Steering Language Models with Activation Engineering* — Turner et al., 2023 — an earlier point on the activation-manipulation spectrum.
