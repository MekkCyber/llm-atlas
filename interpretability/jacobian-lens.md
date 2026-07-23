# Jacobian vocabulary lens
*Depth — reading concepts from an input–output Jacobian rather than absolute hidden states.*

**TL;DR:** A logit-lens-style tool: instead of projecting the *hidden state* onto vocabulary, project the *Jacobian* of the hidden state with respect to the input onto vocabulary. What appears is not what the model *is* thinking, but what it would *change toward* if you nudged the input — surfacing directed/relational structure that static probes miss. Introduced in the Buehler materials-science paper to identify constitutive laws inside `google/gemma-4-E4B-it`.

**Prereqs:** *(none)*
**Related:** [README.md](./README.md)

---

## What it is

The classic logit lens reads the vocabulary distribution the model would produce if it stopped at layer `ℓ`. That's an **absolute-state** readout: it tells you what concept the residual currently emphasizes. It misses **directional** information — whether a relation points forward or reversed, whether stress increases or decreases with strain.

The Jacobian lens replaces the state with its Jacobian. Because the Jacobian is a linearization of *how the hidden state moves in response to input perturbations*, projecting it to vocabulary gives you the vocabulary directions the model actively *tracks* rather than the ones it happens to represent.

## How it works

For a hidden state `h(x)` at layer `ℓ` and an input `x`:

1. Compute `J = ∂h(x) / ∂x` (or a chosen sub-Jacobian — the paper uses input-token embeddings).
2. Project rows / SVD of `J` onto the unembedding matrix `W_U^T` to obtain candidate vocabulary directions.
3. Read out top tokens per direction.
4. Optionally: **directional counterfactual test.** Compare `J` on prompts that differ only in the direction of the physical relation (e.g. "increase" vs "decrease"). The vocabulary directions that flip identify laws the model actually orients.

Blinded from unrelated word sets, the paper shows the Jacobian lens **identifies 9 of 10 mechanism families** on 50 held-out materials descriptions; on 60 directional constitutive laws it correctly orients **39 of 40**.

## Why it matters

- **Direction is often what "understanding" means.** Getting the *sign* of a relation right is a stronger claim than knowing the vocabulary appears somewhere in state space.
- **Cheap, model-internal, no training.** No probes to fit, no counterexamples to write — Jacobians and unembeddings are already in hand.
- **Transferable.** The Buehler paper is materials science, but the recipe applies to any domain with directed relations (physics, causal reasoning, argumentative structure).

## Gotchas & tricks

- Sub-Jacobian choice matters. Full input-token Jacobian is expensive; the paper uses matched direct + Jacobian readouts to control for lexical confounds.
- An *apparent* physical organization can be entirely explained by numerical comparison (the paper explicitly audits for this — the mechanism-family clustering fails a graph audit unless directional counterfactuals are added).
- Requires access to weights and gradients — a closed-model API is not enough.

## Sources

- Paper: *Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model* — Markus J. Buehler (MIT), 2026 — [arXiv:2607.20058](https://arxiv.org/abs/2607.20058)
