# Reasoning-Operations Probing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Chains-of-thought are not just scratchpad text — the reasoning *operations* they express (problem formulation, goal decomposition, deduction, verification) are geometrically distinguishable in the model's hidden representations. Middle layers show peak operation separability; identical surface tokens map to different operation-aligned representations depending on the surrounding CoT context, and attention-masking interventions confirm the operation signal is inherited from prior reasoning context.

**Prereqs:** [README.md](./README.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)

---

## What it is

CoT text obviously contains functionally different steps ("Let me set up the problem…", "Now decompose the goal…", "Therefore…"). The question is whether the model's internal state *also* differentiates these steps geometrically — a prerequisite for treating "which reasoning operation is happening" as a probe-able / steerable variable.

The paper's answer: yes, in mid-layer activations, and not because of lexical or positional confounds.

## How it works

- **Label each CoT chunk with a reasoning operation.** Categories are drawn from the CoT literature: problem formulation, goal decomposition, deduction, verification, etc.
- **Probe.** Train a linear classifier on hidden representations at each layer to predict the chunk's operation label from held-out chunks.
- **Peak-layer test.** Separability is highest in middle layers, low near the input and output.
- **Confound checks.** Ablate lexical cues (mask characteristic keywords) and positional cues (permute chunk order) — separability persists.
- **Attention-mask intervention.** Mask attention to *preceding* CoT context; representations at chunk onset lose their operation alignment. This pins the mechanism: the operation "signal" at a chunk's first tokens is inherited from what came before, not produced from the token identity alone.
- **Token-level bifurcation.** The same surface token maps to different operation-aligned representations depending on surrounding chunk — reasoning context reshapes even identical inputs.

## Why it matters

- **Structured internal computation for reasoning.** First mechanistic evidence that long-CoT models carry a *typed* reasoning state, not merely a text scratchpad. Opens a natural handle for interp work on reasoning.
- **Substrate for steering.** If operations are linearly separable, they are (in principle) steerable — activation-additive interventions that push the model from formulation to deduction, or that stabilize a stuck verification loop.
- **Complements process-reward work.** [PRMs](../post-training/reasoning/prm.md) reward *what* a reasoning step is; this paper measures *how* the model internally represents that step. Together they cover the outside and inside of a step.

## Gotchas & tricks

- **Operation taxonomies are not standardized.** Choice of category set changes the numbers; report separability against a fixed vocabulary.
- **Middle-layer peak is model-dependent.** Depth of peak varies across families; reproduce the sweep before drawing "layer 20 of every model is where reasoning lives" conclusions.
- **Attention-masking is intrusive.** Masking a chunk's back-attention corrupts the model's ability to reason at all; interpret the ablation as removing the *source* of the operation signal, not as a purely diagnostic change.
- **Correlation vs. causation.** Linear separability shows the *representation* aligns with reasoning operations. It does not by itself show that intervening on that representation changes the reasoning outcome — steering experiments are the next step.

## Sources

- Paper: *Beneath the Surface of Chains-of-Thought: A Mechanistic Interpretation of Reasoning Operations in LLMs* — anonymous (NRF-Korea funded) — arXiv:2609.04753 (2026).
