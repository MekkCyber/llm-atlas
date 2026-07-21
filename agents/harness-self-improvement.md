# Recursive Harness Self-Improvement (RHI)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of fine-tuning the *model*, **RHI** iteratively refines the *harness* — the prompt-level agent-loop specification (memory, tool wiring, reflection prompts, message formats). The harness is represented as a text spec; a small number of RHI iterations, each using pairwise comparisons over the harness's own revision history, lift a low-reasoning-effort agent above the max-reasoning-effort setting on the same task while cutting inference cost by up to 60%.

**Prereqs:** [../post-training/dpo.md](../post-training/dpo.md)
**Related:** [skill-libraries.md](skill-libraries.md), [agent-world-models.md](agent-world-models.md)

---

## What it is

An agent's runtime performance is set by two things: **the model** (its weights) and **the harness** (the scaffold that decides when to think, when to call tools, what to remember, how to format context). Most improvement work touches the first — training runs, RL post-training, distillation. RHI leaves the model fixed and evolves the harness.

RHI's core observation: harnesses can be expressed as *prompt-level specifications* — text describing the agent loop's steps, message formats, and reflection points. Text specs are cheap to mutate, and pairwise A/B comparisons over spec versions give a preference signal that drives iterative refinement — the same primitive as DPO, but applied to the harness rather than the model.

## How it works

1. **Represent the harness as a text spec.** A prompt-level description of the agent loop: what messages get sent, what memory is kept, what tools are wired, what the reflection / retry policy is.
2. **Generate candidate revisions.** For each RHI iteration, produce N mutations of the current spec (via an LLM prompted to rewrite the spec).
3. **Evaluate candidates.** Run the agent under each spec on a held-out task set; collect performance + cost metrics.
4. **Pairwise preference over revision history.** Compare specs pairwise (win-rate weighted by margin), using the *history* of prior revisions as a preference dataset — not just the latest tournament.
5. **Update the spec.** Adopt the best candidate; repeat for a few iterations (the paper reports gains saturating quickly).

The paper's framing: RHI is implicitly optimizing **inter-agent information flow** — the quality of the context each turn of the agent sees, not the length of its reasoning trace. That's why gains hold with low reasoning effort at reduced cost.

## Why it matters

- **Fixed-model improvement.** Applies to any deployed model, including closed-weight ones, without retraining.
- **Cheap iterations.** A few tens of rollouts per RHI step, not millions of RL rollouts.
- **Middle path between manual harness engineering and model self-improvement.** Automates what harness engineers do by hand while side-stepping the risk / cost of touching weights.
- **Directly relevant to model–harness co-evolution.** As harnesses grow more complex (memory hierarchies, sub-agent chains, tool ecosystems), tuning them by hand becomes intractable — RHI is the mechanized version.

Reported result: across 30 synthetic ML-research tasks (quant finance, robotics, pharmacy), a few RHI iterations let low-reasoning-effort agents exceed the max-reasoning-effort baseline while running up to 60% cheaper.

## Gotchas & tricks

- **Task-specific.** RHI harnesses generalize weakly across task families; run it per-domain.
- **Pairwise preferences over history, not just latest tournament.** Using the full revision history stabilizes updates against noisy single-round comparisons.
- **Watch for overfitting to the held-out tasks used for scoring.** RHI is optimizing directly against them — hold out a *second* set for reporting.
- **Text-spec representation matters.** A structured spec (with named slots) mutates more usefully than an unstructured prose blob.
- **Compose with model improvements, don't substitute.** RHI is orthogonal to base-model quality; better model + refined harness compounds.

## Sources

- Paper: *Recursive Harness Self-Improvement* — Lee, Xu, Seely, Lee, Zaharia, Tang, 2026 — [arXiv:2607.15524](https://arxiv.org/abs/2607.15524).
- Related: *DPO: Direct Preference Optimization* — Rafailov et al., 2023 — the pairwise-preference primitive RHI re-uses.
- Related concept: *Voyager* — self-authored skill library, an early instance of runtime-level self-improvement.
