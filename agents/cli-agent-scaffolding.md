# CLI agent scaffolding & DCAS
*Depth — cross-scaffold evaluation and planning-aware trajectory collection for CLI coding agents.*

**TL;DR:** Open-model CLI software-engineering agents (Claude Code / OpenHands / Cline / Aider style) score well under the scaffold they were trained on and degrade sharply under any other scaffold. This isn't the base model's fault — untrained base models don't show the divergence. The culprit is fine-tuning on scaffold-specific *planning* conventions. DCAS (Decoupling CLI Agent Scaffolding) is a backend-substitution interception layer that routes API traffic between any CLI scaffold and any backend model, enabling cross-scaffold eval and planning-aware trajectory collection.

**Prereqs:** [README.md](README.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [reasearch.md](reasearch.md)

---

## What it is

Open-model coding-agent trajectory datasets are collected almost exclusively under one scaffold (OpenHands). Models fine-tuned on those trajectories internalize the scaffold's planning conventions and break under other scaffolds. DCAS gives the community an infra layer to break that lock — you can route the same model through any CLI scaffold, and collect training trajectories that internalize planning as a *model* capability rather than a *scaffold* artifact.

## How it works

**Two senses of "planning" the paper distinguishes:**
- **Explicit planning:** a pre-execution plan produced as a first-class artifact (e.g. a TODO block emitted before any tool call).
- **Implicit planning:** structural conventions that shape execution throughout the loop (tool-call ordering, verification pattern, retry policy).

Both are typically the scaffold's responsibility today. Models fine-tuned under one scaffold learn its planning style and lose it under others.

**The DCAS layer:**
1. Sits between any CLI scaffold (frontend) and any backend model.
2. Intercepts API traffic; can substitute backends transparently.
3. No modification to the scaffold's code — plug in via API endpoint override.

**Applications enabled by DCAS:**
- **Cross-scaffold evaluation.** Run the same fine-tuned model under OpenHands, Cline, Aider, etc. — DCAS routes all their traffic to your model.
- **Plan-source intervention.** Deliberately vary the *source* of the plan (scaffold-generated vs model-generated vs oracle-generated), attribute the resulting quality gap.
- **Planning-aware trajectory collection.** Collect trajectories under one scaffold where the plan is model-generated, then fine-tune on them — the resulting model generalizes across scaffolds.

**Reported findings.** A controlled plan-source intervention shows planning quality's contribution *exceeds* the cross-scaffold drops. A model fine-tuned on a small DCAS-collected planning-aware trajectory set under one scaffold generalizes to non-training scaffolds — closing the drop.

## Why it matters

- **Diagnoses a real deployment failure.** Every team deploying open coding-agent models outside OpenHands has hit this drift; DCAS names the mechanism.
- **Gives the community shared infra.** Instead of every team building its own scaffold-swap harness, DCAS is the standard interception point.
- **Moves planning to the model.** Planning-aware trajectories internalize what was scaffold state, unlocking scaffold-agnostic deployment.
- **Model-independent.** DCAS doesn't care about the backend; works with any API-shaped model.

## Gotchas & tricks

- **Requires the scaffold to hit an API endpoint** (not link a model inline). Most CLI scaffolds do.
- **Trajectory-collection quality depends on the plan-source you choose.** Model-generated plans are the target for internalization; oracle plans are best for eval.
- **"Cross-scaffold gain" isn't universal.** Some scaffolds have unique tool APIs; a model trained under one may still lose access to another's tools even after planning is internalized.
- **Backend substitution is safety-relevant.** Routing traffic through DCAS = the layer sees every message; treat access controls as a first-class concern.

## Sources

- Paper: *DCAS: Decoupling CLI Agent Scaffolding to Internalize Planning across Scaffolds* — Thangarajah, Chen, Hassan et al., Centre for Software Excellence, 2026 — arXiv:2608.06113.
