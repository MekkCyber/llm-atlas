# SAEScientist-Bench

*Depth — a benchmark measuring whether AI agents can conduct autonomous sparse-autoencoder (SAE) mechanistic-interpretability research.*

**TL;DR:** Interpretability is a bottleneck for oversight; if agents can automate it, mech-interp scales. SAEScientist-Bench frames SAE-based interpretability as an *agent task*: given a target concept, an agent must design contrastive probes and navigate a Gemma Scope dictionary of 131K+ features on Gemma-2-9B-IT to identify the feature that captures the concept. Agents are evaluated on **activation rank**, **concept selectivity**, and **causal-steering effect** against expert-curated reference features. Across 10 agent configurations × 20 tasks, frontier agents approach experts on selectivity but lag substantially on causal steering.

**Prereqs:** [../interpretability/README.md](../interpretability/README.md)
**Related:** [README](README.md), [discovery-certification-protocol](discovery-certification-protocol.md), [../safety/README.md](../safety/README.md)

---

## What it is

Mechanistic interpretability with SAEs is a labor-intensive, iterative process: pick a concept, design contrastive text pairs, activate the model on them, look through the top-firing features in the SAE, causally intervene to check that steering the feature affects the concept in generation. Expert practitioners spend hours per feature.

SAEScientist-Bench operationalizes the loop as an agent task:

- **Input.** A target concept in natural language.
- **Environment.** Gemma Scope's 131K+ SAE features over Gemma-2-9B-IT, plus tools for scoring feature activation on inputs and steering generation by intervening on a feature.
- **Objective.** Return the SAE feature index that best captures the concept.
- **Reference.** Expert-curated features anchored on Neuronpedia.

## How it works

The agent has three tools:

- **Contrastive probing** — pass texts through the model, read SAE activations.
- **Feature ranking** — score candidate features by activation rank on contrastive texts.
- **Causal steering** — clamp a feature and observe the change in the model's generation.

Scoring uses three axes (matched against the expert reference feature):

- **Activation rank** — how highly ranked the chosen feature is on concept-relevant texts.
- **Concept selectivity** — how well it separates concept vs. contrastive controls.
- **Causal steering** — how much clamping the feature changes generation in the concept-relevant direction.

Across 10 agent configurations (different models, different scaffoldings) × 20 tasks (different target concepts), the paper reports per-axis scores and a composite.

## Why it matters

- **Concrete, quantitative goal for mech-interp agents.** No more "I found a feature" — the score axes are auditable.
- **Diagnoses *where* agents fail.** Frontier agents approach experts on selectivity — they know how to design contrasts — but lag on causal steering — they misread experimental measurements. That's a specific failure mode to target.
- **Complements DCP.** SAEScientist-Bench measures ability; DCP would ask whether a specific claimed feature discovery is *novel*.

## Gotchas & tricks

- **Task pool is small (20 concepts).** Overfitting to the benchmark is a risk. Broader concept coverage would strengthen conclusions.
- **Gemma-2-9B-IT-specific.** Results may not transfer to other model families / SAE dictionaries.
- **The expert reference is not a ground truth.** Multiple features may capture a concept; the reference is *an* expert-vetted answer, not the unique correct one.
- **Steering measurement is noisy.** Small changes in temperature or prompt can flip the direction; the eval fixes both.

## Sources

- Paper: *SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?* — Tan, He, Zhao, Liu, 2026 — [arXiv:2609.09113](https://arxiv.org/abs/2609.09113).
- Companion: Gemma Scope / Neuronpedia (SAE dictionary + labeled features).
