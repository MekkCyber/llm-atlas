# Counterfactual T2I Evaluation (CF-World)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Benchmark for text-to-image causal reasoning: prompts whose visual outcomes deliberately contradict real-world priors. Three progressive levels (factual → explicit counterfactual → implicit counterfactual), a VLM-based evaluator (CF-Eval), and two metrics: Prior Resistance Rate (PRR) and Reasoning Retention Rate (RRR). Both open- and closed-source T2I models degrade sharply going from factual to counterfactual prompts.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md) · [diffusionbench.md](diffusionbench.md)

---

## What it is

T2I leaderboards score models on prompts that match common training co-occurrences ("a cat on a windowsill"). They reward fidelity, not understanding. CF-World instead probes whether T2I models can *break* commonsense correctly — rendering scenes whose physics or semantics contradict the training distribution.

The framing: Russell's *inductivist turkey*, which expects the farmer to bring food every morning until the morning of Thanksgiving. A T2I model trained on millions of "the sun rises" images is the same turkey — and CF-World is its Thanksgiving.

## How it works

Each scenario is structured into three difficulty levels:

| Level | Prompt style | What it tests |
| --- | --- | --- |
| Factual | Ordinary scene under standard physics/semantics | Baseline generation quality |
| Explicit counterfactual | Direct visual instruction overriding the prior (e.g. "water flowing upward") | Can the model obey direct counterfactual instructions? |
| Implicit counterfactual | Causal deduction from an altered rule (e.g. "in a world where gravity is reversed, what does a falling apple look like?") | Can the model deduce visual consequences from altered rules? |

Two metrics:
- **Prior Resistance Rate (PRR)** — fraction of counterfactual prompts where the model overrides its real-world priors rather than retreating to commonsense defaults.
- **Reasoning Retention Rate (RRR)** — fraction of *implicit* counterfactual prompts where the model maintains the counterfactual logic without being told visually.

Evaluation uses a VLM-based judge (CF-Eval) over the generated images.

## Why it matters

- Demonstrates a class of T2I failures invisible on standard benchmarks: the model is fluent on familiar scenes and collapses on prompts that require breaking visual co-occurrences.
- The PRR / RRR distinction separates two kinds of failure — refusing to override priors vs. inability to reason through counterfactual rules — which point to different fixes.
- A diagnostic for the deeper question of whether T2I models encode world knowledge as decoupled facts or as entangled visual co-occurrence patterns. The paper's analysis supports the latter.

## Gotchas & tricks

- VLM-based evaluation inherits the evaluator's blind spots; cross-judge agreement on borderline images is the validity check the paper runs.
- The implicit-counterfactual category is the hardest to construct; sloppy prompts collapse into the explicit category.
- Open T2I models do better on PRR than closed when the closed model has stronger commonsense priors — a counterintuitive ranking flip that the leaderboard format surfaces.

## Sources

- Paper: *Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning* — Lei, Pu, Han, Zhu, et al., SJTU / Shanghai AI Lab / CUHK, 2026 — [arXiv:2606.24548](https://arxiv.org/abs/2606.24548).
