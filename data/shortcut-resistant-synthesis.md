# Shortcut-Resistant Synthesis

*Depth — generating training data for search/reasoning agents in a way that suppresses shortcut routes to the answer.*

**TL;DR:** When you synthesize training questions for a search agent, the *structural* difficulty of the evidence graph isn't enough — if there's a single highly identifying clue, a leaked constant, or a fact the model already knows, the agent will collapse to that shortcut instead of learning to search. FORT formalizes four shortcut risks and applies controls at every pipeline stage (entity selection, evidence-graph construction, question formulation, adversarial refinement). Trained with SFT-only on the resulting trajectories, FORT-Searcher beats comparable-size open-source deep-search agents.

**Prereqs:** [_data-curation](_data-curation.md), [decontamination](decontamination.md)
**Related:** [quality-filtering](quality-filtering.md)

---

## What it is

A data-synthesis discipline for training search and tool-using agents. The setup: each training example is a question whose answer requires acquiring evidence through search; the agent's trajectory is what you actually fine-tune on. The failure mode the paper targets is that *most* synthesis pipelines produce questions that *look* hard (deep multi-hop evidence graphs) but are *cheap* to answer (one clue uniquely identifies the target). The agent then memorizes the shortcut and never learns to search.

Shortcut resistance is the property: the agent has to acquire a meaningful fraction of the evidence before any answer becomes uniquely determined.

## How it works

Four shortcut risks:

1. **Evidence co-coverage** — multiple evidence items collectively cover redundant info, so any subset suffices. Control: make each evidence item necessary.
2. **Single-clue selectivity** — one clue (a rare name, a unique date) uniquely identifies the answer; the model finds it and stops. Control: avoid high-selectivity clues, or require clues that only narrow down combinatorially.
3. **Exposed constants** — numbers, dates, or named entities in the question that act as keys into the knowledge base. Control: replace exposed constants with descriptions, or filter them in the question-formulation stage.
4. **Prior-knowledge binding** — the answer is something the *base model already knows*, so no search is needed at all. Control: decontaminate against the base model with a probe pass.

Each risk gets a control at a specific pipeline stage:

- **Entity selection** — sample entities that aren't disproportionately memorized.
- **Evidence graph construction** — build dependencies so no subset is sufficient.
- **Question formulation** — phrase the question without exposing the constants.
- **Adversarial refinement** — generate candidate questions, simulate trajectories, and reject any that solve with too few steps or hit the answer too early.

Diagnostics: **trajectory signatures** — solving cost, answer hit time, and prior-shortcut rate — measured on the trained agent against the synthesized data. These are *realized* difficulty metrics rather than structural ones.

## Why it matters

- Reveals a hidden failure mode in essentially every existing search-agent training pipeline: structural-complexity-only synthesis under-trains search.
- Provides a measurable framework (the four risks + the three trajectory signatures) so future synthesis methods can be compared on shortcut resistance.
- Shows that **SFT-only** is enough to beat prior open-source search agents when the data is right — the RL phase often masks data-side problems.

## Gotchas & tricks

- **Diagnostics need to run on the actual trained agent.** Static analysis of the evidence graph can't detect prior-knowledge binding; you have to measure it on the policy.
- **Decontamination is per-base-model.** Move to a stronger base and your "shortcut-resistant" dataset may suddenly become trivial again, because the new base already knows the answers.
- **Co-coverage is subtle.** Even if no single evidence item uniquely identifies the answer, two items might. Adversarial refinement that simulates trajectories catches this.

## Sources

- Paper: FORT — Chen et al. (2026) — [arXiv:2606.12087](https://arxiv.org/abs/2606.12087)
