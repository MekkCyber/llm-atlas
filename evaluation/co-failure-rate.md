# Co-Failure Rate
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For any multi-model LLM system that ultimately outputs a *member model's* answer (routing, voting, cascades, fusion, mixture-of-agents), the achievable accuracy is upper-bounded by `1 − β`, where β is the **co-failure rate** — the fraction of queries on which every member model is wrong. The pairwise error correlation ρ that the field usually reports cannot identify β.

**Prereqs:** [_model-ensembling.md](_model-ensembling.md)
**Related:** [README.md](README.md)

---

## What it is

Routing, voting, cascades, fusion, and mixture-of-agents are all attempts to beat a single best model by combining several. Most analyses argue from *pairwise error correlation* ρ between member models: low ρ ⇒ more diverse errors ⇒ bigger ensembling gain.

The paper shows this is the wrong sufficient statistic. For any policy that outputs the answer of *one* member, accuracy cannot exceed `1 − β`, the rate at which *every member* gets it wrong. β depends on the full joint distribution of member errors, which pairwise ρ does not identify — many joint distributions with the same ρ have wildly different β.

## How it works

- For an ensemble of K models on a task, β = `P(all K models wrong on a random query)`.
- Measure β by sampling: count queries where every member's answer is incorrect.
- For policies that pick one member's answer (no synthesis), `1 − β` is a tight upper bound on accuracy.
- Calibration check: a "correctly calibrated" model under the standard sense still **under-prices** the all-wrong tail — by ~2.5× on mathematical tasks in the paper's data, across 67 frontier models from 21 providers.

## Why it matters

- **Caps the ceiling for ensemble research.** A lot of work optimizes ρ-style diversity; that doesn't move β. Without query-level routing signals strong enough to *avoid* co-failures, ensembles rarely beat the best single model.
- Reframes the field around β estimation and the design of router signals that correlate with the co-failure event.
- Provides a clean test for any new ensembling scheme: measure β, compute `1 − β`, compare to single-best — if the headline gain is within this cap, the gain isn't from clever aggregation.

## Gotchas & tricks

- β is task-dependent; an ensemble that breaks the ceiling on one domain can hit it on another.
- The bound is for policies that *output a member's answer*. Schemes that synthesize a *new* answer (e.g. mixture-of-agents with a synthesizer LLM) can sometimes break it — but rarely, because the synthesizer is itself constrained by the same all-wrong cases.
- Calibration under-pricing of the all-wrong tail is a property of common training/evaluation pipelines; better calibration on tail risk is an open problem.

## Sources

- Paper: *When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models* — Josef Chen, KAIKAKU, 2026 — arXiv:2606.27288.
