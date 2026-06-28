# Model Ensembling

*Taxonomy — combine multiple LLMs (routing, voting, cascades, fusion, mixture-of-agents) to beat any single model.*

**TL;DR:** Multi-model systems try to convert *diversity of errors* into *accuracy gain*. There is a hard ceiling — `1 − β`, where β is the **co-failure rate** (every member wrong on a query) — that applies to any policy returning one member's answer. Pairwise error correlation ρ cannot identify β, which is why "low correlation ⇒ big gain" intuition often fails in practice.

**Related taxonomies:** [_computer-use.md](../agents/_computer-use.md) (orthogonal)
**Depth files covered here:** [co-failure-rate.md](co-failure-rate.md)

---

## The problem

Frontier models leave money on the table per query: any single model is wrong on a non-trivial subset, and different models are wrong on *different* subsets. The dream is to aggregate K member models into a system whose accuracy exceeds the best member. Five well-studied policies attempt this, and there's a sixth — co-failure-aware — that the field has only recently started reporting.

## The shared pattern

Every ensembling policy reads each member's answer (and optionally extra signals: confidence, judge scores, query features) and outputs *one* final answer. Two structural choices:

1. **Final answer = a member's answer (selection-style)** or **a new synthesized answer (synthesis-style)**.
2. **Aggregation signal:** confidence, agreement, judge scores, query routing features.

Selection-style policies inherit the **co-failure ceiling**: accuracy ≤ 1 − β where β = `P(all members wrong)`. Synthesis-style policies can in principle break the bound but rarely do in practice — synthesizers themselves struggle when all members are wrong.

## Variants

| Technique | Policy | Final answer | Ceiling-bound? |
| --- | --- | --- | --- |
| Best-of-N / voting | Pick most-voted | Member's answer | Yes |
| Routing | Pick member based on query features | Member's answer | Yes |
| Cascades | Run cheap member first, escalate on low confidence | Member's answer | Yes (over the routed set) |
| Fusion / reranking | Score each member's answer with a judge, pick top | Member's answer | Yes |
| Mixture-of-Agents (MoA) | Synthesizer LLM reads all members' answers | New synthesized answer | Approximately (synthesizer constrained by all-wrong case) |
| [co-failure-rate](co-failure-rate.md) | (Concept, not a policy.) Measure β; use as ceiling | n/a | Defines the bound |

## How to choose

- **Single-best baseline first.** Always compute single-best on the eval; if the candidate ensemble gain is within `1 − β − single_best`, you are not actually gaining from aggregation.
- **Routing wins when query-level signals are strong** (model A is reliably better on math, model B on code). Without those signals, routing degrades to averaging.
- **Voting wins when members are diverse and roughly equal in accuracy.** Otherwise the best member dominates and voting hurts.
- **Cascades win for cost.** Send most queries to the cheap model, escalate only the hard ones. Not an accuracy play.
- **MoA wins when synthesis adds genuine value** — synthesizing across long-form answers can break the selection-policy ceiling, but only marginally.
- **Always report β.** It's cheap to compute and instantly says whether the ensemble has any headroom at all.

## Adjacent but distinct

- **Self-consistency** — sample the *same* model K times and vote. Same voting policy, single member; co-failure rate is the model's error rate.
- **Reward-model reranking** — use an RM to score K samples from one model; not multi-model, same caveats apply.
- **Speculative decoding** — uses two models but for *latency*, not accuracy. Output distribution matches the target by construction.
- **Distillation from many teachers** — aggregates teachers at training time, not at inference. Different problem.

## Sources

- Paper: *Self-Consistency Improves Chain of Thought Reasoning in Language Models* — Wang et al., 2022 — voting from one model.
- Paper: *Mixture-of-Agents Enhances Large Language Model Capabilities* — Wang et al., 2024 — MoA synthesizer.
- Paper: *RouteLLM* — Ong et al., 2024 — learned router across model families.
- Paper: *When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models* — Josef Chen, KAIKAKU, 2026 — establishes the β ceiling.
