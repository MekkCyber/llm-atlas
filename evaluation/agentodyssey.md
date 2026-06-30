# AgentOdyssey
*Depth — procedurally-generated long-horizon text games for evaluating test-time continual learning agents.*

**TL;DR:** A benchmark that drops the standard ML assumption "learning stops at deployment." AgentOdyssey procedurally generates open-ended text games with rich entities, world dynamics, and long horizons, and places agents in a continuous setting that *interleaves* learning and inference. Beyond a final game-progress score, it reports diagnostics for world-knowledge acquisition, episodic memory, exploration, action diversity, and cost.

**Prereqs:** [evaluation README](../evaluation/README.md)
**Related:** [agents README](../agents/README.md)

---

## What it is

A generator + harness rather than a static dataset. Each evaluation instance is a freshly-procedurally-generated text world (entities, rules, objectives) the agent has never seen. Episodes are long enough that effective agents must accumulate world-state knowledge and retrieve relevant past experience while still acting.

## How it works

**Procedural generation.** A schema-driven generator produces worlds with rich entity graphs, action affordances, and dynamics rules; tasks are long-horizon (many decisions, partial observability) so success requires interleaved exploration + exploitation.

**Continuous learning + inference.** Unlike conventional benchmarks (train → freeze → evaluate), AgentOdyssey rewards updating the agent's policy / memory *during* evaluation. This is the operational definition of test-time continual learning.

**Diagnostics.**

- *World knowledge acquisition*: did the agent internalize a true rule of this world?
- *Episodic memory*: does it retrieve relevant past observations when they help now?
- *Object / action exploration*: breadth of contact with the entity / action space.
- *Action diversity*: not just repeating the same successful pattern.
- *Cost*: total compute / model calls per game-progress unit.

## Why it matters

- **All evaluated agents lag humans substantially**, even when built on the strongest base models. The headroom is structural, not just scale.
- **Short-term memory is broadly beneficial** across paradigms — surfaces it from a tool-design choice to a measurable capability axis.
- **Diagnostic granularity** makes regressions/wins interpretable: "score dropped" is replaced by "exploration collapsed in tier 3."
- Fills a real gap: most agent benchmarks are static or short-horizon and underrate continual-learning capabilities.

## Gotchas & tricks

- Procedural generation can drift in difficulty unless calibrated; the paper exposes structural knobs for tier control.
- Reporting only the aggregate game-progress score throws away the diagnostic value — treat diagnostics as first-class metrics.
- Agents that rely on heavy episodic stores have a cost-axis disadvantage; the cost metric makes that trade-off visible instead of hidden.

## Sources

- Paper: *AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents* — Zehao Wen et al., Johns Hopkins — arXiv:2606.24893 — https://arxiv.org/abs/2606.24893
