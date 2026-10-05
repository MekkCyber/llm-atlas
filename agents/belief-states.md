# Belief States for Long-Horizon Agents (PoS)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An inference-time framework (PoS) for long-horizon LLM agents that constructs and continually maintains an explicit **belief state** — the agent's current estimate of the world combined with its still-unresolved task requirements — as the decision context, rather than piling up raw interaction history or compressing it with summarisation.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [graphforge.md](./graphforge.md)

---

## What it is

Luo et al. (Alibaba, 2026) argue that memory-and-summary approaches to agent context do not ensure a coherent understanding of the current world. PoS replaces that implicit organisation with an explicit, continually-maintained belief that both the policy and a monitor read from. No fine-tuning; the whole mechanism lives at inference time around a base LLM.

## How it works

The belief at step $t$ is a pair:

- **World-state estimate** — the agent's best current understanding of relevant external facts (file contents, API responses, environment variables).
- **Unresolved requirements** — the subset of task criteria still outstanding, written out explicitly rather than left implicit in the history.

Each step runs three sub-processes:

1. **Belief update.** New observations are integrated into the world-state estimate; satisfied requirements are checked off.
2. **Consistency validation.** Internal contradictions in the belief (an observation inconsistent with earlier beliefs, or with an active goal) are flagged.
3. **Progress monitoring.** A separate monitor compares successive beliefs; if the unresolved-requirements set hasn't shrunk over a window of steps, **Belief Trapping** is declared.

On a trap, recovery is **specialised to the trapping pattern and the type of unresolved requirement** — e.g. "stuck in a tool-call loop with no new facts" triggers a different repair than "requirement unachievable in current environment state."

## Why it matters

- A separately-maintained belief is a structured context — shorter than raw history, more faithful than a running summary — that scales better as horizons grow.
- Highest overall score on four execution/diagnosis benchmarks across three LLM backbones (ablations show both consistency validation and tailored recovery contribute).
- Context-scaling experiments show resilience to context growth — summary-based baselines degrade faster.
- A natural fit on top of agent harnesses (OpenHands, MCP-based systems) because it layers on the base model without requiring fine-tuning.

## Gotchas & tricks

- The belief's **content** is still LM-generated — hallucinated facts propagate just like in any LLM scaffold; validation only catches *inconsistencies*, not confident falsehoods.
- Trap detection relies on choosing the right window: too short → false alarms, too long → agent wastes compute spinning in a trap.
- "Specialised recovery" is a dispatch over trapping patterns; coverage of real failure modes bounds the method's practical ceiling.
- Belief maintenance adds per-step inference cost (an extra model call); the paper measures net wins despite this.

## Sources

- Paper: *Beyond Memory: Harnessing Long-Horizon Agents with Explicit Belief States* — Luo et al., Alibaba, 2026 — [arXiv:2610.01415](https://arxiv.org/abs/2610.01415).
- Context: belief-state formulations in partially-observable sequential decision-making — Kaelbling, Littman, Cassandra, 1998 — the POMDP ancestry of this idea.
