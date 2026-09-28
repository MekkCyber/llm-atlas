# IterSynth
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A role-decoupled replacement for ReAct-style deep-search agents. Splits the policy into a **Planner** (identifies information needs) and a **Synthesizer** (integrates evidence into an evolving summary), with the summary itself serving as the persistent context across turns. Trained with **Role-Decoupled Policy Optimization (RDPO)**: outcome reward + turn-level rubric evaluations + **role-specific advantages** so each sub-policy gets a clean credit signal.

**Prereqs:** [README.md](README.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/rdpo.md](../post-training/rdpo.md)

---

## What it is

ReAct pushes one policy through three roles at once: plan, consume evidence, synthesize an answer. Two structural problems follow:

1. **Role coupling** — the same weights must be good at all three roles simultaneously; capacity is spread thin and gradient signals conflict.
2. **Context accumulation** — as the agent iterates, the raw search history swells with noise, obscuring the useful signal.

IterSynth's two-step alternation splits the roles and swaps raw history for a curated summary.

## How it works

Two policies:

- **Planner π_P**: given the current summary state, output the next information need (a query, a tool call, a sub-question).
- **Synthesizer π_S**: given the retrieved evidence, produce the new summary state that integrates the evidence, replacing the old one.

The **summary state** is the only long-term memory across turns. Raw retrieval outputs are read once by the Synthesizer and then discarded. The two policies alternate until an answer condition is satisfied.

Training uses **RDPO** — see the sibling depth file — with role-specific advantages so the Planner is rewarded/punished for planning quality and the Synthesizer for integration quality, independently.

## Why it matters

- IterSynth-8B averages 50.7 across five long-horizon deep-search benchmarks (BrowseComp, Xbench-DS, and three others), beating the strongest ≤8B agent by **+4.2**.
- As a pure prompting paradigm it also lifts frontier proprietary models zero-shot over ReAct — the architectural insight is transferable even without RL.
- The Planner/Synthesizer split is a template for other multi-role agent settings (research, tool orchestration, multi-source retrieval).

## Gotchas & tricks

- The summary state is a lossy compression; if the Synthesizer drops the wrong details, later planning is uninformed. Rubric rewards for summary completeness help.
- Two-policy training is more sample-hungry than single-policy ReAct-style RL because each policy sees a subset of turns.
- The zero-shot prompting gains suggest the split itself does much of the work; the RL layer refines but does not create the effect.

## Sources

- Paper: *IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis* — Wu, Yan, Lu, Chen, Zhang, Liu, Deng, Liu, Ma, Shao, Xiao, Shen — 2026 — [arXiv:2609.29444](https://arxiv.org/abs/2609.29444).
