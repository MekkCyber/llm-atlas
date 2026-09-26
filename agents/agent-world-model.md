# Agent-Editing World Model
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An alternative "world model" for LLM agents that does **not** try to predict tool outputs (they are high-entropy and real feedback is already available). Instead, model how a **reasoning-action pair changes the agent's task state**, and add two components: an **Action Judge** that classifies each decision, and **State Revision** that rewrites stale or contradicted reasoning in-place so bad prior assumptions stop poisoning downstream steps. Introduced by AEWM.

**Prereqs:** [agents README](README.md)
**Related:** [cot-monitoring](../safety/cot-monitoring.md), [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

ReAct-style long-horizon agents accumulate their own history — plans, observations, tool responses, intermediate thoughts. Two failure modes are well-known:

1. **Task-state contamination.** An early unsupported assumption ("the API returns JSON") persists in the scratchpad and distorts subsequent decisions even after evidence to the contrary.
2. **Redundant world-model overhead.** Existing "language world models" predict environment observations, but real tool responses are available at runtime, so simulating them is wasted capacity.

AEWM reframes the world model as an *editor of the agent's task state*, not a simulator of the environment.

## How it works

Two lightweight modules stack around the base agent:

- **Action Judge.** Small classifier that reads each (reasoning, action) pair and categorizes it — e.g. progress vs. revisit vs. redundant vs. contradicted. Runs after every step.
- **State Revision.** When the Judge flags a step as contradicted or stale, an editor LLM rewrites the affected reasoning span in the agent's history *in place* — preserving the trajectory format but replacing the outdated claim with the corrected belief.

The base agent (ReAct, planner-executor, etc.) continues to drive tool use; AEWM only edits its scratchpad between steps. No retraining of the base LLM is required.

## Why it matters

- Attacks a specific, documented failure mode of long-horizon agents with a mechanism that plugs into existing architectures.
- Consistent gains across multiple long-horizon benchmarks and across base agents; the biggest wins are on tasks whose original agents suffer most from stale-belief contamination.
- Suggests "world modeling" for LLM agents should be about belief maintenance, not observation prediction.

## Gotchas & tricks

- State Revision must preserve the trajectory shape or downstream tools that pattern-match on the history break — the editor is prompt-tuned, not free-form.
- Aggressive revision can erase context the agent will genuinely need later; the Action Judge's precision matters more than its recall.
- Works best as a wrap around a mid-sized base agent; on very small models it can amplify noise rather than filter it.

## Sources

- Paper: *Agent-Editing World Model: Rethinking World Modeling for LLM Agents* — Sun et al., Renmin U. / Gaoling, 2026 — [arXiv:2609.28416](https://arxiv.org/abs/2609.28416).
