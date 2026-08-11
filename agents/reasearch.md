# ReASearch
*Depth — internalize the outer-loop optimizer into a single tool-using agent.*

**TL;DR:** Prior systems for optimizing prompts, programs, or ML workflows relied on an explicit outer-loop controller (evolutionary search, bandits, textual gradients). ReASearch replaces that controller with a single reasoning-driven agent that itself decides what to evaluate, how to diagnose failures, which edits to make, and when to verify or restart. One shared scaffold + domain-specific tools handles all three optimization surfaces, with 2–40% gains over strong domain-specific baselines across 14 tasks — and a new best-known solution on Circle Packing.

**Prereqs:** [README.md](README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md), [cli-agent-scaffolding.md](cli-agent-scaffolding.md)

---

## What it is

Automated optimization of prompts / programs / ML workflows traditionally uses an *outer-loop controller* — an explicit algorithm (evolutionary, bandit, textual-gradient) that proposes candidates, tracks scores, and allocates budget. The model is a passive proposal generator.

ReASearch collapses controller and generator into one reasoning agent: the agent reads the task, chooses which candidates to evaluate, interprets the results, edits its proposal, and decides when to give up or restart — with persistent memory across steps.

## How it works

The shared agent loop is:

```
while budget remaining:
    reason about current state (memory)
    choose next tool call (propose, evaluate, edit, verify, restart)
    execute tool → observe result
    update memory
```

The scaffold is domain-general; only the *tools* change per domain:
- **Prompts:** propose prompt variants, evaluate on held-out set, edit, verify.
- **Programs:** propose code edits, run tests, diagnose failures, edit.
- **ML workflows:** propose config changes, launch small training runs, read metrics, adjust.

There is no separate scoring/proposal split. The agent's reasoning trace *is* the search policy — including diagnosis, budget allocation, and restart decisions.

## Why it matters

- **Competitive with specialized optimizers** across 14 diverse tasks — same scaffold, no per-domain controller, 2–40% gains over strong baselines.
- **Discovers new best-known solutions.** On Circle Packing — a classical combinatorial-geometry benchmark — ReASearch beats prior human best.
- **Search behaviors emerge from reasoning.** What was hard-coded in evolutionary / bandit controllers (proposal, diagnosis, budget allocation, restart) now falls out of the agent's reasoning trace.
- **Blurs the optimizer/agent boundary.** Suggests that a strong-enough reasoning agent + tools can subsume the specialized-optimizer ecosystem, similar to how LLMs subsumed feature engineering.

## Gotchas & tricks

- **Persistent memory is load-bearing.** Without long-term memory the agent can't allocate budget or recognize repeat failures.
- **Judge/eval quality caps agent quality.** Gains scale with the strength of the reference model used to judge candidates; a weak judge collapses the loop.
- **Tool set is the whole scaffold.** Same agent + different tools = different domain optimizer. Getting the tool interfaces right is more work than getting the agent right.
- **Not necessarily cheaper.** A reasoning agent that thinks per step is often more expensive per iteration than a bandit — the win is in fewer iterations and cross-domain reuse.
- **Emergent-not-guaranteed.** "Search behaviors emerge from reasoning" — the paper's observation, not a theorem. Weaker base models don't emit them.

## Sources

- Paper: *The Optimizer Is the Agent: Reasoning-Driven Search across Prompts, Programs, and ML Workflows* — Li, Liu, Xu et al., UT Austin / Snowflake, 2026 — arXiv:2608.06714.
