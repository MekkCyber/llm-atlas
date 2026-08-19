# Subtask Workflows
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Raw agent demonstrations (recorded click/keystroke traces) are noisy, screen-fragile, and hard to reuse. **Subtask workflows** are an intermediate representation, introduced in UI-Mate (2026), that recast a demonstration into a sequence of *named subtasks* the agent can re-plan against — skip a subtask that's already done, substitute when the UI shifts, or recompose subtasks across tasks. A closed-loop data engine grows the workflow library from live agent runs.

**Prereqs:** [_agent-harness.md](_agent-harness.md), [gui-agents.md](gui-agents.md)
**Related:** [harness-scaling.md](harness-scaling.md) · [model-routing.md](model-routing.md)

---

## What it is

A **subtask workflow** is a structured plan of the form:

```
task: "Send Q3 report to accounting"
  subtask 1: locate the Q3 report file
  subtask 2: open a new email
  subtask 3: attach the file
  subtask 4: address the email to accounting-alias@…
  subtask 5: send
```

Each subtask has (i) a natural-language description, (ii) preconditions checked against the current world state, (iii) a demonstration snippet (or several) showing how to accomplish it, and (iv) a postcondition the harness verifies before moving on. The agent's job at inference is to *pick and adapt* subtasks, not to replay a raw trace.

## How it works

1. **Convert traces to subtasks.** A conversion pipeline (LLM-in-the-loop) segments each raw demonstration into subtask boundaries, names each subtask, and extracts pre/post conditions from the trace context.
2. **Store as a library.** Subtasks are indexed by their descriptions and preconditions. The same "attach a file to an email" subtask can serve many tasks that need it.
3. **Retrieve and compose at runtime.** Given a new task, the agent retrieves candidate workflows, checks which subtasks are already satisfied in the current state (skip), which need adaptation (substitute), and executes.
4. **Closed-loop data engine.** Successful runs are re-ingested and refine the workflow library; failed runs generate hypotheses for new or edited subtasks. The library grows in the direction of what the agent actually needs.

## Why it matters

- Raw demonstrations don't compose. Subtasks do. This is the difference between a demonstration corpus and a *library* of reusable primitives.
- The retrieved workflow acts as a strong plan prior. It converts an open-ended agentic task into a much narrower "adapt this plan to the current state" problem, which VLM-based GUI agents handle far more reliably.
- Powers the UI-Mate result: 77.0% OSWorld-Verified and 66.2% WindowsAgentArena at 27B open-weight, with the workflow representation doing much of the heavy lifting.

## Gotchas & tricks

- **Boundary detection is the whole game.** If the conversion pipeline segments a demonstration at the wrong grain (too coarse → subtasks are opaque; too fine → nothing composes), the library never gets useful.
- **Preconditions must be lightweight.** If checking whether subtask *k* is already satisfied costs another VLM call, the retrieval overhead eats the win.
- **Overlap without contradiction.** Two workflows for the same task may segment differently. The library needs conflict detection or the agent starts thrashing between plans mid-run.
- **Not a replacement for grounding.** Subtasks say *what* to do; the model still needs to ground *which button, which field* against the current screen. Both are required.

## Sources

- Paper: *UI-Mate: Advancing Open-Weight Foundation GUI Agents with In-Context Demonstrations* — Ding et al. — arXiv:2608.15930 — 2026.
- Related concept: hierarchical task networks (classical planning) — long-standing precursor of subtask decomposition as a plan representation.
