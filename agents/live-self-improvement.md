# Live Self-Improvement for Long-Horizon Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Most agent self-improvement loops run *between* episodes: log the trace, mine lessons, update the skill library, then ship a new version. **Live self-improvement** (PILOT) collapses that loop into the active run — a supervisor process reads the worker agent's stream in real time, redirects it when it detects wasted budget or repeated errors, and simultaneously compiles new procedural know-how into reusable skills that the worker can pick up *within the same run*.

**Prereqs:** [README.md](README.md), [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)

---

## What it is

A **supervisor-worker** architecture for long-horizon agents. The worker executes the task; the supervisor watches its trace as it streams and can:

- **Redirect** the worker (inject guidance, change tools, prune a failing subplan).
- **Distill** newly discovered procedures into named skills and register them into the worker's tool namespace, immediately usable.

Both actions happen inside a live episode — the supervisor is not a post-mortem, it's a co-pilot.

## How it works

Two loops sharing state:

1. **Live-steering loop.** Supervisor consumes the worker's tool calls and intermediate reasoning as they emit. Triggers: (a) rollouts repeating a subtask that has already failed similarly; (b) budget-share estimates exceeding thresholds for the current subgoal; (c) supervisor's own model flagging a plan-quality problem. When triggered, the supervisor injects a message into the worker's context (a corrected plan, a new tool binding, a discouragement of the failing path) and lets execution resume.
2. **Skill-distillation loop.** When a subtask completes successfully via a novel procedure, the supervisor compresses the trace of that subtask into a **skill**: a named function-like artifact with a short prompt-level docstring and the successful trace as an executable template. The skill is added to the worker's tool namespace on the fly; future subtasks in the *same* run can call it directly instead of re-deriving it.

Both loops leverage the fact that streaming traces expose intent early enough to act on — waiting for episode end throws away most of the leverage.

## Why it matters

Long-horizon agent runs waste huge token budgets rediscovering the same subprocedures. Substantial performance gains across long-horizon benchmarks with **reduced token usage** — the supervisor pays for itself by preventing wasted rollouts, and skill compression amortizes across subgoals inside the run. Closes the gap between "agent learns from experience" (usually across-run) and "agent learns from *this* experience, now."

## Gotchas & tricks

- **Supervisor thrash.** Over-eager redirection destabilizes the worker; a debounce (only intervene after a minimum number of confirming signals) matters.
- **Skill namespace pollution.** Compress skills aggressively — one long trace becomes one succinct skill, not many. Otherwise the worker's tool list explodes and prompt formatting eats the gains.
- **Skill provenance.** Track which supervisor decision produced which skill; unused or harmful skills need to be garbage-collected between runs.
- **Latency cost.** Supervisor inference sits on the worker's critical path if implemented naively; async supervision with occasional worker checkpoints is the practical shape.

## Sources

- Paper: *PILOT in the Loop: Live Self-Improvement for Long-Horizon Agents* — Xiao et al., 2026 — [arXiv:2608.26530](https://arxiv.org/abs/2608.26530)
