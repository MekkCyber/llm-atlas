# Experiential-Working Memory (Recuris)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-horizon LLM agents drown in their own histories: growing context obscures the current task state and misroutes skill invocation. **Recuris** splits agent memory into a **Working Memory** (task state + skill dispatcher, small and current) and an **Experiential Memory** (accumulated skills, indexed for retrieval). A *fixed* **Meta-Agent** turns execution evidence into validation-gated updates to a **Skill Memory**. Because the Meta-Agent never changes, the recursive memory-evolution loop is bounded and stable. Introduced by Yu et al. 2026.

**Prereqs:** None (agents-cluster fundamentals)
**Related:** [_recursive-self-improvement.md](_recursive-self-improvement.md), [meta-n.md](meta-n.md)

---

## What it is

An agent solving a 100-step task accumulates ~100 turns of interaction context by the end. Common failure modes:

- **State occlusion** — the current task state (what's already been tried, what's still open) is buried inside a huge scroll of past turns.
- **Skill misinvocation** — when the model needs to decide "which of my known skills fits this situation?", it must scan the whole history rather than just the current state.
- **Diffuse failure signals** — when the task fails, the failure could be from any of the ~100 turns; localizing the responsible skill is hard.

Recuris addresses all three by **imposing structure on memory** and reserving self-modification to a fixed layer.

## How it works

### Three memory buffers with distinct jobs

1. **Working Memory (WM)** — small, task-state-only. Tracks what has been tried, what is still open, what the goal is. Fits in a short context window.
2. **Experiential Memory (EM)** — accumulated skills from past tasks, indexed for retrieval. Not scanned linearly; queried by a skill dispatcher using the WM's task state.
3. **Skill Memory (SM)** — the writable subset of EM. New or refined skills land here.

Skill selection uses *only* WM as the retrieval query. The full history never enters the skill-selection loop.

### Structured evidence, not free-form logs

Every execution step is logged as **structured evidence** attributing outcomes to specific memory components: "this skill failed because its preconditions weren't checked", "this WM update was stale", etc. When the task fails, the evidence localizes the responsible memory component rather than yielding a blob-of-context to introspect.

### Fixed Meta-Agent, validation-gated updates

A **Meta-Agent** with a fixed prompt reads structured evidence and proposes updates to Skill Memory. Because the Meta-Agent's own logic is fixed, its behavior across recursive iterations is deterministic; only what it *reads* (evidence) and *writes* (SM patches) changes.

Every Meta-Agent proposal passes through **validation** on held-out task instances before landing in SM. Bad patches don't stick.

### The bounded recursion

```
loop over tasks:
    solve(task) with (WM, EM)   →   evidence
    proposal = MetaAgent(evidence)     # fixed logic
    if validate(proposal) improves task success:
        SM ← apply(SM, proposal)
        EM ← EM ∪ SM
```

Because the Meta-Agent is fixed and updates are validation-gated, the loop cannot destabilize the system — the bounded RSI property Recuris trades for.

## Why it matters

- **Concrete gains at scale.** +17.8 to GPT-5.6 Sol on τ-bench, +15.6 to Claude Opus 5 (to 87.9%), +32.2 on the longest tasks. Wins in **35 of 37** model-benchmark pairs.
- **Names the stability constraint of RSI.** Recuris shares its central design principle with Meta^n: for recursive self-improvement to stay stable, some part of the meta-layer must be fixed. Recuris fixes the Meta-Agent; [meta-n.md](meta-n.md) fixes the meta-operator Ω. Different layers, same architectural lesson.
- **Failure localization gives the debugger a purchase.** Structured evidence turns "the agent failed somewhere in these 100 turns" into "the failure attributes to skill X's precondition check", making both automated update and human review possible.

## Gotchas & tricks

- **Working Memory summarization is load-bearing.** WM must be kept small enough to fit and current enough to route skills correctly. Aggressive summarization can drop signals; permissive summarization loses the "small enough" property. Recuris uses task-state schemas rather than free-text summaries — a lesson worth stealing.
- **Skill-selection retrieval is the bottleneck at scale.** As EM grows to thousands of skills, semantic retrieval becomes the latency-critical path. Consider hierarchical retrieval or task-typed indexes.
- **Fixed Meta-Agent still needs cadence tuning.** Even a fixed Meta-Agent applied every task creates churn; running it every N tasks (or only after M consecutive failures on similar tasks) reduces validation cost.
- **Evidence schema is what you're really designing.** The Meta-Agent can only diagnose what the evidence attributes; missing attribution fields silently limit what can be learned.
- **Skill-selection failures look like skill failures.** When the wrong skill is invoked, the failure attributes to the invoked skill — but the fix belongs upstream in the dispatcher. Distinguish these cleanly in evidence categories.

## Sources

- Paper: *Recursive Experiential-Working Memory Evolution for Long-Horizon Agent Harnesses* — Yu et al., 2026 — introduces Recuris. [arXiv:2608.24876](https://arxiv.org/abs/2608.24876).
- Related: *Meta^n* (Kim et al., 2026) — the same fixed-meta-layer principle applied at strategy-composition level rather than memory management.
- Related: *Voyager* (Wang et al., 2023) — early skill-library approach in a Minecraft agent, precursor to modern Skill Memory patterns.
