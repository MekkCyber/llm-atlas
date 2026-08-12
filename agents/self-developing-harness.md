# Self-Developing Agent Harness (Ouroboros)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Hold the model constant and treat the agent's **harness** — tools, prompts, context assembly, and core implementation — as the mutable object. The agent modifies its own harness through **reviewed commits** that then become the runtime for subsequent tasks. Two evolution modes: *recursive free evolution* (improvement is itself a task) and *experience-driven evolution* (ordinary work exposes rough edges that trigger structural changes). Ouroboros with an Opus 5 backbone scores 86.97% on Terminal-Bench 2.1 — the best reported number on the benchmark, downstream of the evolved harness rather than a new base model.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/on-policy-distillation.md](../post-training/on-policy-distillation.md), [../evaluation/livecodebench.md](../evaluation/livecodebench.md)

---

## What it is

Agent quality is a product of *model × harness*. Almost all recent agent progress has come from improving the model side (bigger base, better post-training). The harness — the scaffolding of tools, prompts, memory, and context construction that wraps the base model — has been improving by hand.

Ouroboros is the first widely-cited demonstration that harness improvement can itself be automated *and* compute-scaled. The agent modifies its own scaffolding; each change goes through a code-review gate before it becomes the runtime for subsequent tasks. Self-modification with a rollback story.

## How it works

Two evolution modes running against a shared code-review gate:

- **Recursive free evolution.** "Improve the harness" is itself scheduled as a task. When completed, the resulting patch is reviewed, merged, and the agent may schedule the next evolution cycle recursively. Improvement is a first-class user of agent capability.
- **Experience-driven evolution.** During ordinary work (or social interaction with users), the agent notices bugs, rough edges, or inefficient context construction. These observations trigger patches to tools, prompts, or the context assembler. Ordinary tasks act as an unsupervised discovery process for harness weaknesses.

Every modification is a **commit** subject to review — either by a supervising process or another instance of the agent. Rejected commits are never merged into the runtime; approved commits become the code the next task runs against. The review gate is the safety mechanism: without it, self-modification can drift into unstable states with no rollback path.

## Why it matters

- **Harness improvement becomes compute-scalable.** Hand-tuning tools and prompts is the current default; Ouroboros converts that human labor into an amortizable RL-style loop.
- **Model-agnostic capability lift.** The 86.97% Terminal-Bench 2.1 result is on the same base model that scores substantially lower with a stock harness. The delta is entirely from the evolved scaffold.
- **Clean rollback story.** Reviewed-commit self-modification is much less scary than in-context self-modification: bad changes get rejected at review time, not after the agent has already acted on them.
- **Opens the "harness as first-class object" design space.** Pairs directly with harness-evolution benchmarks (Evo-Bench) that measure this new capability axis in isolation from base-model strength.

## Gotchas & tricks

- **The review gate is load-bearing.** Without it, recursive free evolution spirals: each iteration proposes changes that break subsequent iterations' assumptions. Every reported ablation collapses at some horizon.
- **Terminal-Bench 2.1 rewards long-horizon planning; general agent tasks may not.** The 86.97% number generalizes only insofar as other benchmarks reward the same kinds of harness improvements (better tool interfaces, better error recovery, richer memory).
- **Attribution of capability is tricky.** After many evolution cycles, it's hard to know which patches did what — the harness becomes an opaque artifact. Version-tag every commit and log the task that triggered it.
- **Requires an editable, versioned codebase.** Ouroboros only makes sense when the agent can commit against a real repo. Sandboxes without persistent state can't accumulate the effects.

## Sources

- Paper: *Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution* — Razzhigaev, Gritsaev, Kaznacheev, Dragunov, Yampolskiy, Kuznetsov, 2026 — Moscow State / AIRI / Skoltech / Joi Lab.
- Related benchmark: Evo-Bench (Huang et al., 2026) — yardstick for measuring harness-evolution capability in isolation.
