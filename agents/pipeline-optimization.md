# Pipeline Optimization (Coding-Agent-Driven)
*Depth — autonomous optimization of multi-step LLM pipelines using a coding agent.*

**TL;DR:** A multi-step LLM pipeline (retrieval → reasoning → formatting) is exposed to a coding agent inside a standardized codebase. The agent runs the pipeline on a benchmark, inspects intermediate traces, attributes failures to specific steps, proposes **prompt edits first** and **structural edits only when prompt edits cannot resolve the bottleneck**. FAPO (Saglam et al., Cisco Foundation AI + Yale, 2026) is the canonical recipe. Beats prompt-only optimizers (GEPA, OPRO) by +14.1 pp on average and +33.8 pp on tasks that genuinely require structural changes.

**Prereqs:** *(none)*
**Related:** [README.md](README.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Prompt optimization frameworks (DSPy/GEPA, OPRO, APE) search over **prompts only** while the pipeline structure is held fixed. This is fine for single-call LLM tasks, but multi-step pipelines fail through interactions across steps — a bug in the retrieval ranker can't be papered over by tweaking the reasoning prompt.

Pipeline optimization gives a coding agent **access to the pipeline source code**, evaluator harness, and per-step traces, and treats optimization as a code-editing task. The agent's policy is:

1. Run the pipeline on a benchmark slice, log per-step inputs/outputs.
2. Read failing traces, hypothesize which step caused each failure.
3. Propose a **scoped edit** — prompt rewrite, retrieval-k change, formatter swap.
4. Validate the variant; revert if it regresses.
5. **Escalate to structural change** (replace a step, add a step, remove a step) only when attribution evidence shows prompts can't fix it.

The frontier coding agent (Claude Code in the FAPO paper) is competent enough to drive this loop end-to-end on standardized codebases.

## How it works

### Standardized codebase

The pipeline is wrapped in a thin harness with:

- **Entry point** `run_pipeline(input) -> output`.
- **Step-level logging** at every LLM call and tool invocation.
- **Evaluator** that scores a single example or a batch.
- A **diff sandbox** so variants can be tried and reverted.

### Attribution rule

Each failure is attributed by **localized intervention**: replace step $i$'s output with a known-good gold value and re-run downstream. If the pipeline now succeeds, step $i$ is the bottleneck. If it still fails, the bottleneck is downstream.

### Edit hierarchy

| Priority | Edit | Cost |
|---|---|---|
| 1 | Prompt rewrite | minutes, low risk |
| 2 | Schema / few-shot example update | low risk |
| 3 | Retrieval k / ranker swap | medium |
| 4 | Add/remove a pipeline step | high — only if 1–3 exhausted |
| 5 | Replace LLM backbone of a step | only as last resort |

The agent climbs this hierarchy lazily, with a held-out validation pass after each step to gate against overfitting to the benchmark slice.

### Stopping criterion

The agent stops when (a) the held-out score plateaus for $N$ rounds, or (b) a per-step compute budget is exhausted. The chronological log of edits + score deltas is the artifact returned to the user.

## Why it matters

- Prompt optimizers have a known ceiling on multi-step pipelines. Pipeline optimization breaks through it.
- **+14.1 pp** average and **+33.8 pp** on restructuring-heavy benchmarks vs GEPA across 18 model × benchmark combinations.
- Demonstrates that current frontier coding agents are competent enough to drive non-trivial system-level optimization autonomously — relevant to anyone building agent-built-agent workflows.

## Gotchas & tricks

- **Held-out gating is critical.** Without it the agent overfits to the benchmark slice within a handful of rounds.
- **Edit budgets matter.** Without a budget, the agent will keep trying structural variants well past diminishing returns.
- **Attribution is fragile under stochasticity.** Sample several traces before declaring step $i$ the bottleneck — a one-trace failure may be sampling noise.
- **Backbone-swap escalation is dangerous.** Letting the agent swap a 7B step for a 70B step is a real cost increase; require explicit human authorization before allowing this level.
- **Reproducibility.** Pipeline-optimized prompts depend on the optimizing agent's specific failure-mode hypotheses; a different agent or different seed may produce a different pipeline at similar score. Log the optimization trace, not just the final pipeline.

## Sources

- Paper: *FAPO: Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines* — Saglam, Zhao, Nelson, Vijay, Priyanshu, Karbasi (Foundation AI–Cisco, Yale), 2026, arXiv 2606.19605.
- Paper: *GEPA: Genetic-Pareto Prompt Optimizer* — prompt-only baseline FAPO compares against.
- Background: DSPy framework — standardizes pipeline interface FAPO-style optimizers target.
