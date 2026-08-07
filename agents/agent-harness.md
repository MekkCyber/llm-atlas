# Agent Harness

*Depth — the control-flow scaffolding that wraps an LLM agent, separate from the underlying model.*

**TL;DR:** A "harness" is the deterministic outer loop that turns an open-ended user request into a managed execution: it decomposes the task into bounded subtasks, maintains execution memory against context overflow, and verifies-and-repairs the final artifact before returning it. Distinct from the *model* (which does the reasoning) and from an *agent framework* (which is a library of harness pieces). OneDayAgent (Fang et al., 2026) is the canonical current example: same harness on 5 backend LLMs from 3 families, 0.821 on AgentIF-OneDay with GLM-5.2, no per-model tuning.

**Prereqs:** [../post-training/README.md](../post-training/README.md)
**Related:** [answer-backtracked-credit.md](answer-backtracked-credit.md) · [skill-kd.md](skill-kd.md) · [memory-staleness.md](memory-staleness.md)

---

## What it is

An LLM agent has three consistent failure modes on long-horizon tasks:

- **Goal drift** — the agent forgets or reinterprets the original request after many steps.
- **State loss** — earlier tool results fall out of context and can't be referenced.
- **Context overflow** — the transcript exceeds the model's window; naive truncation loses critical state.

A harness is the fixed outer program that jointly manages all three. It's the analogue of a browser + JS engine sitting under a JS program — the program (the model's reasoning) runs *inside* the harness, and the harness enforces invariants the program cannot enforce on itself.

## How it works

The OneDayAgent-style harness has three phases in a loop:

1. **Bounded subtask decomposition.** The high-level request is split into subtasks with explicit budgets (steps, tokens, tool calls). Each subtask is a short-horizon problem the model *can* handle without harness intervention.
2. **Execution memory management.** As subtasks complete, their outputs are summarized and stored in a structured working memory (not just the raw transcript). The main context always contains: the top-level goal, the current subtask, a compact memory index, and the last few observations. Old raw transcript is offloaded and re-fetched on demand.
3. **Verify-and-repair.** Before returning, the harness re-reads the top-level request and the produced artifact, checks constraints (format, coverage, cited-tool-results), and either loops back to fix specific gaps or emits.

The harness is model-agnostic: prompts, tool schemas, and step budgets are the same regardless of which LLM sits inside. Only the model's outputs change.

## Why it matters

- **Separates two axes of progress.** Model quality and harness quality can now improve independently — the same harness lifts every backend, the same backend improves under every harness.
- **Makes long-horizon evals meaningful.** Without a harness, benchmark scores blur "did the model reason well" with "did the framework not lose state". Fixing the harness lets a benchmark measure the model.
- **Cheap generalization.** OneDayAgent shows a single harness transfers zero-shot across model families — no per-backend tuning, just plug in the new model API.

## Gotchas & tricks

- **The harness can hide model weakness.** If verify-and-repair loops enough times, a weak model can pass a benchmark the *raw* model can't. Report harness-off scores too, or the eval measures the harness.
- **Subtask decomposition is itself a prompt problem** and the biggest source of variance between backends. Different models induce different execution styles under the same decomposer prompt — the harness generalizes, but the trajectories look different.
- **Memory summarization loses fine detail** (URLs, IDs, exact numbers). Explicit "verbatim" slots in the memory schema — not free-form summaries — are essential.
- **Verify-and-repair can loop forever** on impossible tasks. Hard iteration cap + graceful degrade to a partial answer.

## Sources

- Paper: *OneDayAgent: Towards a Long-Horizon Harness for Autonomous Agents* — Fang, Zhang, Gui, Chen, Zhang, 2026 — [arXiv 2608.05013](https://arxiv.org/abs/2608.05013). Introduces the "harness generalizes across backends" claim, evaluated on AgentIF-OneDay.
