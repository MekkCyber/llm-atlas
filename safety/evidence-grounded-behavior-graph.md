# Evidence-Grounded Behavior Graph (EBG)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free way to help human overseers audit long-horizon agent runs. EBG groups source-linked evidence (reads, writes, tool calls, inputs) into *behaviors*, organises those behaviors into a graph, and presents task-oriented views of the graph. On AgentMonBench (software-engineering oversight) across 8 models, EBG beats direct raw-context access at both decision identification and evidence localisation; gains persist across input scales and hyperparameters.

**Prereqs:** [README.md](./README.md), [cot-monitoring.md](./cot-monitoring.md)
**Related:** [../agents/README.md](../agents/README.md) · [../evaluation/README.md](../evaluation/README.md)

---

## What it is

A post-hoc monitor interface for autonomous agents: instead of handing the overseer the full execution trace, EBG constructs a structured graph where nodes are *behaviors* (coherent clusters of actions pursuing a sub-goal) and edges capture dependencies between them, each annotated with the raw evidence (file reads, tool outputs, prompts) that justifies it.

## How it works

- **Behavior extraction.** From the raw execution trace, cluster adjacent actions that share a sub-goal into named behaviors. Each behavior carries pointers back to the specific trace spans that produced it.
- **Graph assembly.** Draw edges for data flow (behavior A read what behavior B wrote), control flow (B was triggered by A's result), and semantic dependency.
- **Task-oriented views.** For a given monitor task (e.g. "did the agent exceed its authority?"), surface a projection of the graph: behaviors ranked by consequence, with their evidence chains expanded.

No training is involved — the construction uses the LLM as a structuring engine over the trace, nothing more.

## Why it matters

Scalable oversight requires the human (or automated reviewer) to find *consequential* decisions inside long traces fast. Raw transcripts bury them; EBG makes them findable. Across 8 models the structured graph beats raw context on both "did you spot the risky decision" and "did you locate the evidence that proves it," and the gains are stable across input lengths and hyperparameters — the authors' robustness check, not an isolated best case.

## Gotchas & tricks

- Behavior extraction quality depends on the LLM running the structuring step. A weak extractor produces a weak graph.
- The graph is a presentation layer, not a verifier. It helps monitors reason about the trace; it doesn't itself judge safety.
- Evidence links must be lossless — any paraphrase of the raw trace breaks accountability.
- Pairs naturally with the AgentMonBench benchmark; adopters should expect to tune the per-task view templates.

## Sources

- Paper: *What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents* — Sun et al., RUC GSAI, 2026 — [arXiv:2610.06406](https://arxiv.org/abs/2610.06406)
