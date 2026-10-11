# Evidence-Grounded Behavior Graph (EBG)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free monitoring protocol for long-horizon coding agents. EBG organizes an agent's trace (tool calls, code diffs, intermediate outputs) into a graph whose nodes are *source-linked evidence* and whose edges express causal/temporal relations. An oversight LLM answers behavior-vs-requirement questions by citing graph nodes, so its claims are debugable rather than vibes-based. Introduced alongside the **AgentMonBench** benchmark (2026); improves decision identification and evidence localization across eight oversight models.

**Prereqs:** [cot-monitoring.md](cot-monitoring.md), [safety-case.md](safety-case.md)
**Related:** [_scheming.md](_scheming.md) · [../agents/README.md](../agents/README.md) · [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Monitoring an autonomous coding agent means answering two questions: *did the agent do what we asked*, and *did it make a consequential decision outside its mandate*. Direct LLM-judge monitors answer them with free-form text, which is hard to audit, hard to disagree with, and easy to game. EBG converts the monitor's input into a graph of cite-able evidence, then forces the monitor to ground its claims in named nodes.

## How it works

- **Evidence extraction.** Walk the agent's trace and tag each candidate decision point (file edit, tool call, test outcome). Each tag carries a source link (file:line, tool name, timestamp).
- **Graph construction.** Nodes = tagged evidence; edges = causal or temporal relations (code A caused test B to pass; message C triggered tool call D). The graph is a lossy but structured view of the trace.
- **Oversight queries.** The oversight LLM is asked structured questions ("did the agent change the data-retention policy?", "did any tool call touch production?") and must cite node IDs that support its answer.
- **Benchmark.** AgentMonBench runs these questions against curated software-engineering agent traces where ground truth is known. Scoring measures both *decision identification* (did the monitor catch the right event) and *evidence localization* (did it cite the right nodes).

## Why it matters

- Oversight is the load-bearing safety mechanism for autonomous coding agents.
- Grounding monitor claims in graph nodes makes disagreement structured — a reviewer can point to the node the monitor relied on and say "this doesn't support that claim" — raising the ceiling of what weak monitors can enforce on stronger agents.
- Training-free: no model updates; works as a *protocol* layered over any judge LLM.

## Gotchas & tricks

- Graph construction quality dominates. Over-chunked traces produce noisy graphs; under-chunked ones hide the signal. The paper's extractors are tuned to software-engineering workflows.
- The monitor can still produce *plausible-but-wrong* citations if the graph is sparse. Spot-check sampling of monitor outputs is prudent.
- Scales to the trace sizes shown in AgentMonBench; very long sessions may need hierarchical graphs with summary nodes.

## Sources

- Paper: *What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents* — Sun, Yan, Zhao et al., Renmin University of China (GSAI-IIRLab), 2026 — [arXiv:2610.06406](https://arxiv.org/abs/2610.06406).
