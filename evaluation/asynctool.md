# AsyncTool

*Depth — a benchmark for whether LLM agents can interleave concurrent tasks under realistic tool-response latency.*

**TL;DR:** Existing tool-use benchmarks assume zero-latency tools and a single active task. Real deployments have neither — tool calls have variable latency, and agents often handle several concurrent tasks. AsyncTool simulates tool delays and presents heterogeneous tasks concurrently, then measures whether the agent uses idle waiting time to make progress on parallel subtasks. Scores at step / sub-task / task levels with efficiency-aware metrics.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark for *asynchronous tool calling* in LLM agents — the capability to coordinate multiple in-flight tool calls and switch between tasks rather than blocking serially. Targets the gap between current tool-use evaluations (single-task, zero-latency assumption) and the latency-and-parallelism reality of production agents.

## How it works

The benchmark presents an agent with several heterogeneous tasks at once. Each task requires multiple tool calls, and the simulator inserts realistic per-call latency. The agent's score depends on:

1. **Correctness at three levels.**
   - *Step* — was each individual tool call correct?
   - *Sub-task* — were the sub-goals achieved?
   - *Task* — was the whole multi-task batch completed?
2. **Efficiency metrics.**
   - Task coordination — does the agent dispatch new calls while waiting?
   - Completion efficiency — total wall-clock time vs. lower bound from the dependency graph.

A hybrid data-evolution strategy is used to construct the dataset: seed cases are expanded to cover multiple scenarios and tool-use patterns, producing diverse multi-task instances.

## Why it matters

- Closes a known gap between agent benchmarks and deployment reality. Real production agents stall on tool latency unless they're explicitly trained to interleave calls.
- Defines a measurable target. The paper reports pronounced degradation across current agents under delayed feedback, with the agents that explicitly track inter-task dependencies coming out on top.
- Provides a substrate for training. The dataset can be used as the environment for RL training of async-capable agents, not just for offline evaluation.

## Gotchas & tricks

- Latency simulation is per-call randomized — same prompt produces different wall-clock numbers across runs. Report median + variance, not single runs.
- The "use idle time" metric can be gamed by speculative dispatch (fire calls you don't need). Score includes a wasted-call penalty.
- Tool semantics matter. A benchmark with cheap tools and expensive tools mixed reveals different failure modes than uniform-cost tools. AsyncTool covers multiple cost regimes.
- Compatible with both function-calling-trained agents and ReAct-style scratchpad agents; failure modes differ between the two.

## Sources

- Paper: *AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios* — Shi, Zhang, Huang, Nie, Fang, Wang, Chen, Chen, Chen, Zhao — USTC / U. Toronto, 2026 — [arXiv 2605.27995](https://arxiv.org/abs/2605.27995).
