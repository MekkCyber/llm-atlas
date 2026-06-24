# PlanBench-XL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An interactive benchmark of **327 retail tasks over 1,665 tools** that evaluates LLM tool-use agents under *retrieval-limited tool visibility* — the agent never sees the full tool catalogue at once and must iteratively retrieve tools, invoke them to surface intermediate evidence, and use that evidence to find the next useful tool. Targets the failure mode where agents *plan adaptively across multiple retrieval rounds*, not just pick the right tool from a static list.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md), [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

A tool-use evaluation that separates two axes most agent benchmarks conflate:

- **Tool availability** — every tool in the 1,665-tool ecosystem can be called.
- **Tool visibility** — only a small subset is shown to the agent at any moment.

The agent has to issue retrieval queries against the catalogue (often constructed from evidence surfaced by earlier tool calls) to bring the *next* useful tool into its visible window. Tasks are long-horizon retail workflows: the goal is reached only after several rounds of `retrieve → invoke → derive evidence → re-retrieve`.

## How it works

For each task:

1. The agent receives the user's goal plus a small starting tool window.
2. The agent must issue retrieval calls into the 1,665-tool catalogue; results re-populate the visible window.
3. Tool invocations return outputs that often *unlock* the queries needed for the next retrieval round (e.g., an order-id retrieved by one tool is needed to retrieve the right shipping tool).
4. Success is measured by goal completion under varying levels of *blocking* — random removal of tools from the candidate set to simulate stale catalogues, deprecations, or partial outages.

The benchmark reports performance at multiple blocking severities, exposing how much of an agent's apparent capability comes from a near-complete tool view vs. true adaptive planning.

## Why it matters

Most production agent stacks already operate over tool ecosystems too large to inline in the prompt. They rely on a retriever to surface ~10 candidate tools per step. PlanBench-XL is one of the first benchmarks where the *retriever–planner loop* is the unit under test, rather than tool selection from a fixed schema list. The headline finding — GPT-5.4 drops from 51.9% to 11.4% under severe blocking — quantifies how fragile current "long-horizon planning" claims are once tool visibility is realistic.

## Gotchas & tricks

- The benchmark is **agent-policy aware**: an agent that pre-fetches all tools defeats the point. Evaluation harnesses enforce per-step visibility caps.
- Blocking severities are *task-conditional*: a task is blocked if a tool on its critical path is excluded, not at random across the catalogue.
- Reported scores collapse "first-pass" and "after-recovery" into one number — the paper recommends also tracking *recovery rate* (tasks where the first plan failed but a re-plan succeeded).
- Only the retail domain is covered in v1; transfer to coding / OS / browser agents is open.

## Sources

- Paper: *PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems* — Liu, Lin, Qian et al., UIUC, 2026 — [arXiv:2606.22388](https://arxiv.org/abs/2606.22388).
