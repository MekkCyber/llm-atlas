# Think-and-Read
*Depth — an interleaved reasoning / document-consultation control flow for long-context RL agents.*

**TL;DR:** On long-context reasoning tasks, a one-shot "read everything then answer" prompt wastes attention on irrelevant spans and chokes on documents past the native context window. **Think-and-Read** structures the trajectory as alternating **reason** turns and **read** turns: the agent decides when to consult the document, which query to issue, and when it has enough to answer. Paired with a dense per-read utility reward, this becomes the backbone of LongR and similar long-context RL recipes.

**Prereqs:** [../_rl.md](../_rl.md), [long-cot-rl.md](./long-cot-rl.md)
**Related:** [contextual-density-reward.md](./contextual-density-reward.md) · [../grpo.md](../grpo.md) · [../rlvr.md](../rlvr.md)

---

## What it is

A control-flow template for retrieval-aware reasoning. Instead of concatenating the full document into the prompt or doing a fixed RAG pre-fetch, the agent interleaves two action types:

- **Think** — emit free-form reasoning tokens.
- **Read** — emit a structured query that returns a span of the document.

The model chooses the ordering, number, and content of reads. A terminal **answer** turn closes the trajectory.

## How it works

**Action grammar.** The vocabulary is extended with sentinel tokens (`<think>`, `<read query="…">`, `<answer>`), or the control flow is implemented at the harness level with the model producing structured outputs. On a read, the harness runs retrieval (dense, sparse, or a learned retriever) and splices the returned span back into context.

**Policy objective.** Train the policy end-to-end with RL so that it:
1. Decides *when* to read vs continue thinking.
2. Decides *what query* to issue (phrasing, specificity).
3. Decides *when* to answer.

Each read is scored for utility (see [contextual-density-reward.md](./contextual-density-reward.md)) and the final answer is scored for correctness.

**Rewards shape behavior.** A dense per-read reward keeps the policy from reading too little; a per-read cost or length penalty keeps it from reading too much. The combination produces a policy that reads a *necessary and sufficient* set of spans.

## Why it matters

- **Scalable to documents past the native context window.** The policy only pulls in the spans it needs; the full document never has to fit.
- **Composable with retrieval.** The read turn is agnostic to the retriever; swap in dense, sparse, or hybrid.
- **Measurable gains across benchmarks.** As used in LongR: +9% on LongBench v2, consistent gains on RULER and InfiniteBench. Transfers across RL algorithms.
- **Grounded reasoning.** By tying every claim to a consulted span, Think-and-Read reduces hallucinations compared to full-context stuffing.

## Gotchas & tricks

- **Reader-side churn.** Model-emitted queries can degenerate (always the same query, or copy-paste of the question). A diversity bonus on queries helps.
- **Harness latency.** Each read introduces a round-trip to the retriever. For long trajectories this adds up; batch reads where possible.
- **Query ↔ span alignment.** If the retriever is weak, the policy's good queries return bad spans and the utility signal collapses. Co-train or at least co-tune the retriever.
- **End condition.** Without a clear "enough" signal (a terminal correctness reward or a budget), policies will read indefinitely. Enforce a hard read-budget during RL.
- **Compared to full-context attention.** For documents that fit in the model's native window, full-context stuffing can be competitive on accuracy — Think-and-Read wins on efficiency and when the document doesn't fit.

## Sources

- Paper: *LongR: Unleashing Long-Context Reasoning via Reinforcement Learning with Dense Utility Rewards* — Yu, Chen, Wu, Chen, Yu, Hui, Ping, Yan, Chang — SJTU / PKU / THU / BUPT, 2026 — arXiv:2602.05758.
