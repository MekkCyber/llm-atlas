# VisSearch-Bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for long-horizon *visual* search in multimodal agents — tasks that require interleaving visual grounding with external web/tool lookups across many turns. Unlike text-only agent benchmarks that are approaching saturation, VisSearch-Bench is wide open: even the strongest multimodal agent fully solves only **26.47%** of tasks, and the dominant failure modes expose concrete training targets. Introduced in 2611.04894 (Ant Group).

**Prereqs:** none
**Related:** [benchmark-compression](benchmark-compression.md)

---

## What it is

A task suite where the agent must:
1. Receive a visually-grounded query (image + question, possibly with regions of interest).
2. Perform multiple rounds of visual retrieval and external lookup (web search, image search, tool calls).
3. Cross-reference evidence across turns and maintain partial findings.
4. Return an answer requiring joint visual + factual reasoning.

Tasks are constructed so that neither (OCR + text search) nor (pure VLM reasoning without retrieval) is sufficient — the visual channel must drive planning *and* the retrieval must integrate external facts the VLM doesn't know.

## How it works

Benchmark construction:
- **Long-horizon.** Each task requires planning across many turns; the median solution trajectory uses multiple tool calls.
- **Visually anchored.** The query ties the agent to specific visual content that cannot be adequately summarized in text.
- **External-grounded.** Answers depend on information not inside the model's weights, forcing genuine retrieval.
- **Reference answers + partial credit scoring.** Agents can earn partial credit for correct intermediate findings.

Evaluation: agents are scored on completion rate under a fixed harness (ReAct + tool access), with secondary metrics for tool-call efficiency and visual-grounding stability across turns.

## Why it matters

- **Agent eval has been text-heavy.** Visual-channel bottlenecks have been under-measured despite being a known weakness.
- **Natural RL target.** The three failure modes the paper identifies — planning horizon, visual grounding stability, tool-use integration — are exactly the knobs a multimodal agent RL recipe would optimize.
- **Headroom.** A 26.47% ceiling means multiple orders-of-magnitude improvement are possible before saturation.

## Gotchas & tricks

- Completion-rate metrics punish agents that make correct intermediate progress but fail the final step. Use partial-credit scores for diagnosis.
- Harness choice affects results a lot — report results under the standard harness, then secondarily under the agent's native harness.
- Visual-grounding stability degrades as turns increase; this is the dominant failure mode and worth isolating with per-turn drift metrics.

## Sources

- Paper: VisSearch-Bench: Evaluating Long Horizon Visual Search in Multimodal Agents — Wu et al., 2026 — [arXiv:2611.04894](https://arxiv.org/abs/2611.04894)
