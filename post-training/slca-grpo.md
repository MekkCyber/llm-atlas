# SLCA-GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Tool-calling agent rollouts interleave two structurally different token types — **structured tool invocations** and **natural-language summaries**. Vanilla GRPO broadcasts a single trajectory-level advantage to every token, so noisy summary-generation gradients contaminate the tool-decision tokens. SLCA-GRPO adds **Segment-Locked Credit Assignment**: separate advantage estimation per segment type inside one rollout group, and **Hierarchical Rewards** that route execution-quality signals to tool tokens and preference signals to summary tokens.

**Prereqs:** [grpo](./grpo.md), [rlvr](./rlvr.md)
**Related:** [_rl](./_rl.md), [_rewards](./_rewards.md)

---

## What it is

Standard GRPO computes one scalar advantage $A_i$ per response and broadcasts it to every token in that response. For math or code with a single verifiable outcome, that's fine — the "which tokens mattered" question is fuzzy anyway.

For tool-calling agents the picture is different. A typical rollout is:

```
<tool_call>{"name": "search", "args": {"q": "..."}}</tool_call>
<tool_response>...</tool_response>
<summary>Based on the search, ...</summary>
<tool_call>{"name": "calc", ...}</tool_call>
...
<final>...</final>
```

The tool-call tokens and the summary tokens are optimized for different things (correct schema + effective action vs. helpful natural language). Broadcasting one advantage means:

- A rollout with correct tool use but a lazy summary gets its whole tool-token gradient dragged down by the summary reward.
- A rollout with wrong tools but a polished summary reinforces the bad tool decisions.

This is *cross-segment credit misattribution*.

## How it works

Two additions to GRPO:

1. **Segment-Locked Credit Assignment (SLCA).** Within one rollout group, separate the tokens by segment type (`tool_call`, `summary`, `final`, …). Compute a **separate** group-relative advantage for each segment type using only the rewards relevant to that segment. No extra rollouts from intermediate states are needed — everything is computed from the same group of trajectories, just partitioned.

2. **Hierarchical Rewards (HierR).** Reward function returns a *vector* per rollout: execution rewards (tool-call correctness, schema validity, task success) and preference rewards (summary quality, formatting). SLCA routes each reward component to its corresponding segment's advantage.

Also introduces the **Schema-Guided LLM Simulator (SGLS)** — a simulator that emulates tool APIs for training so real API costs don't dominate RL rollouts.

## Why it matters

- **Direct fix to a real GRPO failure mode.** Nearly every tool-calling training paper published in 2025–2026 uses GRPO without addressing the segment-mixing problem.
- **Cost-neutral.** Uses the same rollout group as GRPO; only the advantage aggregation changes.
- **Measurable gains.** On a 7B backbone, SLCA-GRPO beats vanilla GRPO, ToolPO, and RLTR by +2.53 pp in-domain, +1.36 pp on BFCL, and +9.15 pp on τ²-Bench — with fewer tool calls per solved task.

## Gotchas & tricks

- **Segment boundaries must be exact.** Any tokenizer/parser drift misroutes advantages; canonicalize with the same parser you use for rewards.
- **Empty segments are common.** A rollout may skip the summary segment entirely; guard against divide-by-zero in the group-normalization step.
- **Compose with tool-response masking.** Standard practice masks the environment-supplied tool-response tokens from the loss; SLCA is orthogonal.
- **Scales down cleanly.** Works on small teacher-simulator setups (SGLS) before graduating to live APIs.

## Sources

- Paper: *SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL* — Zhan, Liu, Liu, Shi, Xu, et al., 2026 — https://arxiv.org/abs/2609.29050
- Related: [grpo](./grpo.md) — the vanilla algorithm this extends.
- Related: *ToolPO* / *RLTR* — earlier RL-for-tool-use baselines the paper compares against.
