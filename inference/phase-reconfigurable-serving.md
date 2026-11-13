# Phase-reconfigurable serving
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A design for LLM serving clusters where each node's *phase* (prefill vs. decode) is decoupled from its *KV storage*, so a node can switch between phases online without restarting and without losing resident KV. On top of that, a workload-aware planner sizes the prefill/decode split per minute and a KV manager uses agent-emitted lifetime hints for routing and eviction. First concretely demonstrated by *Evolving Infra* (2611.12304), which reports up to 4.1× mean end-to-end latency and 14× P95 TTFT improvement over NVIDIA Dynamo on agent workloads.

**Prereqs:** [README.md](README.md)
**Related:** [../systems/README.md](../systems/README.md), [../agents/README.md](../agents/README.md)

---

## What it is

Prefill-decode (P/D) disaggregation splits serving into a prefill tier (compute-bound, throughput-oriented) and a decode tier (memory-bound, latency-oriented). Classical disagg pins each node to a tier at deploy time. That assumption breaks under agent traffic: tool-call bursts spike prefill demand, long CoT tails spike decode demand, and the pin leaves one tier idle while the other saturates.

Phase-reconfigurable serving removes the pin.

## How it works

Three decoupled components:

1. **KV storage service.** KV cache lives on a storage layer addressable by key (session, prefix hash). Compute nodes attach to KV, they don't own it. Switching phase does not lose the KV.
2. **Phase scheduler.** Measures the current prefill/decode demand ratio and chooses, per short window, how many nodes should serve each phase. Nodes flip phase on command.
3. **KV manager.** Routes each incoming request to the compute node where the useful KV already lives, and uses signals from the agent framework (context trim, tool-call boundary, session end) to decide what to evict.

The result is an elastic pool: the same cluster handles bursty short agent calls and long reasoning tails without over-provisioning either side.

## Why it matters

Classical disagg was built for chat: one model, two phases, predictable token-length distribution. Agent traffic has tool-call bursts, cache-heavy prefill on growing contexts, and decode tails that depend on how much the agent reasons. Phase-reconfigurable serving is the first design to systematically treat those patterns as first-class, and the latency numbers suggest the gap over pinned disagg is big enough to justify another serving-stack rewrite.

## Gotchas & tricks

- Phase-switch cost is dominated by model-weights swap, not KV — keep weights resident on every node.
- KV eviction needs good hints from the agent layer; a generic LRU under-evicts cached tool outputs and over-evicts active sessions.
- Scheduler window matters: too long and bursts outrun the plan; too short and the plan oscillates.
- Works best when prefill and decode share the same model; cross-model elasticity requires a broader runtime.

## Sources

- Paper: *Evolving Infra: Agent-Adaptive LLM Serving at Scale* — Xu, Li, Du, Li, Xia, Bian, Krishna, Mao, Stoica, Han (2026) — [arXiv:2611.12304](https://arxiv.org/abs/2611.12304)
