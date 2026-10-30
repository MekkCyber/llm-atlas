# Agent Serving
*Depth — the inference-stack considerations for LLM workloads where sessions interleave model decoding with external tool calls.*

**TL;DR:** Serving stacks like vLLM and SGLang were designed around **chat sessions**: a single prompt, a continuous stream of generated tokens, no external pauses. **Agent sessions** break every assumption behind that model: long idle windows during tool execution, bursty return traffic, strong KV-cache locality across turns, and multi-step dependencies between calls. **Agent serving** is the sub-discipline that redesigns scheduling, batching, cache management, and placement for this profile. TAO (Li et al., 2026) is a concrete recent example: a joint scheduler using tool-return readiness, time-decayed KV cache credit, and token footprint reports **26.8% lower latency, 36.3% lower TTFT, 29.5% higher decode throughput** vs. the best baseline on each metric.

**Prereqs:** *(none)*
**Related:** [tool-aware-scheduling.md](tool-aware-scheduling.md), [../systems/ray.md](../systems/ray.md), [../agents/computer-use-agents.md](../agents/computer-use-agents.md)

---

## What it is

A session's lifecycle in an agent workload looks like:

```
decode ─► tool call (external) ─► decode ─► tool call ─► … ─► final response
   ▲                                                             │
   └──────────────────────── KV cache stays warm ────────────────┘
```

Three properties the chat-serving stack doesn't optimize for:

- **Long idle windows.** Between a tool call and its return, the session holds KV cache but produces no tokens. Minutes of idle are common (browser agents, batch tools).
- **KV-cache locality.** The next decode step after a tool return will reuse most of the KV cache that was built up before. Evicting mid-session is expensive.
- **Readiness skew.** Multiple concurrent agent sessions have different "time to next decode" distributions. Admitting them in arrival order is suboptimal; admitting by readiness is better.

Agent serving puts these three facts at the center of the scheduler.

---

## How it works

### Session states

A new state machine replaces chat's "prefill / decode":

```
WAITING_TOOL ─► READY ─► DECODING ─► WAITING_TOOL ─► …
```

- **WAITING_TOOL** sessions are not on GPU for compute; they consume KV cache budget.
- **READY** sessions have a tool return queued and want a decode slot.
- **DECODING** sessions are actively generating.

### Scheduling signals

Modern agent schedulers consider at least three signals when picking what to admit:

1. **Tool-return readiness** — probability or ETA that the session's next tool call is about to return. Avoid preempting a session that's about to need the GPU.
2. **KV-cache locality** — per-node KV cache credit, time-decayed. Prefer placing a session on a node that still has its warm cache.
3. **Token footprint** — current context length and expected generation length. Keeps the admit-set within capacity constraints.

TAO combines all three into a single score and runs a dynamic program over waiting sessions each step.

### Cache management

Options for sessions in WAITING_TOOL:

- **Hold in HBM.** Lowest latency on resume, highest cost.
- **Spill to host / disk.** Lower cost, higher resume latency.
- **Evict and re-prefill on return.** Trades cache spend for compute spend on resume.

Choice depends on expected idle duration and cache cost; it is a per-session decision in modern stacks.

### Placement

Multi-GPU / multi-node serving adds a placement dimension: which GPU should take a resuming session? A locality-aware scheduler keeps the session on the node whose KV cache still holds its context, falling back to re-prefill elsewhere only when the first choice is saturated.

---

## Why it matters

- **Agents are the dominant workload trend.** From 2025 onward, tool-using agent traffic is a growing share of LLM-API load. Serving stacks optimized for chat leave double-digit throughput and latency on the table on agent traffic.
- **Headline numbers from TAO.** Up to **26.8% lower latency**, up to **36.3% lower TTFT**, up to **29.5% higher decode throughput** vs. per-metric best baselines across 12 (model, workload, concurrency) settings — purely from scheduler and placement changes.
- **Shifts work into the scheduler, not into model config.** No model changes, no new kernels. The win comes from treating tool-return readiness as a first-class scheduling signal.

---

## Gotchas & tricks

- **Tool-return ETA is noisy.** Browser latency, external-API jitter, long-tail dependencies. The ETA estimator has to be probabilistic, not point-estimate.
- **Prefix caching cuts both ways.** Shared system prompts across agent sessions make prefix caching huge; per-session KV then dominates. Balance pool sizes between prefix caches (shared) and per-session caches (private).
- **Starvation under locality bias.** A pure locality-first policy can starve sessions whose warm node is overloaded. Use time-decay on cache credit so stale affinity decays.
- **Pre-emption is expensive for long sessions.** Re-prefilling a 32K-context session on preemption costs seconds. Treat preemption as a last resort, not a load-balancing default.
- **Benchmarks still mostly chat-shaped.** Published serving benchmarks underweight agent traffic. Measure on your own agent traces, not on sharegpt replay, before deciding what to deploy.

---

## Sources

- Paper: *TAO: Tool-Aware Joint Scheduling for LLM-Based Agentic Serving* — Li et al., 2026, [arXiv 2610.37230](https://arxiv.org/abs/2610.37230) — the recent canonical scheduler for this workload.
- Related: *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM) — Kwon et al., 2023 — chat-era baseline.
- Related: *SGLang* — Zheng et al., 2024 — introduces RadixAttention for prefix cache; a step toward agent-aware serving.
- Related: [../systems/ray.md](../systems/ray.md) — the orchestration layer agent-serving stacks typically run on.
