# Prefill / Decode Disaggregation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Serve LLMs by running **prefill** (compute-bound, one shot per request) and **decode** (memory-bandwidth-bound, iterative) on *different* GPU pools connected by KV-cache handoff over a fast interconnect. Removes prefill-decode interference in continuous batching, lets each phase run on hardware sized for its actual bottleneck, and — critically for MoE — surfaces the prefill's routing decisions as a scheduling signal for decode.

**Prereqs:** [README](README.md)
**Related:** [moe-decode-routing](moe-decode-routing.md), [../architectures/_moe.md](../architectures/_moe.md), [../systems/ray.md](../systems/ray.md)

---

## What it is

A single LLM request has two very different phases:

- **Prefill**: encode the whole prompt in one attention/MLP pass, produce the first token and the full KV cache. Compute-bound (matmul-heavy).
- **Decode**: emit one token at a time, extending the KV cache. Memory-bandwidth-bound (weights + KV must stream from HBM every step).

Continuous batching (vLLM-style) mixes both phases in one running batch. That works but creates two structural problems: a long prefill in the batch stalls all decodes; and the whole cluster is sized for whichever phase is worse, wasting the other. **PD-disaggregation** runs prefill and decode on separate GPU pools and hands the KV cache from one to the other.

## How it works

```
              ┌───────────────┐    KV cache transfer     ┌───────────────┐
  request  ──▶│ Prefill pool  │──── (NVLink / IB)  ─────▶│ Decode pool   │──▶ stream
              │  (compute)    │                          │  (bandwidth)  │
              └───────────────┘                          └───────────────┘
```

Two pools, connected by a high-throughput interconnect:

- **Prefill pool** — sized for compute. Runs the initial forward pass; short-lived per request. Produces the KV cache for every layer.
- **Handoff** — the KV cache (or a compressed variant) is transferred to the decode pool. Prefix caching is layered on top so identical prefixes are reused across requests.
- **Decode pool** — sized for memory bandwidth / batching. Continues generation, adding one token per step, until EOS or a max length.

Scheduling decisions split into two layers: **prefill placement** (which prefill worker; usually low-order-bit balancing) and **decode placement** (which decode worker gets the handoff; the interesting knob, since decode workers accumulate expensive hot state — KV blocks, prefix caches, MoE experts).

## Why it matters

- **No prefill-decode interference.** One long prompt in the queue no longer stalls other users' streaming outputs.
- **Right-sized hardware.** Prefill can use fewer, higher-FLOP GPUs; decode can use more, higher-bandwidth GPUs. The cluster ratio becomes a workload-tunable knob.
- **Scheduling signal for MoE.** Prefill *already* computes routing decisions — expert activation histograms per request. That signal can direct decode placement (see [moe-decode-routing](moe-decode-routing.md)), which continuous batching can't do.
- **First-class prefix caching.** Because decode workers persist, prefix caches survive request boundaries.

## Gotchas & tricks

- KV-cache transfer bandwidth caps throughput; RDMA / NVLink between the pools is basically required at frontier scale.
- Handoff latency adds to TTFT (time-to-first-token). Overlap it with the last prefill layer, or accept a small p50 TTFT regression for the p99 improvement.
- The pool ratio is workload-dependent: chat-heavy (short prompts, long outputs) wants decode-heavy; RAG / agent workloads (long prompts, short outputs) want prefill-heavy.
- With MoE, decode worker specialisation is a real lever — locality-aware routing gains disappear if the scheduler treats decode workers as interchangeable.
- Fault handling: prefill loss is cheap (rerun), decode loss is expensive (KV cache gone). Sticky decode placement + KV replication is the usual answer.

## Sources

- Paper: *ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving* — Cho, Xiong, Yang, Kwon, Cheng, 2026 — [arXiv:2607.00466](https://arxiv.org/abs/2607.00466) — the MoE-specific decode-scheduling variant.
- Paper: *Splitwise* — Patel et al., Microsoft, 2023 — early dense-LLM PD-disaggregation with KV-cache handoff design.
- Paper: *DistServe* — Zhong et al., 2024 — disaggregated prefill / decode with goodput-oriented placement.
