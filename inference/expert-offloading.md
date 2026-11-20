# Expert Offloading (Tideway)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A memory-management strategy for serving **MoE** LLMs on consumer hardware where experts can't all fit in VRAM. Treats host RAM + GPU VRAM as a **unified memory pool** with phase-specific layouts for prefill (compute-bound) and decode (memory-bound), and keeps CUDA Graph execution alive under expert paging. Reports 1.97× end-to-end, 2.40× TTFT and 1.61× TPOT over the fastest baseline on consumer GPUs.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md)
**Related:** [README.md](README.md)

---

## What it is

Running a modern MoE LLM (DeepSeek-V3, Qwen3-MoE, Kimi-K1.5) locally on a consumer PC puts total parameters far above what VRAM holds. Experts spill to host RAM and in the worst case to SSD. The classic fix — page experts in on demand — kills throughput because per-expert paging interacts badly with CUDA Graphs and with the different access patterns of prefill (large batches, bandwidth-bound) and decode (single token, latency-bound).

**Tideway** redesigns expert memory as one tiered pool and switches memory layout between phases.

---

## How it works

1. **Unified RAM/VRAM pool.** The scheduler sees total memory, not two tiers. Expert weights can live in either; the scheduler decides based on current phase and recent access pattern.
2. **Prefill layout** — bandwidth-optimized. Weights are laid out contiguously for large transfers; experts streamed from RAM in batched transfers overlapping compute.
3. **Decode layout** — latency-optimized. Hot experts stay pinned to VRAM; cold experts paged from RAM in small units sized for single-token activations. Activation routing is predicted ahead by one step where possible.
4. **CUDA Graphs under paging.** Expert weight pointers are made addressable from inside the graph so paging doesn't invalidate the captured graph. This is the systems bit that separates Tideway from simpler offload stacks.

---

## Why it matters

- **Local-first MoE deployment** is a growing niche: open MoEs (Mixtral 8×7B, Qwen MoE, OLMoE) on laptops, Jetsons, prosumer GPUs. The bottleneck is not compute but memory hierarchy.
- **Phase-aware memory is a reusable systems pattern.** Prefill and decode have fundamentally different optimal layouts; separating them is a clean primitive.
- **Opens the door to phone-/laptop-scale MoE agents.** Where speculative-decoding / batching tricks for dense LLMs have landed on consumer devices, MoE has been stuck on expert paging. Tideway is a concrete path forward.

---

## Gotchas & tricks

- **Only helps when VRAM is the binding constraint.** On a server with enough VRAM for all experts, Tideway degrades to plain MoE serving.
- **CUDA-graph integration is implementation-heavy.** Off-the-shelf frameworks (vLLM, TGI) don't yet support paged weights inside captured graphs; porting is nontrivial.
- **SSD tier is worst case.** If working set spills beyond RAM, performance craters. Tideway improves over naive offloading but doesn't rescue undersized machines.
- **Expert-access prediction quality bounds decode speedup.** When routing is unpredictable (e.g. highly non-stationary input), prefetch misses dominate.

---

## Sources

- Paper: *Tideway: Phase-Aware Tiered Memory Management for MoE Inference on Consumer Devices* — Wang, Yang, Xu, Ye, Wang, Zhou, Chen, Li, Zhuo, Chen (Duke) — 2026 — [arXiv:2611.14848](https://arxiv.org/abs/2611.14848).
- Background: *DeepSeekMoE* — fine-grained expert design that motivates aggressive paging.
- Related systems: FlexGen (Sheng 2023) for dense-LLM offloading.
