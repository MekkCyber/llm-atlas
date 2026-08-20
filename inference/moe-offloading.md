# MoE Offloading & Edge-Native Serving
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Serve large Mixture-of-Experts models on personal machines (laptop → workstation) by treating the box as a **heterogeneous, elastic inference platform**: continuously remap expert residency, CPU–GPU execution, and KV state as the workload changes, instead of committing to a fixed offloading strategy. Enables frontier-scale MoE (35B–753B) on single-GPU consumer hardware.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md)
**Related:** [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

Serving an MoE at scale on a personal machine differs from datacenter serving in two ways:

1. **The workload is nonstationary.** A coding or tool-using agent will oscillate between long-context reads (many experts touched broadly) and short tool-call bursts (small set of hot experts). No fixed placement policy is optimal for both.
2. **The hardware is heterogeneous.** Consumer setups mix a small dedicated GPU, an integrated GPU, a large CPU with fast DDR, and PCIe-limited bandwidth. The right *split* between compute and memory depends on the specific machine and even the current thermal/power state.

An edge-native MoE serving stack solves both by making expert placement and CPU–GPU execution *adaptive at each step*, not chosen once at deploy time.

## How it works

### Co-designed components

- **Model layout & loading.** Weights are split into chunks that can live in GPU VRAM, CPU DRAM, or on NVMe, with metadata about per-expert access frequency.
- **Expert residency.** A hot-cold cache in VRAM; cold experts fetched from CPU or disk on demand. Residency decisions are reweighted continuously by observed hit rates.
- **CPU–GPU execution.** Whether the attention/FFN of a given layer runs on GPU or CPU is decided per-step based on measured bandwidth headroom, not statically. When GPU compute is idle waiting for a PCIe transfer, CPU picks up the FFN for cold experts.
- **Agentic KV state reuse.** Across turns in an agent loop, most of the KV cache is reusable; a segment-aware KV manager keeps hot prefixes resident while paging cold segments.
- **Runtime memory management.** The scheduler monitors free VRAM/DRAM and rebalances weights, KV pages, and activations to stay under the ceiling.

### The scheduler's core loop

```
each decode step:
  1. estimate needed experts for the upcoming forward
  2. measure current PCIe/HBM bandwidth headroom
  3. plan the residency delta (which experts to evict/promote)
  4. issue transfers overlapped with compute
  5. execute layers on their assigned devices
  6. observe hit rates + latency → update the plan for next step
```

There is no *fixed* offloading strategy: the scheduler continuously maps computation and model state onto the resources actually available.

## Why it matters

The gap between "open weights are available" and "runnable on my machine" is the largest friction in local AI. Datacenter-optimal serving assumptions (all experts fit in HBM, fixed layout) do not survive the edge. FreeToken and its peers change what a consumer machine can practically serve:

- **35B MoE on an 8GB laptop GPU** (only a few experts resident, rest on CPU DRAM).
- **284B MoE on a gaming desktop.**
- **753B GLM-5.2 on a single workstation GPU.**
- Supports **20+ open MoE models** and real code/tool-using agents end-to-end.

## Gotchas & tricks

- **Expert-hit prediction is the win.** A scheduler that predicts the *next* few experts a request will hit can pre-fetch during the current step, hiding PCIe latency almost entirely on typical agent workloads.
- **Fine-grained MoE helps offloading.** Many small experts (DeepSeek-style) fit better in edge cache/DRAM tiers than a few large ones — grain matters as much as sparsity.
- **KV cache dominates memory at long context.** Segmented, resumable KV storage is as important as expert offloading for long-running agents.
- **Speculative decoding stacks cleanly.** Small drafts run on integrated GPU / CPU while the big MoE verifies on the dedicated GPU.
- **Beware quantization interactions.** Different experts under FP8/INT4 have very different sensitivity — a naive "quantize everything to INT4" evicts recall from the least frequently updated experts.
- **Thermal/power throttling breaks static plans.** A scheduler that assumes a fixed compute budget will schedule beyond what the machine can sustain; observe wall-clock latency, not paper FLOPs.

## Sources

- Paper: *FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution* — Yang, Fan, Pan, Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu, Stoica — UC Berkeley / MIT / NVIDIA, 2026 — https://arxiv.org/abs/2608.16157
- Related: DeepSpeed-MII, llama.cpp MoE offloading, PowerInfer — earlier expert-offloading systems along individual axes of this design.
