# FlashMask-3
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distributed, mask-aware attention kernel for long-context training. Takes structured attention masks (document packing, bidirectional + causal mixes, custom multimodal patterns) and lowers them into (a) a Hopper-tuned single-GPU kernel and (b) a multi-GPU extension that localizes mask metadata without communication, load-balances across ranks whose per-rank workload differs under the mask, and overlaps communication with attention compute. End-to-end: 7.3–11.0% training speedup for a 118B-A14B MoE at long context.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [dualpipe.md](./dualpipe.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md), [partial-rollouts.md](./partial-rollouts.md)

---

## What it is

Modern long-context training uses structured attention masks for several reasons:

- **Document packing**: multiple short sequences packed into one long training example, with cross-document attention masked out.
- **Mixed causality**: encoder-style bidirectional attention over some spans, causal elsewhere.
- **Multimodal masks**: image tokens attend bidirectionally within an image but causally overall.

The compute problem: expressing these masks naively through FlexAttention / Megatron-Core-CP style abstractions leaves significant performance on the table because kernels can't exploit the mask structure, and distributed implementations exchange mask metadata that doesn't need to move.

FlashMask-3 attacks both: a Hopper-tuned kernel and a distributed design that treats mask structure as locally recoverable.

---

## How it works

### Single-GPU kernel

Hopper-targeted implementation (TMA, async warp specialization) that reads mask structure into kernel-level control flow instead of materializing a mask tensor. Treats structured masks (block-wise, prefix-causal, document-packed) as compilation-time variants; emits specialized loops that skip masked blocks entirely.

### Distributed extension

Three complementary moves:

1. **Zero-communication mask localization.** Each rank reconstructs the mask it needs from the global sequence partition without exchanging mask tensors across ranks. The mask-partition rule is a pure function of rank id + global offsets.

2. **Load balancer.** Structured masks make per-rank workload unequal (document packing hands some ranks shorter effective sequences). FlashMask-3 adjusts the per-rank slice sizes so attention FLOPs are balanced across ranks, not just token counts.

3. **Communication-compute overlap.** The sequence-parallel all-to-all for queries and keys overlaps with attention compute: start the next communication while the current attention is running. Combined with (1), the metadata-exchange tail disappears.

---

## Why it matters

- **Throughput beats incumbents.** Reported wins over FlashMask, FlexAttention, MagiAttention, and Megatron-Core CP at both single-GPU and multi-GPU scales.
- **End-to-end 7.3–11.0% on a 118B MoE.** Not a kernel-micro win — a real pretraining speedup. On a model that costs 7+ figures to train, single-digit percentages are real money.
- **Keeps the mask abstraction.** Users still describe masks declaratively; the kernel + distribution take care of the lowering. Doesn't force the model author to specialize code paths manually.

---

## Gotchas & tricks

- **Hopper-specific kernel.** The single-GPU kernel targets H100-class features (TMA, async warp specialization). Works on older GPUs via a fallback path, but speedup shrinks.
- **Load balancer tuning.** For very irregular masks (e.g., long-tail document length distributions) the static per-rank slicing can still leave stragglers. The paper's balancer is heuristic; worst-case masks may need a different allocation.
- **Compose with pipeline-parallel.** FlashMask-3 operates on tensor-parallel + sequence-parallel axes; it is orthogonal to [DualPipe](./dualpipe.md). The 7.3–11.0% number is on top of a Megatron-Core-CP baseline.
- **Mask reconstruction cost.** "Zero-communication" means no cross-rank exchange, but local reconstruction uses CPU-side bookkeeping; large sequence counts need careful implementation to avoid making that the bottleneck.

---

## Sources

- Paper: *FlashMask-3: Efficient and Expressive Mask-Aware Distributed Attention* — He et al., Baidu, 2026 — arXiv [2610.17901](https://arxiv.org/abs/2610.17901).
- Baselines: FlashMask v1/v2, FlexAttention, MagiAttention, Megatron-Core Context Parallel.
