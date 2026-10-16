# KV Cache Quantization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Decoding is memory-bandwidth bound; the KV cache dominates DRAM traffic. Quantizing cached keys and values to **INT4** (with mixed-precision escapes for high-leverage positions) cuts cache traffic ~4× and gives large wall-clock speedups. For RL rollouts, KV4-RL shows 1.80× cumulative rollout speedup on Qwen3-8B GRPO while keeping accuracy within 0.5pp of BF16.

**Prereqs:** [../quantization/_number-formats.md](../quantization/_number-formats.md), [../architectures/mla.md](../architectures/mla.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../post-training/grpo.md](../post-training/grpo.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

The KV cache stores per-head key/value tensors for every past token. During decoding, each generated token reads the entire cache — bandwidth is the bottleneck well before FLOPs are. Quantizing cache entries from BF16 (or FP16) to INT4 shrinks the read volume by ~4×.

Two concerns dominate the design: (1) which positions can be quantized aggressively without hurting generation quality, and (2) how the dequantization kernel schedules the reads so bandwidth savings translate to real latency.

## How it works

### Per-token / per-channel INT4

For each cached key or value vector, store:
- 4-bit quantized values,
- a per-token (or per-channel-group) scale,
- optionally a per-token zero-point for asymmetric quant.

At attention time, dequantize on the fly inside the kernel. The scales/zeros are small — the cache-size reduction stays close to 4×.

### Mixed precision for sinks and recent tokens

Not all positions are equal. **Attention sinks** (very early tokens that attention concentrates on) and **recent tokens** (last few hundred) take a disproportionate share of attention mass. KV4-RL keeps these at INT8 while INT4-quantizing the rest. The result preserves high-leverage positions and limits the drift of the sampling distribution — critical for RL, where the rollout policy must stay close to the training policy or the gradient is biased.

### Why RL rollouts in particular

In GRPO / RLVR, each training step generates $G$ rollouts per prompt. Rollout decoding is a large fraction of per-step wall-clock. Quantizing the **rollout** KV cache does not touch the training-time cache or weights at all — it's a pure inference-engine change inside the rollout worker. The tradeoff is sample distribution drift vs rollout throughput.

## Why it matters

- **Rollouts dominate RL cost.** Post-training stacks (vLLM, SGLang, veRL, OpenRLHF) spend most time generating rollouts; INT4-with-INT8-sinks is a drop-in speedup.
- **Numbers:** 1.80× cumulative rollout speedup on Qwen3-8B-Base GRPO, 1.27× full-iteration speedup, accuracy within 0.5pp of BF16 on a six-benchmark mean for both 4B and 8B.
- **Decouples training-side and inference-side precision.** Training can stay at BF16/FP8 while rollouts run at INT4 — a composition most RL stacks don't fully exploit yet.
- **Portable.** The sink + recent-tokens mixed-precision pattern applies to any serving stack with a programmable attention kernel.

## Gotchas & tricks

- **Sink identification.** Static (first $k$ tokens) is easy and usually fine; dynamic sink detection costs more and buys little.
- **Group-wise vs per-token scales.** Per-token is accurate but adds metadata overhead. Per-head-channel-group of 64–128 is the common sweet spot.
- **Distribution drift ≠ accuracy loss.** Even when final answers match BF16, the token-by-token distribution shifts under INT4. For RL this biases the importance ratio in GRPO; truncated importance sampling or a KL floor can absorb this.
- **Not for training caches.** INT4 training-side activations/weights is a separate (harder) problem — see FP8/MXFP4 for the training analogue.
- **Kernel-bound savings.** Theoretical 4× bandwidth savings only materialize when the attention kernel is actually bandwidth-bound; small models or very short contexts may be compute-bound and benefit less.

## Sources

- Paper: *KV4-RL: Unlocking Faster Asynchronous Rollouts for RLVR with INT4 KV Cache Quantization* — Park, Kim, Jang, Yang, et al. (NAVER Cloud / KAIST / AITRICS), 2026 — [arXiv:2610.14448](https://arxiv.org/abs/2610.14448).
