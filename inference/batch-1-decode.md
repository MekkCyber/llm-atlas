# Batch-1 LLM Decode

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Single-stream autoregressive decode (one user, one session, batch size 1) is the dominant inference workload for physical-AI / edge deployments — robots, autonomous vehicles, embodied agents. It is widely (and correctly) said to be HBM-bandwidth-bound. But that's not the whole story: on fast GPUs (H100), achieved fraction of peak HBM bandwidth falls to ~27% while a slow GPU (L4) reaches ~81% on the same workload. A 44-cell controlled study identifies *launch-side overhead* (kernel launches, scheduling) as the missing term — visible on fast GPUs, hidden on bandwidth-bound slow ones. CUDA Graphs recover most of the gap on H100 (1.26×) but barely help on L4 (1.03×).

**Prereqs:** *(no in-graph prereq files yet — assumes familiarity with autoregressive decoding and KV cache)*
**Related:** [README](README.md) · [../quantization/_number-formats](../quantization/_number-formats.md) · [../quantization/fp8](../quantization/fp8.md)

---

## What it is

In cloud LLM serving, latency-per-token is largely amortized across a batch — many users' decode steps run in parallel, weight reads from HBM are shared, and compute units stay busy. Batch-1 decode is the opposite regime: one stream, one token at a time, no amortization. The classical model is:

```
time_per_step  ≈  (model_weights + active_KV_cache) / HBM_bandwidth
```

The implication is that batch-1 decode latency is set by HBM bandwidth — buy a faster-HBM GPU, decode faster.

The source paper measures this on three 7–8B GQA transformers across H100 SXM5, A100-80GB SXM4, L40S, and L4 GPUs, over contexts 2048–16384, in bf16 SDPA, for 44 valid measurement cells. Empirically, the bandwidth model is *true but incomplete*: achieved bandwidth as a fraction of peak HBM falls as peak HBM rises.

---

## How it works

### The missing term: launch overhead

On a slow GPU, each kernel runs long enough that scheduling and launch overhead is a small fraction of step time — the GPU mostly waits for HBM. On a fast GPU, kernels are short and launch overhead becomes a meaningful slice of the step. The same workload that is bandwidth-bound on an L4 becomes launch-bound on an H100.

The paper isolates this term with a CUDA-Graphs A/B test. CUDA Graphs replace many small kernel launches with a single graph launch, eliminating launch overhead. Results:

| GPU | CUDA-Graphs speedup | 95% bootstrap CI |
| --- | --- | --- |
| H100 (ctx=2048) | **1.259×** | [1.253, 1.267] |
| L4 (ctx=2048) | **1.028×** | (much smaller) |

The H100 was substantially launch-overhead bound on the bf16 path; the L4 wasn't. Quantization-decode and longer contexts shift the balance further.

### The 44-cell matrix

3 models (Qwen-2.5-7B and two others) × 4 GPUs × multiple contexts (2048, 4096, 8192, 16384). The achieved-bandwidth-fraction-of-peak metric falls monotonically with peak HBM. Faster memory does not translate to proportional latency gains.

### Implications for hardware choice

For batch-1 decode workloads, the optimal GPU is the one that minimizes `weights/bandwidth + launch_overhead`, not the one with the highest HBM bandwidth. For a fast GPU, the launch term must be paid down via CUDA Graphs or similar before the bandwidth advantage materializes.

---

## Why it matters

- **Corrects the canonical "decode is memory-bound, full stop" model.** The bandwidth model is correct as a first-order story and wrong as the full story.
- **Practical hardware-procurement implication.** Edge / physical-AI deployments choosing GPUs for batch-1 production workloads may rationally prefer slower-HBM, lower-cost devices — the H100's bandwidth advantage is partly squandered without CUDA Graphs.
- **CUDA Graphs are not optional on fast GPUs.** Out-of-the-box bf16 decode on H100 leaves ~25% of decode speed on the table. Graph-based decode (or equivalent launch-amortization) is required to realize the hardware's spec.

---

## Gotchas & tricks

- **CUDA Graphs need static shapes.** Variable sequence length and dynamic KV-cache layouts make naive graph capture impossible. Production stacks (vLLM, TensorRT-LLM) work around this with bucketed graph caches or piecewise capture.
- **Quantization changes the balance.** FP8 / INT4 weights shrink the per-step memory traffic, which *increases* the launch-overhead share. The 1.26× CUDA-Graphs speedup on H100 grows with quantization, not shrinks.
- **Multi-stream batch-1 (multiple sessions, each batch-1)** is a different workload — has launch amortization across streams but introduces inter-stream contention. Don't extrapolate from the paper's single-stream results.
- **Long contexts shift the bottleneck back to bandwidth.** As KV cache grows past ~32K tokens, per-step bytes climb and bandwidth dominates again — launch overhead becomes negligible.

---

## Sources

- Paper: *The Physical AI Inference Gap in Batch-1 LLM Decode — A 44-Cell Cross-GPU Study of Memory Floors, CUDA Graphs and Quantized Decode*, 2026. 44-cell measurement matrix; H100/A100/L40S/L4 across three 7–8B GQA transformers; CUDA-Graphs A/B isolating launch-side overhead.
