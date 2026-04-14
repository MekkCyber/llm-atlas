# Inference

*Serving trained models to users — the systems, kernels, and algorithms that decide how fast and how cheaply a model can respond.*

---

## What This Is

Inference is a distinct discipline from training. The constraints flip: training cares about throughput over weeks with checkpointing; inference cares about per-request latency (p50, p99), throughput per GPU, and cost per token, all while serving many users concurrently.

This folder covers how modern LLM serving actually works — from the math of a single decode step to the systems that batch thousands of concurrent requests.

---

## What Belongs Here

- **KV cache** — what it is, how it's stored, paged attention, prefix caching.
- **Batching** — static batching, continuous batching, in-flight batching.
- **Speculative decoding** — draft models, Medusa, EAGLE, lookahead decoding.
- **Prefill vs. decode** — why they're asymmetric, disaggregated serving.
- **Serving systems** — vLLM, SGLang, TensorRT-LLM, TGI — architecture and tradeoffs.
- **Kernels** — FlashAttention at inference time, fused kernels, attention variants.
- **Structured output & constrained decoding** — JSON mode, grammars, logit processors.

## Reading Order

1. KV cache & paged attention
2. Prefill vs. decode asymmetry
3. Continuous batching
4. Speculative decoding
5. Serving systems (vLLM / SGLang)
6. Disaggregated serving

---

## Related

- [quantization/](../quantization/) — quantization is applied inference optimization
- [systems/](../systems/) — training-time infra (distinct concerns, shared primitives)
- [architectures/](../architectures/) — architectural choices that affect inference cost
