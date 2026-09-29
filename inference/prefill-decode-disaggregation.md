# Prefill/Decode Disaggregation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLM inference has two phases with opposite hardware profiles: **prefill** (processing the prompt in parallel — compute-bound) and **decode** (generating one token at a time — memory-bandwidth-bound). Running them on the same GPU pool wastes throughput because either phase starves the other. Disaggregated serving splits them across dedicated pools that hand off the KV cache between phases.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [_number-formats](../quantization/_number-formats.md), [fp8](../quantization/fp8.md)

---

## What it is

A single-request LLM inference consists of:

1. **Prefill** — process the prompt tokens through the model in a single forward pass; produce the KV cache for every prompt token and the first output logits. Cost scales with prompt length; heavy matmuls saturate the tensor cores.
2. **Decode** — sample the next token, extend the KV cache, sample again. One token per forward. Per-step compute is tiny; the bottleneck is loading model weights and the KV cache from HBM.

In **colocated** serving (one GPU pool handles both), prefill batches occupy the GPU with big matmuls while decode batches wait, and long-context prefills spike TTFT for concurrent decodes. Continuous batching (vLLM style) helps but does not eliminate the coupling.

**Disaggregated** serving splits the pools:
- **Prefill workers** — optimized for throughput: large batch sizes, high-arithmetic-intensity kernels, compute-native weight formats.
- **Decode workers** — optimized for latency and memory: KV-cache-friendly, small batch, weight-only quantization.

After the prefill worker finishes, it streams the KV cache over the fabric (NVLink, InfiniBand, or PCIe) to a decode worker, which continues generation.

## How it works

- **KV transfer.** After prefill, the worker sends the per-layer $(K, V)$ tensors to the decode target. Transfer cost is a function of layers × hidden × prompt_len × 2 × bytes-per-element. Some systems compress or quantize the KV during transfer.
- **Scheduling.** A router assigns prefill and decode requests to their pools separately. Ratios (e.g. 1 prefill GPU per 4 decode GPUs) tune to the workload's average prompt/output length.
- **Weight specialization.** Because the two phases run on separate copies of the model, weights can differ: compute-native (FP8 / NVFP4) on prefill, memory-native (INT4 / MXFP4) on decode. This is the "disaggregated quantization" extension — see [disaggregated-quantization](../quantization/disaggregated-quantization.md).
- **Colocated fallback.** When two GPUs cannot be spared per request, "Offloaded Disaggregated Prefill" (ODP) streams the prefill weights from SSD on demand, amortizing weight-load cost over the prompt.

## Why it matters

- **Higher throughput at the same TTFT.** Long-context prefills stop stalling short decodes. Reported gains: 2–10× throughput per GPU vs. colocated serving at equivalent latency SLOs (DistServe, Splitwise, Mooncake).
- **Unlocks phase-specialized weights.** Different quantizations for prefill vs decode; a compute-heavy prefill format can be 4× smaller for decode.
- **Standard architecture for frontier serving stacks.** vLLM (production), TensorRT-LLM, Mooncake, SGLang all now support disaggregation.

## Gotchas & tricks

- **KV transfer is the risk.** For very long prompts on non-NVLink fabrics, transferring $K, V$ can exceed decode-step time; batch prompts by cluster locality.
- **Ratio tuning matters.** Wrong prefill:decode ratio means one pool sits idle. Auto-scalers adjust the ratio per traffic profile.
- **Failure isolation.** A crashed decode worker loses in-flight requests; retry logic must re-run prefill (or persist the KV).
- **Compatibility with speculative decoding.** Draft-verify loops sit inside decode; the disaggregation split still holds.
- **Chunked prefill** is a related but distinct optimization — split *one* long prefill into chunks that interleave with decodes on the *same* worker. Complements disaggregation; doesn't replace it.

## Sources

- Paper: *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving* — Zhong et al., 2024 — https://arxiv.org/abs/2401.09670
- Paper: *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* — Patel et al., Microsoft/UW, 2023 — https://arxiv.org/abs/2311.18677
- System: *Mooncake: A KV-Cache-Centric Disaggregated Architecture for LLM Serving* — Kimi/Moonshot, 2024.
- Paper: *Disaggregated Quantization: Specializing LLM Prefill and Decode* — Panferov et al., ISTA + Qualcomm, 2026 — https://arxiv.org/abs/2609.26333 (weight-format disaggregation on top of serving disaggregation).
