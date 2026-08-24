# Block-Sparse Prefill Attention
*Depth — cut prefill FLOPs by dropping whole attention blocks, without breaking paged KV cache or FP8.*

**TL;DR:** During long-context prefill, most of the attention matrix carries little signal — a block-sparse pattern can drop 80–95% of blocks with modest quality loss, provided (a) the residual bias from dropped blocks is corrected, (b) the kernel runs natively on FP8 tensor cores, and (c) sparsity metadata composes with the paged / continuously batched KV layouts modern serving stacks demand. FlashPrefill V2 addresses all three and reports up to **47.26× vs FlashAttention-2** at 128K context in FP8.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../quantization/fp8.md](../quantization/fp8.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [../fundamentals/dca.md](../fundamentals/dca.md), [README.md](README.md)

---

## What it is

Prefill is the compute-bound phase of serving: attention is `O(L²)`, and long-context requests (RAG, coding sessions, transcripts) pay this cost up front on every request. Block-sparse prefill partitions the `Q × Kᵀ` matrix into tiles and evaluates only a chosen subset — driven by fast top-k, hash-based routing, or a learned oracle — leaving the rest zero.

Block granularity (typically 64–256 tokens per side) matters as much as sparsity ratio: too small and per-block launch overhead dominates; too large and you drop information that matters.

## How it works

Three orthogonal levers that FlashPrefill V2 combines:

1. **Sparsity pattern** — for each query block, pick the top-k key blocks by an approximate score (block mean, low-rank sketch, or online proxy). Everything else is skipped.
2. **Mean correction** — the omitted attention mass is not zero on average. Subtract a per-row estimate of the missing softmax contribution so the output is unbiased. Cheap: one scalar per block, computed alongside the sketch.
3. **FP8 + paged compatibility** — the sparse kernel is written for FP8 tensor cores from the start (`e4m3` for weights, `e5m2` for activations), and takes its K/V pointers as paged-KV descriptors so vLLM/SGLang can batch requests with different sparsity patterns in one kernel launch.

The output shape is unchanged, so the kernel drops in behind a standard FlashAttention interface.

## Why it matters

Long-context prefill is currently the wall on serving cost: MLA and GQA compressed KV size, but prefill FLOPs remained quadratic. A block-sparse kernel that (i) preserves quality, (ii) speaks FP8, and (iii) integrates with paged batching is the shape serving infra can adopt without a rewrite. Reported gains — **47.26× (FP8)** and **27.19× (BF16)** vs FlashAttention-2 at 128K — apply directly to the tokens/sec-per-GPU that determines serving margin.

## Gotchas & tricks

- **Mean correction is not optional at high sparsity.** Below ~15% density, plain top-k prefill visibly biases downstream logits; a mean-correction term keeps quality flat.
- **Block size is a serving-time knob.** Bigger blocks amortize kernel overhead but coarsen the sparsity decisions; 128 is a common default. Retune when the model's typical context length shifts.
- **FP8 quantization interacts.** FP8 already introduces per-block scale drift; combining with block-sparse selection can compound. Calibrate scales on the sparse output, not on the dense baseline.
- **Paged KV compatibility is the deployment gate.** A sparse kernel that assumes contiguous K/V is unshippable to vLLM-shaped stacks. Design the pattern index in paged descriptors from the start.
- **Doesn't help decode.** Decode is memory-bandwidth-bound, not FLOP-bound — this is a prefill-only optimization. Pair with a decode-side trick (speculative decoding, quantization) for end-to-end gains.
- **Quality regressions concentrate on retrieval-heavy prompts.** Aggressive sparsity hurts needle-in-a-haystack more than natural-language reasoning. Benchmark on RULER before shipping.

## Sources

- Paper: *FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving* — Fan, Huang, Wu, Wang, He, 2026 — [arXiv:2608.19758](https://arxiv.org/abs/2608.19758).
- Related: *FlashAttention-2* (Dao, 2023) and *FlashInfer* (Ye et al., 2024) — the dense prefill kernels this displaces.
- Related: *Native Sparse Attention* / *MInference* — earlier block-sparse prefill designs without FP8 or paged-KV support.
