# FlashAttention
*Depth — tiling attention so it fits in on-chip SRAM, exact and memory-efficient.*

**TL;DR:** Classical attention materializes the `T × T` attention matrix in HBM, which is the memory bottleneck for long sequences. FlashAttention computes the exact same output by tiling the computation into blocks that fit in on-chip SRAM, never writing the full attention matrix to HBM. Saves `O(T²)` memory, runs ~2–4× faster for typical sizes, is numerically exact (not an approximation), and composes with causal masking, dropout, and relative-position biases through later kernel variants.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [mla](../architectures/mla.md), [rope](../fundamentals/rope.md), [_positional-encoding](../fundamentals/_positional-encoding.md)

---

## What it is

Standard scaled-dot-product attention:

```
S = Q K^T / √d_h       # [T, T]
P = softmax(S)         # [T, T], causal-masked if needed
O = P V                # [T, d_h]
```

Memory traffic to HBM is dominated by writing and reading the `T × T` matrices `S` and `P`. On modern GPUs, HBM bandwidth is the bottleneck, not compute — attention is a memory-bound kernel.

FlashAttention's observation: `S` and `P` are intermediate results. If you compute attention in **blocks** small enough to live in on-chip SRAM, and you can maintain the softmax normalization statistics *online* as you move between blocks, you never need to materialize `S` or `P` in HBM at all. The output `O` is the same, bit-for-bit, as the naive implementation.

## How it works

### Block decomposition

Split `Q`, `K`, `V` into blocks of rows: `Q_i` of size `B_r`, and `K_j`, `V_j` of size `B_c`. The block sizes are chosen so `B_r · B_c · fp16_bytes` fits in SRAM along with partial sums.

### Online softmax

Softmax is normally one-pass over the full row: subtract the max, exponentiate, normalize. For attention this normalizes across all `T` keys. FlashAttention computes it **incrementally** over `K/V` blocks:

```
# For each Q block i:
m_i, l_i, O_i = -inf, 0, 0     # running max, normalizer, output
for each K/V block j:
    S_ij = Q_i K_j^T / √d_h
    m_new = max(m_i, rowmax(S_ij))
    P_ij  = exp(S_ij - m_new)
    l_new = exp(m_i - m_new) · l_i + rowsum(P_ij)
    O_i   = exp(m_i - m_new) · O_i + P_ij V_j
    m_i, l_i = m_new, l_new
O_i = O_i / l_i
```

Each block of `K/V` is loaded once, used, then discarded. The running `(m_i, l_i, O_i)` state is tiny (two scalars + one row of `d_h`). No `S` or `P` ever lives in HBM.

### Backward pass

The backward pass recomputes `S` and `P` on the fly from `Q`, `K`, `V` rather than storing them. Costs 1.3–1.5× the forward FLOPs but saves `O(T²)` activation memory.

### Kernel implementation

Implemented in CUDA (FlashAttention-1, -2) and Triton (FlashAttention-3, FlashAttention-2 ports, and research variants). The Triton implementation is more adaptable to custom attention variants but slightly slower than hand-tuned CUDA for the canonical case.

## Why it matters

- **Removes the `O(T²)` memory wall.** Before FlashAttention, 8K-token training was already pushing activation memory limits. After, 128K+ context lengths became tractable at reasonable cost.
- **Not an approximation.** Earlier "efficient attention" methods (Linformer, Performer, Longformer) saved memory by *approximating* attention, which cost quality. FlashAttention is bit-exact.
- **The default in modern training stacks.** PyTorch's `torch.nn.functional.scaled_dot_product_attention`, HuggingFace Transformers (`attn_implementation="flash_attention_2"`), vLLM, SGLang, DeepSpeed, and nearly every production training pipeline default to FlashAttention.
- **Base for every "variant attention" kernel.** Tiled attention kernels for GQA, MLA, relative-position bias, ALiBi, sliding-window, and others all extend the FlashAttention template.

## Gotchas & tricks

- **Head dimension matters.** FlashAttention-2 is tuned for `d_h ∈ {64, 128}`. Non-power-of-two head dims work but are slower.
- **Backward recomputation.** If you're memory-constrained for training at long context, the recompute cost shows up as 20–30% slower end-to-end. Usually worth it; sometimes gradient checkpointing is a cheaper substitute.
- **Causal masking is handled inside the kernel.** Don't construct a `T × T` causal mask separately — pass the `is_causal=True` flag to the kernel.
- **Relative-position biases need a variant.** Vanilla FlashAttention handles RoPE (which fuses into Q/K pre-attention) and additive biases up to a size. For arbitrary relative-position schemes (T5-bias, ALiBi), a specialized kernel is needed — see *Fast and Exact Relative-Position Attention via Structure-Aware Kernels* (arXiv 2611.01182) for a modern Triton variant.
- **Dropout changes the recomputation.** Random dropout masks must be *regenerated* identically on the backward pass. FlashAttention stores the dropout seed; custom variants must do the same.
- **Not useful on small sequences.** The kernel's launch overhead and SRAM-sizing logic is overkill for `T < 128`. PyTorch's native SDPA dispatches to the right implementation per shape.
- **Numerical precision.** The online softmax uses bf16 accumulators by default; for very long sequences or tight numerical requirements, use fp32 accumulators (slower but safer).

## Sources

- Paper: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — Dao, Fu, Ermon, Rudra, Ré, 2022 — arXiv 2205.14135 — the original.
- Paper: *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* — Dao, 2023 — arXiv 2307.08691 — the production-default revision.
- Paper: *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* — Dao et al., 2024 — H100-specific optimizations and fp8 support.
- Paper: *Fast and Exact Relative-Position Attention via Structure-Aware Kernels* — Boychev, Petrov, Dimitrov, 2026 — arXiv 2611.01182 — extends FlashAttention to exact relative-position biases.
