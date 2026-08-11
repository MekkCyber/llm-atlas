# Fused chunked KL loss
*Depth — a memory-linear-in-sequence-length KL kernel for distillation training.*

**TL;DR:** The naive KL loss for LLM distillation materializes a $(\text{seq} \times \text{vocab})$ logit tensor, which caps trainable context length: at vocab ~150k, a single 32k-token sequence's logits alone exceed 20 GB in bf16. The fused chunked KL loss splits the sequence into chunks and fuses forward + softmax + KL per chunk, never materializing the full tensor. Peak memory becomes linear in sequence length, enabling **32,768-token context on a single GPU** — a 4× extension over the naive baseline.

**Prereqs:** [README.md](README.md), [../_post-training.md](../_post-training.md)
**Related:** [offline-top-k-distillation.md](offline-top-k-distillation.md), [_distillation.md](_distillation.md)

---

## What it is

Standard PyTorch/JAX implementations of the KD loss compute `student_logits = student(...)` (a full $(B, L, V)$ tensor) and then `loss = KL(student_logits, teacher_logits)`. The full logit tensor stays live in memory for backward, and at LLM-scale vocab (100k–200k) this is what pins peak memory.

The fused chunked KL loss rewrites this as a per-chunk fused kernel: for each contiguous chunk of the sequence, compute logits + softmax + KL + backward-grad accumulation without ever materializing the full-sequence logit tensor off-chip.

## How it works

Given a sequence of length $L$ and chunk size $C$:

```
grad_accumulator = 0
for chunk_start in range(0, L, C):
    chunk_logits = student_head(hidden_states[chunk_start:chunk_start+C])   # (C, V)
    chunk_teacher = teacher_topK[chunk_start:chunk_start+C]                 # cached
    chunk_loss, chunk_grad = fused_kl_kernel(chunk_logits, chunk_teacher)
    grad_accumulator += chunk_grad @ hidden_states[chunk_start:chunk_start+C].T
    del chunk_logits                                                        # freed
```

Key implementation properties:
- **Chunk logits are freed** after per-chunk backward — never all in memory at once.
- **Fused kernel**: softmax + KL + backward gradient are done in one CUDA kernel with only the top-K teacher slot (compatible with [offline-top-k-distillation](offline-top-k-distillation.md)).
- **Peak memory linear in $L$**, independent of chunk count once $C \ll L$.

The paper's isolated head-only benchmark verifies memory and iteration-rate scaling from **4K to 256K tokens**.

## Why it matters

- **4× context length extension** — 32,768 tokens on a single GPU vs ~8k naive baseline.
- **Enables long-context healing.** Distillation of long-context capabilities requires long-context training data; the naive KL kernel is what blocked most teams from doing it on single-GPU setups.
- **Independent of Top-K.** Works with full-vocab KL too, though the memory savings are much larger paired with top-K.
- **Kernel-level, not framework-level.** Once integrated, transparent to the training loop.

## Gotchas & tricks

- **Chunk size $C$ is a tradeoff.** Small $C$ = more kernel launches, higher overhead; large $C$ = higher peak memory. Sweep to find the memory/throughput knee (paper suggests $C$ ~ 256–1024).
- **Doesn't help attention memory.** This kernel addresses the *output-head* memory only. Attention memory is still $O(L^2)$ unless you also use FlashAttention.
- **Backward-through-chunk carefulness.** Standard autograd needs to see the full chain; the kernel implements a custom backward that recomputes softmax within each chunk. Bug-prone; use the released kernel rather than reimplementing.
- **Compat with sequence packing.** Packing multiple docs into one sequence works, but chunk boundaries must not straddle document boundaries or the per-doc loss weighting drifts.

## Sources

- Paper: *Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss* — Ryskulov, García-Ferrero, Montero et al., Multiverse Computing, 2026 — arXiv:2608.03796.
- Code: https://github.com/CompactifAI/Full-Chunked-KL-Loss.
- Related: FlashAttention (Dao, 2022) — the same "fuse + tile" idea applied to attention rather than to the output-head KL.
