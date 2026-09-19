# Linear-Memory Attention Family

*Taxonomy — attention variants that trade the Softmax quadratic for a linear-memory recurrence.*

**TL;DR:** Standard Softmax attention is $O(n^2)$ in sequence length. A family of "linear-memory" attention variants replaces the pairwise interaction with a running key/value memory updated once per token (or per group of tokens), giving $O(n)$ compute with a constant memory footprint. The modern default in this family for long-sequence LLMs is a **hybrid** — Softmax attention for local windows plus a linear-memory branch for long range — because pure linear attention loses fine-grained interactions. Variants differ in how the memory is updated (delta rule, gated recurrences, state-space) and at what granularity.

**Related taxonomies:** none yet
**Depth files covered here:** [video-delta-attention](video-delta-attention.md) · adjacent — [mla](mla.md) (Softmax with cache compression, not linear)

---

## The problem

Attention over $n$ tokens costs $O(n^2)$ FLOPs and $O(n^2)$ memory reads at compute time. For long-context LLMs (100k+ tokens) and video-diffusion DiTs (10k+ tokens per frame × many frames), this dominates end-to-end latency and forces sequence tricks (sliding window, sparse attention) that lose long-range interactions.

Linear-memory attention side-steps the quadratic by keeping a fixed-size memory that summarizes all prior tokens; each new token reads from and writes to that memory in $O(1)$.

## The shared pattern

All variants replace pairwise $\text{softmax}(QK^\top)V$ with a *memory* $M_t$ evolved token-by-token:

```
M_{t} = update(M_{t-1}, k_t, v_t)      # write
o_t   = query(M_t, q_t)                # read
```

Total compute: $O(n)$ in sequence length. Total memory: $O(1)$ in sequence length (the size of $M$).

They differ in:

- **How the memory update is defined** (linear map, delta rule, gated recurrence, state-space evolution).
- **What granularity of tokens updates it** (per token, per frame, per chunk).
- **Whether the memory is causal, bidirectional, or both.**
- **Whether they're deployed pure or in a hybrid stack with Softmax.**

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Linformer / Performer | Approximate Softmax with a low-rank / kernel-features projection | Approximation error on hard heads | Baseline; rarely competitive at scale |
| Mamba / SSM | State-space recurrence with learned dynamics | Requires special CUDA kernels | Long-context LLMs, audio |
| DeltaNet | Delta-rule update (correct memory when a new key overlaps an old one) | Extra bookkeeping | Long-sequence LLM heads |
| Gated DeltaNet | DeltaNet + learnable gates on read/write | More parameters | Same, but higher-fidelity |
| [Video Delta Attention (VDA)](video-delta-attention.md) | Frame-granular delta update using all spatial tokens jointly | Per-frame batching required | Video DiTs where temporal is dominant |
| Sliding-window / local Softmax | Not linear — but paired with linear-memory in hybrids | Loses long range on its own | Always as a hybrid partner |
| Compressed KV ([mla](mla.md)) | Not linear — Softmax with a shared low-rank latent | Different family, complementary | LLM KV-cache reduction |

## How to choose

For a fresh LLM: start with **hybrid** — sliding-window Softmax for locals + a linear-memory branch (Gated DeltaNet or Mamba) for long range. Pure linear attention rarely matches Softmax on hard heads (retrieval, in-context learning). For a video DiT: use **VDA** as the linear branch and Softmax for cross-modal (text/audio) heads. For KV-cache reduction on an existing Softmax stack: use [mla](mla.md) rather than switching families.

## Adjacent but distinct

- **KV-cache compression** ([mla](mla.md)) — same Softmax attention, smaller cached state. Not linear.
- **Sparse attention** — sub-quadratic Softmax that skips pairs. Complementary to linear-memory (can co-exist in the same layer).

## Sources

- Paper: *Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation* — Xi et al., 2026 — [arXiv:2609.20744](https://arxiv.org/abs/2609.20744).
- Prior art: Mamba (Gu & Dao, 2023), DeltaNet (Schlag et al., 2021), Gated DeltaNet (2024).
