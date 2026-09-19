# Compressed Sparse Attention 2 (CSA2)
*Depth — MLA extended with cross-layer KV cache reuse for agent-scale KV compression.*

**TL;DR:** MLA (V2/V3) stores one low-rank KV latent per token per layer. CSA2 (DeepSeek V4.1-Flash, 2026) goes further: a single compressed KV latent is **shared across multiple layers**, and each layer applies a **sparse-attention pattern** on top. Combined with FP4 quantization of the latent, this shrinks the global (HBM-resident) KV cache to ~890 bytes/token in V4.1-Flash — roughly 1/4 of V4-Flash's per-token footprint.

**Prereqs:** [mla](mla.md), [multi-head-attention](multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [deepseek-moe](deepseek-moe.md), [../inference/fp4-kv-cache.md](../inference/fp4-kv-cache.md)

---

## What it is

**MLA** (Multi-head Latent Attention) shrinks the per-token KV cache within each layer: one low-rank latent + a rotary slice, reconstructed per head via up-projections. Cache size per layer drops from $2 H d_h$ scalars to $d_c + d_h^R$ (in V3: from ~32k scalars to 576).

**CSA2** extends this along the *layer* axis. Where MLA saves a factor per layer, CSA2 lets a single compressed KV latent serve multiple layers at once, complemented by a per-layer **sparse-attention** pattern that selects which tokens each layer actually attends to. The net effect on the KV cache is a multiplicative saving on top of MLA's per-layer saving.

Introduced in DeepSeek-V4.1-Flash (2026). It is what makes long-context agent workloads tractable at frontier scale.

---

## How it works

*Note: the exact CSA2 recipe is under-specified in the abstract; the description below is the mechanism the abstract implies. Full paper appendix should be checked once available.*

### Cross-layer KV sharing

Group the model's layers into $G$ groups (e.g. every 2–4 consecutive layers form a group). Each group shares **one** cached KV latent per token. Individual layers reconstruct their own keys/values via per-layer up-projections, exactly as MLA does per layer — but the compressed latent is not recomputed per layer.

Storage cost per token: $G$ compressed latents instead of $L$ (where $L$ is the total layer count) — a factor of $L/G$ saved on top of MLA.

### Per-layer sparse attention

Because layers in a group share the latent, they cannot arbitrarily differ in which tokens they attend to without cost. CSA2 adds a **sparse-attention pattern** at the layer level: each layer selects a subset of tokens (structured or learned mask) to actually read from the shared cache. This preserves per-layer flexibility while keeping the cache shared.

### Combined with FP4 KV cache

The remaining cached latents are stored at 4-bit precision (see [fp4-kv-cache](../inference/fp4-kv-cache.md)). CSA2 sets the *shape* of the cache; FP4 sets the *precision*. Both compress against MLA's baseline.

---

## Why it matters

- **Multiplies against MLA.** MLA's per-layer compression was already 5–10× vs. MHA. CSA2 stacks another factor of $L/G$ on top by sharing across layers.
- **Sets the long-context frontier.** ~890 B/token in HBM makes 1M-token contexts serve-able without exotic hardware — this was previously the constraint on serving long-horizon agents.
- **Agent-first design.** Prefill and KV bandwidth are the bottleneck for input-heavy agent workloads. CSA2 is the attention-side of DeepSeek's answer, paired with CED and SWA Bounded Replay.
- **Extends the MLA line cleanly.** CSA2 is a strict generalization: set $G = L$ and turn off sparsity and you recover MLA. Backward-compatible with the MLA ecosystem.

---

## Gotchas & tricks

- **Group size $G$ trades quality for compression.** Very small $G$ (aggressive sharing) risks quality; very large $G$ (near-MLA) gives up the win. V4.1-Flash's specific grouping isn't in the abstract.
- **Sparse patterns need per-layer flexibility.** Fixed strided sparsity is cheap but rigid; learned masks help but add training complexity.
- **Reconstruction overhead scales with heads.** As with MLA, per-head up-projection must fuse into the attention kernel or the reconstruction cost dominates.
- **Interacts with rotary handling.** MLA carries a small decoupled RoPE slice; CSA2 must preserve that logic or per-token positional encoding breaks under sharing.
- **Deployment stack co-design.** CSA2's benefit shows only when the serving runtime supports it. Naive vLLM/SGLang forks will not batch efficiently until they understand the shared-latent layout.

---

## Sources

- Paper: *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* — DeepSeek-AI, 2026 — [arXiv:2609.19969](https://arxiv.org/abs/2609.19969).
- Predecessor: MLA in *DeepSeek-V2* — DeepSeek-AI, 2024 — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434).
- Case study: [deepseek-v4-1-flash](../case-studies/deepseek-v4-1-flash.md).
