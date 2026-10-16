# Joint Subspace Sparse Attention (JSSA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-context decoding is bottlenecked by KV-cache bandwidth. Prior sparse-attention methods pick a subset of past tokens but pay for it with a scatter/gather over the KV cache that erodes the theoretical savings. JSSA learns a **shared low-dimensional query-key subspace** for selection and uses a **gather-free kernel** that reads selected positions in contiguous order.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../architectures/mla.md](../architectures/mla.md)
**Related:** [kv-cache-quantization.md](./kv-cache-quantization.md) · [README.md](./README.md)

---

## What it is

Sparse attention at inference: for a long context, score each past token against the current query cheaply, keep the top-$k$, and attend only to those. The two sub-problems:

1. **Scoring:** how to estimate which past tokens matter without reading the full KV cache.
2. **Reading:** how to actually read the chosen positions efficiently on a GPU.

JSSA attacks both with the same observation — queries and keys share a learnable low-dimensional subspace that captures most of the attention mass.

## How it works

### Shared Q/K subspace

Learn two matrices $U_Q \in \mathbb{R}^{d \times r}, U_K \in \mathbb{R}^{d \times r}$ with $r \ll d$. Project:
$$\tilde q = Q U_Q, \qquad \tilde k = K U_K.$$

The approximate attention score $\langle \tilde q, \tilde k \rangle$ is computed in $r$-dimensional space — $d/r$× cheaper to score than full attention. Share the subspace **across heads** (one subspace per layer, not per head) so the projection cost is amortized.

Approximate scores identify a top-$k$ set of past positions.

### Gather-free kernel

Standard sparse-attention kernels pick positions, then scatter/gather them into a contiguous buffer, then compute attention. The gather is a memory-bound bottleneck.

JSSA's kernel instead walks the KV cache **in order** and accepts/rejects each position inline — the chosen positions are read as they appear in memory, no gather. This matches GPU memory-coalescing well and preserves the bandwidth savings sparse selection promises in theory.

## Why it matters

- **Addresses the real bottleneck.** Prior sparse-attention papers often reported FLOP savings while the gather cost ate the latency. JSSA shows a path to actual decoding speedups.
- **Shared subspace is a cheap approximation.** One $r$-dimensional space per layer, not per head, keeps parameter count low and preserves training stability.
- **Portable to inference systems.** The gather-free scheme is a kernel-level change; no architectural retraining required beyond fine-tuning the subspace projections.

## Gotchas & tricks

- **Subspace rank is the key knob.** Too low and the approximate scoring is noisy — top-$k$ selection becomes random. Too high and the compute savings vanish. Papers typically use $r = 64$–$128$.
- **Fine-tune the projections.** The subspace is learned, usually with a short fine-tuning run on long-context data; cold-start initialization (e.g., from SVD of a trained attention block) helps convergence.
- **Gather-free doesn't help short contexts.** For prefill or short-context attention, dense attention wins — JSSA is a long-context decoding optimization.
- **Interaction with KV-cache quantization.** JSSA's gather-free walk composes well with INT4 KV caches: both are bandwidth optimizations and both operate at the kernel level.
- **Head-shared subspace loses per-head specialization.** For tasks where different heads attend to very different positions, consider grouping heads into subspace clusters.

## Sources

- Paper: *Joint Subspace Sparse Attention for Efficient Long-Context Decoding* — Song, Kim, Kim (Seoul National University VLSI Lab), 2026 — [arXiv:2610.19026](https://arxiv.org/abs/2610.19026).
