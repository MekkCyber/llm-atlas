# Gated DeltaNet (Stable GDN-2)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Gated DeltaNet is a linear-attention architecture whose hidden recurrent state tends to blow up during training at scale. Stable GDN-2 diagnoses this as an asymmetry in the update rule and fixes it with two changes — **symmetric erasure** and a **decay-regulated residual write** — with a provable O(√T) bound on state norm and consistent wins on long-context retrieval at 1.6B.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [../architectures/qk-norm.md](../architectures/qk-norm.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

Linear-attention transformers approximate softmax attention with a recurrent update, trading the O(N²) KV-cache scan for O(N) state propagation. Gated DeltaNet (GDN) adds a per-token forget gate on top of a delta-rule key-value write. GDN-2 scales the idea up. Stable GDN-2 is a numerics / parameterization fix, not a new architecture: same block shape, same inference cost, different update law.

## How it works

The vanilla GDN-2 update is roughly S_t = α_t · (I − β_t · k_t k_tᵀ) · S_{t-1} + β_t · v_t k_tᵀ, where S_t is the per-head state, α is the decay gate, β is the write strength, and k,v are key/value projections. The *erase* term (I − βkk ᵀ) is not symmetric in its effect on state propagation — in some directions it contracts, in others it expands — and the state norm grows as O(T).

Stable GDN-2 makes two changes:

1. **Symmetric erasure.** Replace the one-sided erase with a symmetric update so the eigenvalue structure of the propagation operator is bounded on both sides. Intuitively: forget *and* its transpose are applied together, so there is no preferred growth direction.
2. **Decay-regulated residual write.** The residual write v_t k_tᵀ is gated by the current decay α_t, so writes slow down as the state fills. Writing flexibility is preserved — the gate is learned, not hard-capped.

Together these yield a provable O(√T) bound on ||S_t||, replacing GDN-2's unbounded growth.

## Why it matters

Linear-attention variants (DeltaNet, GLA, Mamba, RWKV family) are the current cleanest path to sub-quadratic inference on long sequences. At scale, they are notoriously unstable: loss spikes, degraded recall, and NaNs correlate with state-norm blow-up. A general-purpose stabilizer that costs nothing at inference is a direct enabler for using them as drop-in replacements for softmax attention in production models.

## Gotchas & tricks

- The fix is a *reparameterization* — inference is unchanged, so deployed GDN-2 checkpoints can be converted without retraining the output head.
- Symmetric erasure alone is not enough; without the decay-gated write, the bound reverts to linear growth at the margin.
- Long-context retrieval (needle-in-haystack class) is the most sensitive downstream indicator — perplexity alone understates the instability.
- Still linear-attention; it does not recover softmax attention on tasks that need unbounded associative recall.

## Sources

- Paper: Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability — Zhang et al., 2026 — [arXiv:2611.34051](https://arxiv.org/abs/2611.34051)
- Related: Gated DeltaNet (original) — see the linear-attention taxonomy
