# Gated Delta Networks (GDN)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A linear-time sequence mixer that updates a recurrent state via a **delta rule** (write a new key/value pair into an associative-memory-style state) with a learned **gating** signal that controls how much of the prior state to keep vs overwrite. Linear-time in sequence length, constant memory per token for the recurrent variant. Clover-35B-A3B uses GDN (adapted to non-causal within-block attention) as the mixer in a block-diffusion LM.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md) · [block-diffusion-lm](block-diffusion-lm.md)

---

## What it is

Softmax attention is `O(N²)` in sequence length. Linear-attention variants replace softmax with a kernel so the cumulative state can be maintained incrementally, reducing cost to `O(N)` with a constant-size state `S ∈ R^{d_k × d_v}` acting as an associative memory.

Gated Delta Networks extend the linear-attention family with two ideas:

1. **Delta rule update** — when a new key `k_t, v_t` arrives, first *read out* the current best value for `k_t` from `S`, then *write* a correction: `S ← S + (v_t − S k_t) k_t^T`. This is the classical Hebbian/delta rule for associative memories.

2. **Gating** — a learned per-step gate `g_t` modulates how much of the prior state to retain vs replace: `S ← g_t · S + (1 − g_t) · (…update…)`. The gate is content-dependent (a function of the current input) and gives the model control over memory horizon.

---

## How it works

**State-space view.** A GDN layer maintains a recurrent state `S_t ∈ R^{d_k × d_v}`. Per token:

```
k_t, v_t, q_t, g_t = proj(x_t)            # linear projections
S_t = g_t ⊙ S_{t-1} + (1 − g_t) ⊙ Δ(k_t, v_t, S_{t-1})
y_t = S_t q_t
```

Where `Δ` is the delta-rule update. For autoregressive generation, this is a constant-time recurrence per step with constant memory.

**Chunk-parallel training.** For training, the recurrence can be expressed in chunk-parallel form over a block of tokens using tensor contractions — giving GPU-efficient training similar to Mamba or GLA. The gate `g_t` makes the chunked form slightly more delicate than vanilla linear attention but still tractable.

**Non-causal variant for block diffusion.** The recurrence is causal by construction. For use inside [block-diffusion-lm](block-diffusion-lm.md), where within-block attention must be bidirectional, Clover's adaptation either (a) runs a forward and backward GDN pass and combines them, or (b) treats the within-block positions as a parallel update (all tokens visible at once, delta rule applied as a batched associative-memory write). The preview report does not fully specify which variant is used.

---

## Why it matters

GDN gives the modeling expressiveness of attention (associative lookup) with the complexity of a recurrent model. Compared to:

- **Mamba / SSM**: GDN's delta rule writes content-dependent key/value pairs, closer to attention's associative structure than pure state-space mixing.
- **Linear attention (basic)**: GDN's gating fixes the "state grows unbounded" failure mode that vanilla linear attention suffers on long contexts.
- **Full softmax attention**: `O(N)` vs `O(N²)`, at the cost of a bounded-capacity state — wins on long context, loses on tasks requiring precise many-key recall.

Its use in Clover at MoE frontier scale is evidence it scales. Previously GDN had been mostly research-scale.

---

## Gotchas & tricks

- **Gate initialization.** Starting with `g ≈ 1` (keep state) is often necessary — otherwise the state churns and training fails. Many implementations warm up the gate.
- **State capacity is bounded.** `d_k × d_v` controls how much associative-memory content fits. On recall-heavy evals, GDN underperforms softmax attention; hybrid (GDN + sparse attention) architectures are common.
- **Non-causal adaptation is non-trivial.** For block diffusion, naively running forward-then-backward doubles compute; parallel-update forms can lose expressive power. This is an active area; Clover's specific variant is not fully disclosed.
- **Chunk-size trade-off.** Training in chunks balances parallelism and recurrence overhead; the sweet spot depends on sequence length and hardware.

---

## Sources

- Paper (GDN origin): *Gated Delta Networks* — prior work, cited by Clover.
- Paper (first frontier-adjacent deployment): *Clover-35B-A3B-Preview Technical Report* — Chen et al., DiffuSpace, 2026 — [arXiv:2611.24636](https://arxiv.org/abs/2611.24636)
- Related mixer families: Mamba (SSM), GLA, RWKV, Based — surveyed in the linear-attention / state-space literature.
