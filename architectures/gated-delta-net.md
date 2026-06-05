# Gated Delta Network (GDN)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Gated Delta Network is a sub-quadratic linear-attention architecture in the Mamba family: a recurrent state with a *delta-rule* update and an output gate. Each token writes a low-rank update to a fixed-size state, then reads from it through a gated projection. Trains and runs in linear time and constant memory in the sequence length, but — until recently — without principled scaling rules.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [mla.md](mla.md), [multi-head-attention.md](multi-head-attention.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

Transformers compute attention in $O(L^2)$ FLOPs and store a KV cache that grows linearly with sequence length. Linear-attention architectures replace softmax attention with a recurrent state $S_t$ of fixed size $d \times d$:

- **Read:** $y_t = q_t^\top S_t$  (or a gated version of this).
- **Write:** $S_{t+1} = f(S_t, k_t, v_t)$  where $f$ is a recurrent update.

Gated Delta Network specifies $f$ as a **delta-rule** update — a low-rank correction $\eta_t (v_t - k_t^\top S_t) k_t^\top$ that nudges $S_t$ to better predict $v_t$ from $k_t$ — combined with an *output gate* $g_t$ that controls how much of the read is passed downstream. This is in the lineage of DeltaNet, Mamba-2, and RWKV-7.

## How it works

The full GDN block, for hidden state $h_t$:

1. **Projections.** $q_t, k_t, v_t, g_t = W_{q,k,v,g} h_t$. Gate $g_t$ has its own activation (sigmoid).
2. **Delta-rule state update.**
   $$S_{t+1} = S_t (I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$
   where $\beta_t \in (0,1)$ is a learnable write strength. This *overwrites* the old value associated with key $k_t$ — unlike additive linear attention, which only adds new associations.
3. **Read with gate.** $y_t = g_t \odot (q_t^\top S_t)$.
4. **Output projection** to the residual stream.

In compute terms, each step is $O(d^2)$ FLOPs (constant in $L$), and total memory is also $O(d^2)$ regardless of sequence length. Long-context inference is therefore free relative to context size — the bottleneck moves from KV cache to state quality.

## Why it matters

- **Sub-quadratic.** Long-context inference is constant memory per layer; quadratic attention's KV cache wall disappears.
- **Better than additive linear attention.** The delta rule lets the state *forget* old associations rather than accumulating them, which historically was where linear-attention models lost to softmax attention on long-context retrieval.
- **Practical scaling now possible.** With μP derived for GDN (Liu & Gu, 2026), learning-rate sweeps transfer across widths zero-shot, removing the main barrier to scaling these models past Transformer baselines.

## Gotchas & tricks

- **Initialization is delicate.** The recurrent state's spectral radius must stay $\leq 1$ for stable rollouts; the standard parametrization can drift, motivating μP-style coordinate-checking.
- **Gating fights state forgetting.** The output gate $g_t$ can suppress useful signal that the delta state actually stored; tuning $g_t$ activation is non-trivial.
- **Hybrid wins.** As with Mamba-2, pure-GDN models often underperform hybrid Transformer-GDN stacks on hard retrieval; one or two softmax-attention layers per N GDN layers recovers most of the gap.

## Sources

- Paper: *Unlocking Feature Learning in Gated Delta Networks at Scale* — Liu, Gu, 2026 — [arXiv:2606.04048](https://arxiv.org/abs/2606.04048) — derives μP for GDN.
- Related: DeltaNet (Schlag et al., 2021); Mamba-2 (Dao & Gu, 2024); RWKV-7.
