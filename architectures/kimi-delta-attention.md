# Kimi Delta Attention (KDA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A linear-RNN attention variant whose recurrent state update combines a **delta-rule transition** (rank-one correction to a diagonal state) with a **channel-wise scalar gate**. Efficient (linear in sequence length, diagonal-plus-rank-one state transitions) and stable (transitions are non-expansive), used in the Kimi line of models as a Transformer-attention alternative.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md) · [complex-kda.md](complex-kda.md) · [../case-studies/kimi-k1-5.md](../case-studies/kimi-k1-5.md)

---

## What it is

A **linear RNN** parametrized so that each per-step state transition is a **diagonal-plus-rank-one** matrix — cheap to apply, easy to make stable — plus a **channel-wise gate** in $[0, 1]$ that scales each state dimension before the delta-rule correction.

At step $t$, the state $S_t \in \mathbb{R}^{d\times d}$ (or an equivalent per-channel vector representation) evolves as:

$$
S_t = \mathrm{diag}(g_t)\,S_{t-1} \;+\; \beta_t\,(v_t - S_{t-1}\,k_t)\,k_t^\top
$$

where $k_t, v_t$ are per-step "key/value" vectors, $g_t \in [0,1]^d$ is the gate, and $\beta_t \in [0,1]$ is the delta-rule coefficient controlling how much of the residual $(v_t - S_{t-1}\,k_t)$ to write. Read-out is a linear projection $y_t = S_t\,q_t$.

## How it works

**Delta rule** = a rank-one correction toward the target $v_t$ along direction $k_t$, with the current estimate $S_{t-1}\,k_t$ subtracted first (that's the "delta"). This is the same mechanism as fast weights / Hebbian associative memory, adapted for linear-time streaming.

**Diagonal gate** decays each state channel independently — a per-channel forgetting factor. Combined with the delta rule, the update stays *diagonal-plus-rank-one*:

$$
S_t = \underbrace{\mathrm{diag}(g_t)}_{\text{diagonal}} \, S_{t-1} \;+\; \underbrace{(\beta_t\,v_t - \beta_t\,S_{t-1}k_t)\,k_t^\top}_{\text{rank-one}}
$$

which means:

- **Cost:** $O(d^2)$ per step to represent and $O(d)$ per step for a properly-parameterized version, vs. $O(N \cdot d)$ per token for Transformer attention.
- **Total sequence cost** is $O(N \cdot d^2)$ — linear in sequence length $N$, matching linear-attention.
- **Stability:** with $g_t \in [0,1]$ and $\beta_t \in [0,1]$, transitions are non-expansive (spectral radius $\leq 1$), so states don't blow up.

Because each transition is diagonal-plus-rank-one, its eigenstructure is constrained: a **single** KDA step can only realize transitions in that specific matrix class. To get a genuine 2D rotation you need to chain two delta-rule updates (DeltaProduct₂) — or use CKDA (the sibling paper's trick).

## Why it matters

- **Linear-time, streaming.** No attention cache that grows with $N$; state is a fixed-size $d \times d$ matrix (or its structured equivalent).
- **Stable by construction.** No RoPE-style decay tuning, no attention-sink hacks — the update itself is non-expansive.
- **Practical.** Used in production Kimi models; matches or exceeds Transformer perplexity at comparable parameter counts in the reported settings.
- **Composable.** The gate/coefficient parameterization is a natural handle for expressivity extensions (see CKDA).

## Gotchas & tricks

- **Parameter ranges dictate expressivity.** Restricting $g_t \in [0, 1]$ and $\beta_t \in [0, 1]$ makes the model stable but caps the class of transitions it can realize. Widening the ranges (e.g. $g_t \in [-1, 1]$) unlocks reflection-and-rotation composites — this is exactly the CKDA extension.
- **Read-out depth.** A single KDA layer has bounded state-tracking expressivity; deeper stacks recover more, but at compute cost. Trade layers against expressivity via CKDA-style extensions.
- **Not a drop-in for attention in every setting.** Tasks that need long-range *exact* retrieval (needle-in-a-haystack) still favor attention or hybrid stacks. Hybrid Transformer + linear-RNN layouts remain common.
- **State reset semantics.** Between documents you typically reset $S$ to zero (or a learned prior) — carrying state across unrelated documents will silently corrupt the readout.

## Sources

- Paper: *Kimi K2 / Kimi Delta Attention* — Kimi team, 2025 — the KDA formulation used in production Kimi models.
- Paper: *Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention* — Siems et al., 2026 — a formal expressivity study of KDA and its natural extension. [arXiv:2609.24797](https://arxiv.org/abs/2609.24797)
- Background: DeltaNet / DeltaProduct — the delta-rule linear-RNN line KDA extends.
