# MPI Router (Manifold Power Iteration)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A redesign of MoE router weights that **constrains each router row to track the principal singular direction of its assigned expert**. Standard routers learn the router matrix freely; MPI Router updates each row toward `argmax‖Wᵢ v‖` (the top singular direction of expert $i$) via a *Power-then-Retract* step — one power-iteration update on the router weights using the expert matrix, followed by a norm-preserving retraction. Empirically improves MoE pretraining quality from 1B to 11B parameters at matched FLOPs.

**Prereqs:** [_moe.md](_moe.md), [deepseek-moe.md](deepseek-moe.md)
**Related:** [load-balancing-loss.md](load-balancing-loss.md) · [aux-loss-free-balancing.md](aux-loss-free-balancing.md) · [sequence-wise-balance-loss.md](sequence-wise-balance-loss.md)

---

## What it is

In a standard MoE, the router is a linear map $h_t \cdot e_i$ where $e_i$ is the router row "representing" expert $i$. There's no design principle for what $e_i$ *should* look like — it's trained from scratch alongside everything else. MPI Router argues that the most expressive single-vector summary of expert $i$'s weight matrix $W_i$ is its **principal singular direction** (the right-singular vector with the largest singular value), because that direction captures the most variance in $W_i$'s action on inputs. So the router row $e_i$ should be aligned with that direction.

This is purely a router-side change. Expert layout, top-$K$ routing, and load balancing are all untouched.

---

## How it works

### Power-then-Retract update

Every few training steps, apply one step of **power iteration** on the router row using the corresponding expert weights:

```
e_i ← W_iᵀ W_i · e_i        # one power-iteration step toward principal singular direction
e_i ← e_i / ‖e_i‖ · r       # retraction: rescale to a fixed-norm sphere of radius r
```

The retraction is the "Manifold" part: it constrains $e_i$ to live on a fixed-norm manifold, so the power-iteration update can't blow up the router-row magnitude (which would dominate gradient dynamics). The combined step is a Riemannian-style update on a sphere.

### Theoretical claim

Repeated Power-then-Retract steps drive $e_i$ to converge to the principal right-singular vector of $W_i$. The paper proves this convergence under the assumption that the gradient updates to $W_i$ from the task loss are bounded — so the singular structure changes slowly relative to the power-iteration rate.

### Why this should help routing

The router's job is to compute a token-expert affinity. If $e_i$ is the principal singular direction, then $h_t \cdot e_i$ is large exactly when the expert's response $W_i h_t$ would be large — the score literally tracks "how much does this expert care about this token." Whereas a freely-learned $e_i$ has no such guarantee; it's just a vector that empirically separates tokens well enough.

---

## Why it matters

- **Free quality bump.** Same expert layout, same FLOPs, better routing → consistently better pretraining quality from 1B to 11B params.
- **Principled.** Most MoE design space is engineering empirics (top-1 vs top-8, capacity factor, balance loss coefficient). MPI Router gives a *mathematical* statement of what router rows should be — closer to first-principles design.
- **Composable.** Orthogonal to load balancing (works with aux-loss, aux-loss-free, or sequence-wise). Orthogonal to expert granularity (fine-grained or coarse). Drops into any existing MoE pipeline.
- **Cheap.** One power-iteration step per router row is $O(d^2)$ for FFN width $d$ — tiny compared to forward/backward through the experts.

---

## Gotchas & tricks

- **Power-iteration frequency matters.** Too often (every step) and the router can't adapt to changing task signal — it just chases the singular direction. Too rare and the router drifts off-manifold between updates. The paper sweeps this; treat as a hyperparameter.
- **The retraction norm $r$ is load-bearing.** Without a fixed norm, the power-iteration step grows $\|e_i\|$ unboundedly, and softmax-style routing collapses to one expert. The retraction is what makes the manifold update stable.
- **No interaction with load balancing.** The bias from [aux-loss-free balancing](aux-loss-free-balancing.md) and the MPI alignment update are mechanically independent — bias acts on selection scores, MPI shapes the underlying router rows. Combine freely.
- **Expert weights move too.** Both $e_i$ and $W_i$ are trained; MPI assumes the gradient updates to $W_i$ are slow relative to the alignment updates. If you crank the learning rate hard, the principal singular direction can shift faster than the router can track.

---

## Sources

- Paper: *Redesign Mixture-of-Experts Routers with Manifold Power Iteration* — Wu, Lv, Xie, Lin (Renmin U / Tencent), 2026 — [arXiv 2606.12397](https://arxiv.org/abs/2606.12397).
- Paper: *DeepSeekMoE* — Dai et al., 2024 — the fine-grained MoE design MPI Router benefits most from.
