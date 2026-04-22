# Load-Balancing Auxiliary Loss
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The classical way to keep MoE routers from collapsing: add a small **auxiliary term** to the training loss that pushes toward **uniform expert usage**. The loss is `α · N · Σ_i f_i · P_i` (Switch) — where `f_i` is the fraction of tokens the router *dispatched* to expert `i` and `P_i` is the mean *routing probability* the router assigned to expert `i`. Minimizing this pair product drives both statistics toward `1/N`. Computed **per MoE layer** and summed into the total loss. Introduced in GShard (Lepikhin 2020) and cleaned up in Switch Transformer (Fedus 2021).

**Prereqs:** [_moe](_moe.md), [deepseek-moe](deepseek-moe.md)
**Related:** [capacity-factor](capacity-factor.md) · [sequence-wise-balance-loss](sequence-wise-balance-loss.md) · [aux-loss-free-balancing](aux-loss-free-balancing.md)

---

## What it is

A sparse-MoE router is trained end-to-end with the rest of the network. Left to its own devices, it collapses: a few experts win, the rest starve (no tokens → no gradient → no recovery). The standard fix since GShard has been to add an auxiliary loss term to the total training objective that explicitly penalizes imbalance.

The load-balancing loss is the oldest and simplest mechanism. It's been superseded at frontier scale by [aux-loss-free balancing](aux-loss-free-balancing.md), but it's still the starting point for anyone implementing MoE and the baseline every new balancer is compared to.

---

## How it works

### The Switch Transformer form (clearest statement)

For a single MoE layer processing a batch `B` with `T` total tokens and `N` experts:

```
loss_aux = α · N · Σ_{i=1..N}  f_i · P_i

f_i = (1/T) · Σ_{x ∈ B}  1{ argmax p(x) = i }       ← fraction of tokens dispatched to expert i
P_i = (1/T) · Σ_{x ∈ B}  p_i(x)                      ← mean routing probability for expert i
```

where `p(x) ∈ R^N` is the softmax router output for token `x`, and `p_i(x)` is its `i`-th coordinate.

**Under uniform routing**, both `f_i = 1/N` and `P_i = 1/N`, so `Σ_i f_i · P_i = N · (1/N)² = 1/N`. The explicit `·N` scale factor up front cancels this, leaving `loss_aux = α` under uniform load — a constant that doesn't depend on `N`. That's the reason `N` appears: it makes the loss value comparable across model configurations with different expert counts.

### The GShard form (earlier, per-group)

GShard partitions the batch into `G` groups of size `S` each, applies the router group-locally, and computes the loss per group (Algorithm 1):

```
ℓ_aux = (1/E) · Σ_{e=1..E}  (c_e / S) · m_e

c_e = number of tokens (in the group) dispatched to expert e
m_e = (1/S) · Σ_{s=1..S}  g_{s,e}            ← mean gate value for expert e
```

Same functional form as Switch's (`c_e/S` plays the role of `f_e`, `m_e` plays the role of `P_e`); different constants and per-group instead of per-batch. Switch is the cleaner presentation; GShard is the historical origin.

### Why the product `f_i · P_i`

The *ideal* objective is to minimize `Σ_i f_i²` — when `Σ_i f_i = 1`, this is minimized at `f_i = 1/N` (uniform dispatch). But `f_i` contains a hard **argmax** over expert scores, which has zero gradient almost everywhere. The router would see no signal.

GShard's trick (direct quote from the paper): *"we use the mean gates per expert `m_e` as a differentiable approximation and replace `(c_e/S)²` with `m_e · (c_e/S)`, which can now be optimized with gradient descent."*

Reasoning: in expectation, a well-calibrated router has `f_i ≈ P_i` (the fraction of tokens it dispatches matches its mean probability). So `f_i · P_i ≈ P_i²` — and `P_i` is differentiable. The gradient flows through `P_i` only; `f_i` acts as a *detached weighting* telling the gradient which direction to push router probabilities. Concretely, `∂loss/∂p_i(x) = α · N · f_i / T` — larger for experts that were over-dispatched, smaller for underused ones, so the router's softmax is pushed down on overloaded experts and up on underused ones.

### Scope: per layer, per batch

**Per MoE layer.** Switch Transformer is explicit (§2.2): *"For each Switch layer, this auxiliary loss is added to the total model loss during training."* Each MoE layer runs its own router, computes its own `f_i` / `P_i` over the same batch of tokens, and produces its own per-layer `loss_aux`. The per-layer losses are summed into the total training objective. There is **no** global `f_i` / `P_i` shared across layers — each layer has its own router with its own expert set.

**Per batch (not per sequence).** `f_i` and `P_i` sum over the entire batch of `T` tokens — mixing tokens from many sequences into the same balance statistic. This is a **loose** constraint: balance is enforced on average over the batch, but a single sequence can still concentrate all its tokens on a few experts as long as *other sequences in the batch* use the remaining experts. The [sequence-wise variant](sequence-wise-balance-loss.md) tightens this to per-sequence statistics.

**GShard is per group.** GShard's tokens are partitioned into groups; each group computes its own `ℓ_aux`. A group can span one sequence or multiple sequences depending on grouping strategy. Less common today — most implementations follow Switch's per-batch convention.

**Ambiguity flag:** neither GShard nor Switch explicitly states how per-layer losses are combined into the total. "Added to the total" is standard English for summation; the implementation convention is always `Σ_layers loss_aux_layer`, but if you're reading either paper carefully, note this is assumed rather than stated.

### Full training objective

```
L_total = L_CE  +  Σ_{ℓ ∈ MoE layers}  α · N · Σ_i  f_i^(ℓ) · P_i^(ℓ)
```

Switch uses `α = 10⁻²` (§2.2, explicit hyperparameter sweep between `10⁻¹` and `10⁻⁵`). GShard states only "a constant multiplier `k`" without a numeric value.

---

## Why it matters

- **Prevents router collapse.** Without some form of balancing, sparse MoE routers converge to using a tiny subset of experts. The aux loss is the simplest mechanism that reliably prevents this.
- **Cheap to implement.** A few extra ops per MoE layer (one argmax, one softmax mean, one dot product). No extra parameters, no control loops.
- **Well-understood failure modes.** A decade of ablations: we know it interferes with task gradients, we know roughly by how much (~1% benchmark drop vs a perfectly-balanced oracle), we know it scales to at least Mixtral / GShard sizes.
- **The baseline for every new balancer.** When a paper claims a new load-balancing scheme, the comparison is against this.

---

## Gotchas & tricks

- **Gradient interference is real.** The aux loss's gradient competes with the task loss's gradient at the router. At small scale this is negligible; at 100B+ it's a measurable quality hit. This is the headline reason [aux-loss-free balancing](aux-loss-free-balancing.md) was proposed.
- **Balance is averaged, not enforced.** `f_i = 1/N` across the batch does not mean each sequence is balanced; one sequence can be all-expert-7 as long as another is all-expert-3. For fine-grained-MoE where per-sequence skew matters, add a [sequence-wise term](sequence-wise-balance-loss.md) or switch mechanisms.
- **Don't drop the `·N` factor.** Some implementations lose the leading `N` in Eq. 4 from Switch, giving a loss that scales as `1/N` under uniform routing. The `·N` exists to keep the loss value comparable across `N`.
- **`argmax` vs `top-K`.** The formula as written uses top-1 routing (Switch). For top-K, `f_i` becomes the fraction of tokens dispatched to expert `i` among all dispatches (not tokens), with an extra `K` factor in the denominator to preserve the `1/N` uniform target. Implementations vary; check the exact formula you're comparing against.
- **Capacity-factor drops do NOT enter `f_i`.** `f_i` counts routed-before-drop tokens. If you use a capacity factor < ∞, dropped tokens are still counted as "dispatched to expert `i`" for load-balance purposes even though they receive no compute.
- **Coefficient `α` is surprisingly insensitive** — Switch swept `10⁻¹` to `10⁻⁵` and found `10⁻²` worked. At frontier scale where gradient interference matters, people often push `α` lower (`10⁻³`) to trade some balance quality for less interference.
- **It's a per-layer loss, not a global one.** A model with 32 MoE layers has 32 aux losses summed into the total, not one aux loss computed from concatenated statistics across layers. Each layer has its own router and its own balance pressure.
- **Doesn't help with the cold-start problem.** Early in training, the router's softmax is near-uniform, so all experts get similar dispatch rates and the aux loss is near zero. Once some experts win, the loss starts doing work. If an expert "dies" completely (gets near-zero gradient for many steps), the aux loss alone may not revive it — reviving needs token-level initialization tricks or periodic expert reset.

---

## Sources

- Paper: *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* — Lepikhin et al., 2020 — Sec 2.2, Algorithm 1 line 13. Introduces the `m_e · (c_e/S)` differentiable surrogate.
- Paper: *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* — Fedus et al., 2021 — Sec 2.2, Eqs. 4–6. The clean `α · N · Σ f_i P_i` statement with `α = 10⁻²`.
- Paper: *ST-MoE: Designing Stable and Transferable Sparse Expert Models* — Zoph et al., 2022 — documents coefficient sensitivity and combines aux loss with router z-loss.
- Paper: *DeepSeekMoE* — Dai et al., 2024 — adapts the same formula with per-sequence statistics (see [sequence-wise-balance-loss](sequence-wise-balance-loss.md)).
