# Load-Balancing Auxiliary Loss
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The classical way to keep MoE routers from collapsing: add a small **auxiliary term** to the training loss that pushes toward **uniform expert usage**. The loss is $\alpha \cdot N \cdot \sum_i f_i \cdot P_i$ (Switch) — where $f_i$ is the fraction of tokens the router *dispatched* to expert $i$ and $P_i$ is the mean *routing probability* the router assigned to expert $i$. Minimizing this pair product drives both statistics toward $1/N$. Computed **per MoE layer** and summed into the total loss. Introduced in GShard (Lepikhin 2020) and cleaned up in Switch Transformer (Fedus 2021).

**Prereqs:** [_moe](_moe.md), [deepseek-moe](deepseek-moe.md)
**Related:** [capacity-factor](capacity-factor.md) · [sequence-wise-balance-loss](sequence-wise-balance-loss.md) · [aux-loss-free-balancing](aux-loss-free-balancing.md)

---

## What it is

A sparse-MoE router is trained end-to-end with the rest of the network. Left to its own devices, it collapses: a few experts win, the rest starve (no tokens → no gradient → no recovery). The standard fix since GShard has been to add an auxiliary loss term to the total training objective that explicitly penalizes imbalance.

The load-balancing loss is the oldest and simplest mechanism. It's been superseded at frontier scale by [aux-loss-free balancing](aux-loss-free-balancing.md), but it's still the starting point for anyone implementing MoE and the baseline every new balancer is compared to.

---

## How it works

### The Switch Transformer form (clearest statement)

For a single MoE layer processing a batch $B$ with $T$ total tokens and $N$ experts:

$$
\mathrm{loss\_aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

$$
f_i = \frac{1}{T} \sum_{x \in B} \mathbf{1}\{ \arg\max p(x) = i \} \quad \text{(fraction of tokens dispatched to expert } i\text{)}
$$

$$
P_i = \frac{1}{T} \sum_{x \in B} p_i(x) \quad \text{(mean routing probability for expert } i\text{)}
$$

where $p(x) \in \mathbb{R}^N$ is the softmax router output for token $x$, and $p_i(x)$ is its $i$-th coordinate.

**Under uniform routing**, both $f_i = 1/N$ and $P_i = 1/N$, so $\sum_i f_i \cdot P_i = N \cdot (1/N)^2 = 1/N$. The explicit $\cdot N$ scale factor up front cancels this, leaving $\mathrm{loss\_aux} = \alpha$ under uniform load — a constant that doesn't depend on $N$. That's the reason $N$ appears: it makes the loss value comparable across model configurations with different expert counts.

### The GShard form (earlier, per-group)

GShard partitions the batch into $G$ groups of size $S$ each, applies the router group-locally, and computes the loss per group (Algorithm 1):

$$
\ell_{\mathrm{aux}} = \frac{1}{E} \sum_{e=1}^{E} \frac{c_e}{S} \cdot m_e
$$

$$
c_e = \text{number of tokens (in the group) dispatched to expert } e
$$

$$
m_e = \frac{1}{S} \sum_{s=1}^{S} g_{s, e} \quad \text{(mean gate value for expert } e\text{)}
$$

Same functional form as Switch's ($c_e/S$ plays the role of $f_e$, $m_e$ plays the role of $P_e$); different constants and per-group instead of per-batch. Switch is the cleaner presentation; GShard is the historical origin.

### Why the product $f_i \cdot P_i$

The *ideal* objective is to minimize $\sum_i f_i^2$ — when $\sum_i f_i = 1$, this is minimized at $f_i = 1/N$ (uniform dispatch). But $f_i$ contains a hard **argmax** over expert scores, which has zero gradient almost everywhere. The router would see no signal.

GShard's trick (direct quote from the paper): *"we use the mean gates per expert $m_e$ as a differentiable approximation and replace $(c_e/S)^2$ with $m_e \cdot (c_e/S)$, which can now be optimized with gradient descent."*

Reasoning: in expectation, a well-calibrated router has $f_i \approx P_i$ (the fraction of tokens it dispatches matches its mean probability). So $f_i \cdot P_i \approx P_i^2$ — and $P_i$ is differentiable. The gradient flows through $P_i$ only; $f_i$ acts as a *detached weighting* telling the gradient which direction to push router probabilities. Concretely, $\partial \mathrm{loss} / \partial p_i(x) = \alpha \cdot N \cdot f_i / T$ — larger for experts that were over-dispatched, smaller for underused ones, so the router's softmax is pushed down on overloaded experts and up on underused ones.

### Scope: per layer, per batch

**Per MoE layer.** Switch Transformer is explicit (§2.2): *"For each Switch layer, this auxiliary loss is added to the total model loss during training."* Each MoE layer runs its own router, computes its own $f_i$ / $P_i$ over the same batch of tokens, and produces its own per-layer $\mathrm{loss\_aux}$. The per-layer losses are summed into the total training objective. There is **no** global $f_i$ / $P_i$ shared across layers — each layer has its own router with its own expert set.

**Per batch (not per sequence).** $f_i$ and $P_i$ sum over the entire batch of $T$ tokens — mixing tokens from many sequences into the same balance statistic. This is a **loose** constraint: balance is enforced on average over the batch, but a single sequence can still concentrate all its tokens on a few experts as long as *other sequences in the batch* use the remaining experts. The [sequence-wise variant](sequence-wise-balance-loss.md) tightens this to per-sequence statistics.

**GShard is per group.** GShard's tokens are partitioned into groups; each group computes its own $\ell_{\mathrm{aux}}$. A group can span one sequence or multiple sequences depending on grouping strategy. Less common today — most implementations follow Switch's per-batch convention.

**Ambiguity flag:** neither GShard nor Switch explicitly states how per-layer losses are combined into the total. "Added to the total" is standard English for summation; the implementation convention is always $\sum_{\text{layers}} \mathrm{loss\_aux}_{\text{layer}}$, but if you're reading either paper carefully, note this is assumed rather than stated.

### Full training objective

$$
L_{\mathrm{total}} = L_{\mathrm{CE}} + \sum_{\ell \in \text{MoE layers}} \alpha \cdot N \cdot \sum_i f_i^{(\ell)} \cdot P_i^{(\ell)}
$$

Switch uses $\alpha = 10^{-2}$ (§2.2, explicit hyperparameter sweep between $10^{-1}$ and $10^{-5}$). GShard states only "a constant multiplier $k$" without a numeric value.

---

## Why it matters

- **Prevents router collapse.** Without some form of balancing, sparse MoE routers converge to using a tiny subset of experts. The aux loss is the simplest mechanism that reliably prevents this.
- **Cheap to implement.** A few extra ops per MoE layer (one argmax, one softmax mean, one dot product). No extra parameters, no control loops.
- **Well-understood failure modes.** A decade of ablations: we know it interferes with task gradients, we know roughly by how much (~1% benchmark drop vs a perfectly-balanced oracle), we know it scales to at least Mixtral / GShard sizes.
- **The baseline for every new balancer.** When a paper claims a new load-balancing scheme, the comparison is against this.

---

## Gotchas & tricks

- **Gradient interference is real.** The aux loss's gradient competes with the task loss's gradient at the router. At small scale this is negligible; at 100B+ it's a measurable quality hit. This is the headline reason [aux-loss-free balancing](aux-loss-free-balancing.md) was proposed.
- **Balance is averaged, not enforced.** $f_i = 1/N$ across the batch does not mean each sequence is balanced; one sequence can be all-expert-7 as long as another is all-expert-3. For fine-grained-MoE where per-sequence skew matters, add a [sequence-wise term](sequence-wise-balance-loss.md) or switch mechanisms.
- **Don't drop the $\cdot N$ factor.** Some implementations lose the leading $N$ in Eq. 4 from Switch, giving a loss that scales as $1/N$ under uniform routing. The $\cdot N$ exists to keep the loss value comparable across $N$.
- **$\arg\max$ vs top-$K$.** The formula as written uses top-1 routing (Switch). For top-$K$, $f_i$ becomes the fraction of tokens dispatched to expert $i$ among all dispatches (not tokens), with an extra $K$ factor in the denominator to preserve the $1/N$ uniform target. Implementations vary; check the exact formula you're comparing against.
- **Capacity-factor drops do NOT enter $f_i$.** $f_i$ counts routed-before-drop tokens. If you use a capacity factor $< \infty$, dropped tokens are still counted as "dispatched to expert $i$" for load-balance purposes even though they receive no compute.
- **Coefficient $\alpha$ is surprisingly insensitive** — Switch swept $10^{-1}$ to $10^{-5}$ and found $10^{-2}$ worked. At frontier scale where gradient interference matters, people often push $\alpha$ lower ($10^{-3}$) to trade some balance quality for less interference.
- **It's a per-layer loss, not a global one.** A model with 32 MoE layers has 32 aux losses summed into the total, not one aux loss computed from concatenated statistics across layers. Each layer has its own router and its own balance pressure.
- **Doesn't help with the cold-start problem.** Early in training, the router's softmax is near-uniform, so all experts get similar dispatch rates and the aux loss is near zero. Once some experts win, the loss starts doing work. If an expert "dies" completely (gets near-zero gradient for many steps), the aux loss alone may not revive it — reviving needs token-level initialization tricks or periodic expert reset.

---

## Sources

- Paper: *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* — Lepikhin et al., 2020 — Sec 2.2, Algorithm 1 line 13. Introduces the $m_e \cdot (c_e/S)$ differentiable surrogate.
- Paper: *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* — Fedus et al., 2021 — Sec 2.2, Eqs. 4–6. The clean $\alpha \cdot N \cdot \sum f_i P_i$ statement with $\alpha = 10^{-2}$.
- Paper: *ST-MoE: Designing Stable and Transferable Sparse Expert Models* — Zoph et al., 2022 — documents coefficient sensitivity and combines aux loss with router z-loss.
- Paper: *DeepSeekMoE* — Dai et al., 2024 — adapts the same formula with per-sequence statistics (see [sequence-wise-balance-loss](sequence-wise-balance-loss.md)).
