# Auxiliary-Loss-Free Load Balancing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A load-balancing mechanism for MoE routers that **avoids auxiliary losses entirely**. Instead of adding a term to the training loss that penalizes imbalanced routing, maintain a per-expert **bias** that's added to the router's affinity scores for routing decisions only (not for gating weights). The bias is updated by a simple control loop — overloaded experts get their bias reduced, underloaded ones get theirs increased — and receives **no gradient**. Introduced in a DeepSeek paper (2024) and used in DeepSeek-V3.

**Prereqs:** [_moe](_moe.md), [deepseek-moe](deepseek-moe.md)
**Related:** [load-balancing-loss](load-balancing-loss.md) · [sequence-wise-balance-loss](sequence-wise-balance-loss.md)

---

## What it is

MoE routers need load balancing. Without it, a few experts get all the tokens while the rest starve. The classical fix (from GShard and Switch Transformer) is the [load-balancing auxiliary loss](load-balancing-loss.md) — a term added to the training objective that penalizes imbalanced token dispatch.

That works, but has a subtle cost: the gradient of the aux loss **flows back through the router** and fights with the main task loss. The router is pushed to balance load even when that means routing a token to a worse-fitting expert. The effect on quality is real (measurable in ablations) but hard to diagnose — models train stably, they just plateau lower.

Aux-loss-free balancing removes the loss term entirely and replaces it with a **per-expert bias** that acts only on routing decisions, not gradients.

---

## How it works

### The routing scheme

For each routed expert `i`, maintain a scalar bias `b_i` (initialized to 0). During routing:

```
# Affinity score (unchanged)
s_{i,t} = sigmoid( h_t · e_i )

# Biased score — used ONLY to pick which experts to route to
s̃_{i,t} = s_{i,t} + b_i

# Top-K selection on the biased score
S_t = { top-K_r experts by s̃_{i,t} }

# Gating weights — use the UNBIASED score, normalized
g_{i,t} = s_{i,t} / Σ_{j ∈ S_t} s_{j,t}     for i ∈ S_t
```

The key split: **bias affects selection, not weighting**. Once an expert is in the top-K, its gate weight comes from the clean affinity score, so the output mixture is driven by what the router actually thinks is the best match.

### The bias update

After each step, count how many tokens were routed to each expert (`n_i`) and compare to the ideal balanced count `n̄ = B · K_r / N` (where `B` is batch size). Update biases:

```
if n_i > n̄:   b_i ← b_i - γ         # expert overloaded, make it less attractive
if n_i < n̄:   b_i ← b_i + γ         # expert underloaded, make it more attractive
```

`γ = 0.001` in DeepSeek-V3, for the first 14.3T tokens of training; then `γ = 0.0` for the final 500B (so the final model has frozen biases).

**The bias receives no gradient.** It's not a parameter in the usual sense — it's a control-loop variable that the training framework updates by hand.

### Why the bias doesn't wreck the gate weights

Because gating weights are computed from the unbiased score `s_{i,t}`, the main task loss sees only the natural expert affinity. The bias shifts *which* K experts get picked but not *how they're mixed* once picked. The router's representation of affinity stays clean.

### Pairs with a small sequence-wise guardrail

Aux-loss-free balancing is a **batch-level** mechanism: the bias update after each step uses whole-batch token counts. It can leave single sequences with concentrated routing even while global balance looks fine. DeepSeek-V3 pairs aux-loss-free with a very small [sequence-wise balance loss](sequence-wise-balance-loss.md) (`α = 10⁻⁴`) as a safety net against per-sequence pathologies. The sequence-wise loss is a distinct mechanism — documented separately, not part of aux-loss-free itself.

---

## Why it matters

- **Removes router gradient interference.** The main task loss no longer has to share the router's gradient with a balance term. Ablations in the DeepSeek paper show a consistent ~1% downstream benchmark improvement at equal compute vs the aux-loss version.
- **Tuning-free.** `γ = 0.001` works across model sizes. No coefficient to sweep, no schedule to design.
- **Composable with any router.** Works with sigmoid-per-expert (DeepSeekMoE) or softmax routing. The bias is just a pre-top-K shift.
- **Costs one scalar per expert per layer.** For 256 experts × 58 MoE layers that's ~15K scalars total — rounding error.

---

## Gotchas & tricks

- **Only bias the selection, never the gate weights.** If you add the bias to both (as a first implementation mistake), you've re-introduced the gradient-interference problem through a side door, because gate weights flow into the task loss.
- **Update `γ` scheduling.** Too large and the bias oscillates (an expert flips from overloaded to underloaded every step). Too small and it can't track rapid router shifts at the start of training. `γ = 0.001` is empirical; don't go above 0.01.
- **Freeze the bias near the end of training.** DeepSeek-V3 drops `γ` to 0 for the last 500B tokens so the final checkpoint has fixed biases. Otherwise, last-batch stochasticity leaves the final model with a slightly perturbed bias vector.
- **Per-sequence pathologies can still slip through.** The bias operates at batch scope; a single long sequence can route lopsidedly even when global balance is fine. Pair with a small [sequence-wise balance loss](sequence-wise-balance-loss.md) when this matters.
- **Count the right thing.** The bias update uses `n_i` = number of tokens routed to expert `i` in this step. Using gate-weighted counts (e.g. `Σ_t g_{i,t}`) or probability masses (`Σ_t s_{i,t}`) gives subtly different control dynamics — stick with raw token counts.
- **Batch size matters.** Smaller batches have noisier `n_i`, so the control loop has more jitter. At very small batch sizes (`B · K_r < N`, so on average not every expert even gets a token) the bias update can go unstable — consider accumulating counts over multiple steps before updating.
- **Not the same as router z-loss.** ST-MoE's router z-loss bounds the *magnitude* of router logits for stability; aux-loss-free balancing bounds the *per-expert load*. Orthogonal concerns; can be combined, but DeepSeek doesn't.
- **No equivalent of this for Expert Choice routing.** Expert Choice balances by construction (each expert picks exactly K tokens), so the bias trick is unnecessary there.

---

## Sources

- Paper: *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* — Wang et al., DeepSeek, 2024 — the original proposal with ablations.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — large-scale application with `γ = 0.001`, paired with a small sequence-wise guardrail.
- Paper: *Switch Transformers* — Fedus et al., 2021 — the auxiliary-loss baseline this replaces. See [load-balancing-loss](load-balancing-loss.md).
- Paper: *ST-MoE* — Zoph et al., 2022 — auxiliary losses for stability (orthogonal but contrasting).
