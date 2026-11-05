# Muon-MoE Routing Interaction
*Depth — how the Muon optimizer systematically perturbs first-MoE-layer routing bias dynamics, and the partial fixes available.*

**TL;DR:** A controlled study on two MoE language models (55M and 108M active params, FineWeb-Edu) showing that **Muon systematically raises the span of the first MoE layer's routing bias at every seed**, while AdamW does not. The rise follows Muon-updated weights on the rest of the network, and the first router's inputs share a common direction. **Halving Muon's learning rate removes most of the rise**; moving **only** the router to AdamW shifts loads more during warmup and slightly raises the loss. First empirical story on how optimizer choice shifts MoE load-balancing dynamics *before* any balancing loss kicks in.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../architectures/aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md)
**Related:** [../architectures/load-balancing-loss.md](../architectures/load-balancing-loss.md) · [_training-stability.md](_training-stability.md)

---

## What it is

Muon (Keller Jordan 2024) is an optimizer that orthogonalizes the update step via a Newton-Schulz iteration before applying it to 2D parameter matrices. It has become a popular default for frontier training over AdamW. Independently, [aux-loss-free load balancing](../architectures/aux-loss-free-balancing.md) and related mechanisms shape MoE routing via a per-expert bias on the router's affinity scores.

This paper documents a specific **interaction**: Muon, applied to standard MoE training, systematically raises the dynamic range of the *first* MoE layer's routing bias — the layer where tokens are first routed to experts. The effect is reproducible across seeds and model sizes.

---

## How it works

### The observed phenomenon

Pretrain two MoE models (55M / 108M active) on FineWeb-Edu under matched conditions, varying only the optimizer (Muon vs AdamW). Measure the **span of the first-MoE-layer routing bias** $b^{(1)}$ — the difference between the largest and smallest expert bias — across training steps.

- Under AdamW: bias span grows gradually and plateaus.
- Under Muon: bias span rises **faster and farther**, at every seed. The gap is especially pronounced in early training.

### Diagnosis — three findings

1. **Rise follows Muon on the rest of the network.** When the first router alone is switched to AdamW while everything else stays on Muon, the rise shrinks but doesn't vanish — the router inherits an inputs distribution shaped by Muon-updated layers.
2. **First router's inputs share a direction.** The hidden states entering the first MoE's router have a common dominant direction. Muon's orthogonalized updates preserve/amplify this direction, which compounds into the bias span.
3. **Halving Muon's LR fixes most of it.** Reducing Muon's learning rate by 2× recovers AdamW-like bias-span dynamics at a mild loss cost.

### The naive fix is not a Pareto improvement

Moving **only** the router (not the rest of the network) to AdamW:
- Shifts expert loads more during warmup (undesirable).
- Slightly raises the training loss (also undesirable).

So "just use AdamW for routers" doesn't clean up the interaction without side effects.

---

## Why it matters

- **Optimizer choice is architecturally entangled.** The Muon-vs-AdamW debate has mostly been framed as "which converges faster on language loss." This paper shows that at the MoE scale that everyone is moving toward, optimizer choice also perturbs routing dynamics — and routing dynamics drive quality via expert specialization.
- **Partial recipe for teams pairing Muon with DeepSeekMoE-style MoE.** Halve the LR in the first MoE layers, or at least measure the bias-span effect. A silent training issue worth catching early.
- **Clean mechanistic story.** Three findings together (rise, shared-direction inputs, LR sensitivity) make a coherent explanation rather than a bare observation.
- **A bachelor's thesis as primary source.** The paper is small-scale (55M/108M active) and methodologically focused. The mechanism generalization to frontier scale (671B DeepSeek-V3-class) is an empirical open question.

---

## Gotchas & tricks

- **First layer specifically.** The paper's findings are about the *first* MoE layer. Later MoE layers see hidden states that have passed through load-balanced routing and don't exhibit the same shared-direction effect.
- **Scale extrapolation uncertain.** 55M / 108M active is a small-scale study; frontier-scale (37B+ active) may amplify or mask the effect. Measure before committing to a fix.
- **"Halve Muon LR" is a global knob.** It affects all Muon-updated parameters, not just the router. Prefer per-layer LR scaling (half the first-MoE-layer Muon LR, keep others) as a cleaner intervention.
- **Watch the router bias span directly.** Easy metric to add — log the bias span per MoE layer every $N$ steps. The paper's finding is that this span is a reliable early warning.
- **Compatible with aux-loss-free balancing.** The aux-loss-free scheme already updates the router bias via a control loop; the Muon-driven span rise is *separate* from that intervention and lives in the pre-update bias state. Both can be in play simultaneously.

---

## Sources

- Paper: *Muon Makes the First Mixture-of-Experts Layer Harder to Balance* — Anonymous (bachelor's thesis, Universidad Autónoma de Madrid, supervised by Daniel Hernández-Lobato), 2026, arXiv 2610.37914 — primary source.
- Related: *Muon: An Optimizer for Hidden Layers in Neural Networks* — Keller Jordan, 2024 — the optimizer under study.
- Related: [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) — the bias-based balancing scheme that Muon perturbs.
