# Loopback Transformer (three-knob looped transformer)
*Depth — a looped Transformer that decouples three previously entangled controls — input strength, total update weight, per-update allocation — into separate scalars.*

**TL;DR:** Looped Transformers reuse a block multiple times to trade parameter count for effective depth. Prior variants conflate three design choices: how much of the original input each loop sees, how much total change the loop can apply to the residual, and how that total is split across iterations. The Loopback Transformer exposes all three as independent scalars, letting them be tuned separately. Reports lower validation loss and higher downstream accuracy than standard and prior looped residual baselines at smaller parameter counts.

**Prereqs:** [./transformer-block.md](./transformer-block.md)
**Related:** [./README.md](./README.md)

---

## What it is

A Transformer architecture that runs the same block `L` times with scalar re-weighting at each pass. Parameter count is fixed; effective depth varies with `L`. The Loopback factorization isolates the knobs that previous looped designs had entangled.

## How it works

Let `h_0` be the embedding, `f(h)` be the block's update, and `h_L` be the final hidden. At each iteration `ℓ = 1..L`:

```
h_ℓ = (1 - α_input) · h_{ℓ-1} + α_input · h_0 + β_ℓ · f(h_{ℓ-1})
```

where:
- **`α_input`** — *input strength*. How much of the original embedding is re-injected each loop. `α_input = 0` is the pure residual chain; `α_input > 0` keeps the input visible across all iterations.
- **`Σ β_ℓ` = β_total** — *total update weight*. How much the loop is allowed to rewrite the residual across all `L` steps combined.
- **`β_ℓ / β_total`** — *per-update allocation*. How the total budget is distributed over iterations (uniform, front-loaded, back-loaded, learned).

Prior variants fix two of three or tie them together. Loopback trains all three independently and finds the knee experimentally.

## Why it matters

Looped designs keep resurfacing as a parameter-efficient depth lever — Universal Transformer, ALBERT, deep-equilibrium models. The literature's cross-comparisons were muddled because different papers set the three knobs implicitly. A clean factorization makes it possible to ask which knob a given task benefits from, and gives looped transformers a fairer shot against deep residual baselines.

Practical implications:
- **Parameter-efficient depth** for memory-constrained inference (edge, on-device).
- **Adaptive compute**: vary `L` per token or per layer without retraining.
- Likely useful for reasoning-heavy tasks where extra effective depth helps more than extra parameters.

## Gotchas & tricks

- **Initialization matters.** Looped blocks compound updates; standard Transformer init can diverge quickly at higher `L`. Report uses scaled init.
- **Per-update allocation.** Uniform `β_ℓ` is a safe default; learned allocation sometimes helps but adds optimization instability.
- **Not a free speedup.** Smaller parameter count ≠ faster inference — `L` serial passes still cost `L` block forwards. The win is memory, not latency.
- Fair comparisons to standard deep Transformers must match `L · P_block` FLOPs, not just parameter count.

## Sources

- Paper: *Loopback Transformer: Towards a Unified View of Residual Looped Transformers* — Ouyang, Cai, Zhang, Luo, Wan, Hu (UW–Madison), 2026 — [arXiv:2611.07226](https://arxiv.org/abs/2611.07226)
