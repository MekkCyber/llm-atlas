# Optimizers for LLM Pretraining

*Taxonomy — the first-order and quasi-second-order optimizers used for pretraining LLMs, and how to choose.*

**TL;DR:** The LLM-pretraining optimizer landscape has three layers: **AdamW** (the universal default, well understood), the **Shampoo line** (Shampoo → SOAP → FOAM) which approximates a second-order update via Kronecker-factored Gram matrices, and **Muon** (muon-style updates on hidden layers, AdamW on embeddings) which sits structurally between them. The frontier question is no longer "does second-order beat AdamW on loss" — it does, by small but consistent margins — but "is the extra memory and per-step cost worth it at frontier scale." FOAM is the 2026 move that makes that answer closer to yes.

**Related taxonomies:** [_training-stability](_training-stability.md) · [_lr-schedules](_lr-schedules.md)
**Depth files covered here:** [foam](foam.md)

---

## The problem

Pretraining a frontier LLM is dominated by one choice per weight matrix: *given this gradient, how do I update the weight.* AdamW is a diagonal-preconditioning first-order method; Shampoo-style methods are block-diagonal second-order approximations. More structure in the preconditioner → better per-step progress → fewer steps to a target loss, in exchange for more memory and more compute per step.

Three failure modes any optimizer choice has to survive:

1. **Loss spikes at scale.** Some optimizers interact badly with the stability stack (QK-norm, z-loss) — see [_training-stability](_training-stability.md).
2. **LR sensitivity.** Need to work across a wide range because sweeping LR at frontier scale is expensive.
3. **Memory footprint.** Optimizer state can be 3–8× the model weight memory, which pushes sharding/FSDP strategy.

## The shared pattern

Every LLM pretraining optimizer is

$$
W \leftarrow W - \eta \cdot P(G, \text{state}) \cdot M^{-1}(G, \text{state})
$$

where $P$ is an *update direction* (gradient, momentum, Shampoo-preconditioned gradient) and $M$ is an *element- or block-scale* that normalizes magnitudes (Adam's $\sqrt{V}$, Shampoo's inverse root). Variants differ on three axes:

1. **What preconditioner $P$ uses.** None (SGD), diagonal (AdamW), Kronecker block (Shampoo/SOAP/FOAM), orthogonalized (Muon).
2. **How the preconditioner's state is maintained.** Full matrix (expensive), factored (cheap), amortized eigendecomposition (periodic).
3. **How scale is tracked.** Adam-style per-element second moment, factored second moment (Adafactor, FOAM), or embedded in the preconditioner (Shampoo original).

## Variants

| Technique | Preconditioner | Memory vs AdamW | Per-step cost vs AdamW | Who uses it |
| --- | --- | --- | --- | --- |
| **AdamW** | Diagonal (per-element) | 1× | 1× | Everyone, as the default |
| **Adafactor** | Diagonal, factored | **0.5×** | ~1× | Memory-constrained pretraining (T5, PaLM) |
| **Shampoo** | Kronecker block (full $L, R$) | ~3× | ~1.5× + periodic eigendecomp | Research runs, Google in production |
| **SOAP** | Kronecker block + Adam scale | ~3× | ~1.5× + two eigendecomps | Scaling-RL runs, recent Shampoo successor |
| **[FOAM](foam.md)** | One-sided Kronecker + factored Adam scale | ~1.5× | ~1.1× + one eigendecomp | Ships competitive with Muon per step; emerging alternative |
| **Muon** | Orthogonalized hidden layers + AdamW on embeddings | ~1.2× | ~1.1× | Modern frontier open recipes |

## How to choose

- **New open pretraining recipe, from scratch, 10B+ scale.** AdamW is still the safe default. Muon is a reasonable modern pick if you're comfortable with its hybrid structure (needs AdamW on embeddings).
- **You want better per-step progress and can afford the memory.** SOAP or [FOAM](foam.md). FOAM is strictly cheaper than SOAP with matched quality on 124M–760M sweeps, so FOAM is the preferred Shampoo-line entry in 2026.
- **Memory-constrained** (large model, no FSDP headroom). Adafactor or FOAM — both halve the second-moment footprint.
- **Fine-tuning rather than pretraining.** AdamW. Second-order methods' advantage on a short fine-tune run is usually not worth the implementation cost.

## Adjacent but distinct

- **[_lr-schedules](_lr-schedules.md).** Orthogonal axis — any optimizer can pair with any schedule.
- **[_training-stability](_training-stability.md).** Optimizer choice interacts with the stability stack (e.g., Muon's orthogonalization already bounds certain update magnitudes, so one usually leaves gradient clipping at 1.0 regardless).
- **Distributed strategies** (ZeRO, FSDP, tensor-parallel). The optimizer's state size determines what sharding options are feasible.

## Sources

- Paper: *Adam: A Method for Stochastic Optimization* — Kingma & Ba, 2014.
- Paper: *Decoupled Weight Decay Regularization* — Loshchilov & Hutter, 2017 — defines AdamW.
- Paper: *Adafactor* — Shazeer & Stern, 2018 — factored second moment.
- Paper: *Shampoo: Preconditioned Stochastic Tensor Optimization* — Gupta et al., 2018.
- Paper: *SOAP: Improving and Stabilizing Shampoo using Adam* — 2024.
- Paper: *Muon* — 2024.
- Paper: *FOAM* — Yu & Han, 2026 — https://arxiv.org/abs/2611.16905
