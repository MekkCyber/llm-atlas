# Optimizers

*Taxonomy — the algorithm that turns a gradient into a weight update.*

**TL;DR:** Three eras: SGD (with momentum) for vision and early NLP, **AdamW** as the default for Transformer pretraining since GPT-3, and **Muon** as the emerging 2D-weight optimizer that beats AdamW at LLM pretraining by ~2× wall-clock. Modern recipes mix optimizers along the parameter axis (Muon for matrices, AdamW for embeddings / norms) rather than treating "the optimizer" as a single global choice.

**Related taxonomies:** [_lr-schedules](../pre-training/_lr-schedules.md) · [_training-stability](../pre-training/_training-stability.md)
**Depth files covered here:** [muon](muon.md)

---

## The problem

A gradient tells you the steepest *local* descent direction; turning it into a *good* update means picking a step size, a smoothing/preconditioning strategy, and a way to handle different parameter shapes and scales. Get this wrong and you either crawl (under-shoot) or blow up (over-shoot on sharp curvature directions). The optimizer is the algorithm that resolves all of those at once.

## The shared pattern

All practical LLM optimizers are first-order methods with **momentum** plus some form of **preconditioning** (diagonal, full-matrix, or matrix-aware) that adapts step size per direction. They differ in *what they precondition by* and *what shape of parameter they treat as a unit*.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| SGD + momentum | Polyak momentum over raw gradients; no preconditioning | Sensitive to LR & curvature anisotropy | CNNs, small models with tuned schedule |
| Adam / AdamW | Diagonal preconditioning by EMA of squared gradients; decoupled weight decay (W) | Per-coordinate scaling ignores matrix structure | Transformer pretraining default since GPT-3 |
| Lion | EMA + sign step (no second moment); memory-light | Discrete-sign step loses some adaptivity | Vision Transformers; memory-tight runs |
| Shampoo | Full-matrix preconditioning via Kronecker factors of $G G^\top$, $G^\top G$ | Compute and memory cost > diagonal methods | Large 2D weights when you can pay the precon cost |
| [muon](muon.md) | Orthogonalize the momentum buffer (Newton–Schulz), apply only to 2D weights | Needs gather for sharded matrices; LR transfer differs | LLM pretraining of 2D weights; pair with AdamW for 1D |

## How to choose

- **Default:** AdamW on every parameter. Boring, predictable, well-understood scaling laws.
- **If you want pretraining speedup and can re-tune LR:** Muon on the 2D weight matrices (attention, MLP, output projection) + AdamW on the 1D parameters (embeddings, RMSNorm scales, biases). This is the modern recipe — see [muon](muon.md) for the curvature reason.
- **If memory is the binding constraint:** Lion or AdamW with `bf16` master weights. Lion drops the second-moment buffer entirely.
- **If you're chasing per-step efficiency at huge matrix size:** Shampoo — but compute and engineering cost is steep, and Muon usually catches up at lower complexity.

Combining is the norm: nobody runs one optimizer on every parameter once you're past trivial scale.

## Adjacent but distinct

- **LR schedules** — picks the magnitude trajectory $\eta_t$, not the direction. See [_lr-schedules](../pre-training/_lr-schedules.md).
- **Weight decay / regularization** — orthogonal axis to the optimizer (AdamW decouples it explicitly; SGD bundles it via L2).
- **Mixed precision** — about *where* arithmetic happens, not what direction the update points. See [fp8-training.md](../pre-training/fp8-training.md).

## Sources

- *Adam: A Method for Stochastic Optimization* — Kingma & Ba, 2014.
- *Decoupled Weight Decay Regularization* — Loshchilov & Hutter, 2017 — the W in AdamW.
- *Symbolic Discovery of Optimization Algorithms* — Chen et al., 2023 — Lion.
- *Why Muon Outperforms Adam: A Curvature Perspective* — Wang et al., 2026, arXiv 2606.04662.
