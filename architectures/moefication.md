# MoEfication — Converting Dense LLMs to MoE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** "MoEfication" is the post-hoc conversion of a pretrained dense LLM into a sparse Mixture-of-Experts model — partitioning each FFN's neurons into expert groups so inference activates only a subset per token. Earlier methods used **heuristic neuron clustering** (group by co-activation) or **random splits**; both are static and lossy. **DOT-MoE** (2026) formulates the partition as a *Differentiable Optimal Transport* problem solved with Sinkhorn-Knopp iterations under straight-through estimators, jointly learning the neuron-to-expert assignment and the token-to-expert router end-to-end. Retains 90% of the dense model's performance at 50% active parameters.

**Prereqs:** [_moe](_moe.md), [load-balancing-loss](load-balancing-loss.md)
**Related:** [aux-loss-free-balancing](aux-loss-free-balancing.md), [deepseek-moe](deepseek-moe.md)

---

## What it is

MoE from scratch is unstable and compute-intensive — most labs would rather convert an already-trained dense model. The conversion centers on the FFN sub-layer: each FFN's $4d$ inner neurons are partitioned into $E$ experts of $4d/E$ neurons each, plus a learned router that picks top-$K$ experts per token. The resulting MoEfied model has the same total parameter count as the dense model but activates only $K/E$ of the FFN per token.

Two coupled discrete decisions need to be made:

1. **Static partition:** which input neuron belongs to which expert (one-time).
2. **Dynamic routing:** which token visits which expert (per forward pass).

Earlier MoEfication methods made (1) heuristically (cluster neurons by co-activation patterns; or random) and trained (2) afterwards. DOT-MoE makes both decisions *jointly differentiable*.

---

## How it works

### The DOT-MoE formulation

Treat the neuron-to-expert assignment as a **balanced transport plan** $\Pi \in \mathbb{R}^{4d \times E}_+$ where rows sum to 1 (each neuron is assigned somewhere) and columns sum to $4d/E$ (every expert has equal capacity). Solve with **Sinkhorn-Knopp iterations**: alternate row and column normalizations of $\exp(\text{logits})$ until convergence. Sinkhorn iterations are differentiable, so the assignment logits get gradients from downstream task loss.

**Straight-through estimators (STE):** in the forward pass, harden $\Pi$ to a one-hot assignment $\hat\Pi = \mathrm{argmax}_{\text{column}}(\Pi)$; in the backward pass, pretend the soft $\Pi$ was used. This lets the discrete assignment train end-to-end against the cross-entropy / distillation loss.

### Joint routing + assignment

Both the **router** $r_\phi$ (token-to-expert) and the **assignment matrix** $\Pi$ are trained simultaneously. The Sinkhorn constraint on $\Pi$ enforces *expert capacity at the parameter level* (every expert gets equal neuron budget), while the router's auxiliary load-balance loss enforces *expert utilization at runtime* (every expert receives roughly equal token traffic). The two constraints together prevent expert collapse.

### Training recipe

- Initialize $\Pi$ from a heuristic baseline (random or co-activation clustering) for warm-start.
- Fine-tune jointly on a calibration / distillation corpus; loss = task CE + distillation KL against the original dense model.
- Anneal Sinkhorn entropy regularizer (low entropy → harder assignments) over training.
- Active parameters: 50% of dense (top-$K = E/2$ in their experiments).

---

## Why it matters

- **Principled alternative to heuristic clustering.** Existing MoEfication baselines (Zhang 2022 co-activation clustering, random splits) are static — once you've partitioned, you can't change your mind. DOT-MoE keeps the partition trainable until you commit.
- **Sinkhorn replaces aux losses for the *partition* step.** Just as [aux-loss-free balancing](aux-loss-free-balancing.md) replaced auxiliary load-balance losses for *routing*, Sinkhorn capacity constraints provide a gradient-free way to enforce per-expert capacity at the *parameter* level.
- **90% retention at 50% active params.** Best reported MoEfication efficiency-quality tradeoff; outperforms structured pruning, heuristic-clustering MoEfication, and random splits on the same backbone and benchmarks.
- **Path to sparse inference for any existing dense model.** Most production LLMs are still dense; MoEfication is the cheapest way to recover dense quality at sparse-MoE inference cost without a from-scratch pretraining run.

---

## Gotchas & tricks

- **Sinkhorn entropy matters.** Too entropic and the assignment is soft (no sparsity benefit); too sharp early and gradients vanish. Anneal.
- **STE bias.** The forward/backward asymmetry of straight-through introduces gradient bias; can manifest as slow convergence in the assignment matrix late in training. A small temperature on the soft assignment helps.
- **Capacity is enforced strictly.** Unlike a soft auxiliary loss, Sinkhorn keeps experts *exactly* equal. This is a constraint, not a regularizer — if your task benefits from unequal experts, DOT-MoE will hurt.
- **Distillation loss is the right calibration signal.** Pure CE on calibration data has lots of variance; KL against the dense model's full distribution provides a denser signal that the assignment can exploit.
- **Not the same as upcycling.** Upcycling (Komatsuzaki et al., 2023) initializes an MoE *from* a dense model and trains further from scratch on MoE-shaped data; MoEfication *partitions* the dense FFN and keeps weights local to their assigned expert. Upcycling = re-train; MoEfication = re-partition.
- **Compatible with quantization.** Each expert can be quantized independently after the partition is fixed; doesn't interact pathologically with GPTQ/AWQ on the experts (each expert is a smaller dense FFN).

---

## Sources

- Paper: *DOT-MoE: Differentiable Optimal Transport for MoEfication* — 2026 — [arXiv:2606.01666](https://arxiv.org/abs/2606.01666).
- Background: *MoEfication: Transformer Feed-forward Layers are Mixtures of Experts* — Zhang et al., 2022 — original heuristic-clustering MoEfication.
- Background: *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints* — Komatsuzaki et al., ICLR 2023 — the alternative dense → MoE route.
- Foundational: *Sinkhorn Distances: Lightspeed Computation of Optimal Transport* — Cuturi, NeurIPS 2013 — the differentiable Sinkhorn-Knopp algorithm DOT-MoE applies.
