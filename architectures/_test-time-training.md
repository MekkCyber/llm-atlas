# Test-time training (TTT)
*Taxonomy — sequence-model families that update fast weights via an inner learning rule at inference time.*

**TL;DR:** TTT rewrites sequence modeling as an *online-learning* problem: for each token, an inner learner takes a training step on some auxiliary loss and produces fast-weight updates that condition the next forward pass. The family unifies DeltaNet / Gated DeltaNet / RWKV-7 / TTT-Linear as different (fast-weight-network, inner-loss, learning-rule) triples. TTT is one of the strongest current attention-alternative families for long context.

**Related taxonomies:** [_normalization.md](_normalization.md), [_moe.md](_moe.md)
**Depth files covered here:** [modular-ttt.md](modular-ttt.md)

---

## The problem

Softmax attention is $O(L^2)$ in sequence length; state-space models (SSMs) trade expressivity for linearity; linear attention variants lose retrieval sharpness. TTT is an alternative recurrent formulation: run a *small optimization problem* per token to produce fast weights, then use those fast weights as the recurrent state. Expressivity comes from the inner learner rather than from softmax or SSM parameterization.

## The shared pattern

Every TTT variant defines:
1. **A fast-weight network $W_t$** — the parameters updated per token (small, e.g. a low-rank matrix or a shallow MLP).
2. **An inner loss** — the objective the inner learner minimizes at each step (MSE, inner-product, contrastive).
3. **An inner learning rule** — how $W_t$ is updated from $W_{t-1}$ using the current token's inner-loss gradient (SGD, Adam-lite, delta rule).
4. **A query readout** — how the current token's query reads from the updated $W_t$ to produce the output.

The forward pass at token $t$ is: (a) compute inner loss on the current token given $W_{t-1}$; (b) apply the inner learning rule to produce $W_t$; (c) read $W_t$ with the query to emit the output. This is causal by construction.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| DeltaNet (Yang et al., 2024) | Delta-rule inner update: additive rank-1 correction per token | Simple; no gating | Baseline TTT variant |
| Gated DeltaNet (Yang et al., 2024) | Adds forget gate to delta update | More expressive, ~SOTA in family | Long-context sequence modeling |
| TTT-Linear (Sun et al., 2024) | Full inner SGD step on a linear fast-weight net | Cheaper than TTT-MLP; less capacity | Efficient long-context |
| [modular-ttt](modular-ttt.md) | DAG framework; ablate every axis | No new SOTA vs Gated DeltaNet; systematic study | Understanding what matters |
| RWKV-7 (2025) | Time-decay + delta hybrid | Battle-tested but idiosyncratic | Production TTT-family deployments |

## How to choose

The **modern default** for a fresh TTT deployment is Gated DeltaNet (or the equivalent Modular-TTT best-variant configuration). Deep fast-weight networks and heavy normalization *hurt* — the Modular TTT ablations point to small LR + weight decay + a single-layer nonlinearity as the sweet spot. If you're using TTT as one component of a hybrid architecture (attention + TTT layers), keep the TTT layers shallow and well-normalized.

For research on new inner learners, work in the Modular TTT DAG substrate — it's the cleanest ablation surface.

## Adjacent but distinct

- **SSMs (Mamba, S4)** — recurrent but with a fixed linear state-space parameterization, not an inner learner.
- **Linear attention** — recurrent decomposition of softmax attention; no per-token optimization loop.
- **Fast-weight programmers (Schmidhuber, 1992)** — the historical ancestor; TTT variants inherit the fast-weight idea but with modern parameterizations.

## Sources

- Paper: *Modular TTT: Rethinking Test-Time Training as Composable Modules* — Tang et al., 2026 — arXiv:2608.07110.
- Paper: *Test-Time Training with Self-Supervised Learning* — Sun et al., 2020.
- Paper: *Gated Linear Attention Transformers with Hardware-Efficient Training / Gated DeltaNet* — Yang et al., 2024.
