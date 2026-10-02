# Looped Transformer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A transformer where some block (or stack of blocks) is applied repeatedly $R$ times, reusing the same weights, before producing output. Trades parameter count for depth: $R \times$ the compute, $1 \times$ the weights. Combines cleanly with MoE: Chen et al. (Meta, 2026) fit joint scaling laws for sparsity × recurrence and find roughly 3× active-parameter efficiency from sparsity and 2× total-parameter efficiency from recurrence on reasoning, with the two composing.

**Prereqs:** [transformer-block.md](./transformer-block.md), [_moe.md](./_moe.md).
**Related:** [deepseek-moe.md](./deepseek-moe.md), [../pre-training/mtp.md](../pre-training/mtp.md).

---

## What it is

A standard transformer stacks $L$ distinct blocks. A looped transformer stacks $L_{\text{unique}}$ blocks and applies them $R$ times, giving effective depth $L_{\text{eff}} = R \cdot L_{\text{unique}}$. Hidden states are forwarded through the loop, allowing iterative refinement — closer in spirit to a universal transformer than a dense feed-forward stack.

The appeal: more compute per token (good for reasoning tasks that benefit from sequential refinement) at a fraction of the parameter cost (good for serving).

When combined with MoE, you get three orthogonal knobs: **active parameters** (compute-per-token via dense dim), **total parameters** (expert count × dim), and **recurrence depth** (loop count $R$).

---

## How it works

Let $f_\theta$ be a block (or several). Standard transformer:

$$
h^{(L)} = f_{\theta_L} \circ f_{\theta_{L-1}} \circ \cdots \circ f_{\theta_1}(x)
$$

Looped transformer with recurrence $R$ on the first $B$ blocks:

$$
h^{(0)} = x; \quad h^{(r)} = f_\theta(h^{(r-1)}) \quad \text{for } r = 1, \ldots, R
$$

followed by the remaining non-looped blocks. The loop block $f_\theta$ typically has its own residual/layer-norm structure so iterating stays stable.

**Joint scaling with MoE.** Chen et al. fit:

$$
L(N, D, S, R) = f_1(N, D) + f_2(S) + f_3(R) + \text{interaction}
$$

over params $N$, data $D$, sparsity $S$ (= experts activated per total experts), recurrence $R$. The result: additive gains in log-loss from each knob, with modest interaction terms.

**Empirical headline:**
- Sparsity gives ≈ 3× active-parameter efficiency (as expected from MoE literature).
- Recurrence gives ≈ 2× total-parameter efficiency on reasoning.
- Combined > either alone.

---

## Why it matters

- **Serving efficiency.** $R \times$ compute, $1 \times$ weights means lower memory per unit of reasoning depth. Matters for KV-cache-constrained long-context serving.
- **Reasoning gains.** Recurrence appears to selectively help reasoning benchmarks over knowledge-recall benchmarks — consistent with iterative refinement being the mechanism.
- **Composes with MoE.** The two efficiency levers are independent; frontier labs can pick $(S, R)$ along a Pareto frontier for their compute budget.
- **Revives a dormant architecture family.** Universal transformers (Dehghani et al., 2018) were set aside at scale; this gives a modern scaling-laws justification for picking them back up when combined with MoE.

---

## Gotchas & tricks

- **Instability at high $R$.** Repeated application of the same block can amplify numerical drift; QK-norm and reordered-norm help. Chen et al.'s stable training recipe isn't fully specified in the abstract.
- **Not all blocks loop well.** Attention blocks loop cleanly; FFN blocks sometimes collapse. Looping a block that includes an MoE routing step changes expert utilization — the paper's scaling law handles this but implementations should audit per-expert load after unrolling.
- **$R$ is a *compute* knob, not a *parameter* knob.** You can set it at inference time without retraining, within limits; too far from training-time $R$ degrades.
- **Interaction with KV cache.** Recurrent reuse of the same block multiplies attention compute but not KV cache (the cache is per-layer in parameter space, not effective-depth space). This is part of why serving cost is favorable.

---

## Sources

- Paper: *Scaling Laws for Looped Mixture of Experts* — Chen, Goyal, Krishnamoorthi, Meta, 2026. arXiv:2609.40316.
- Prior art: *Universal Transformers* — Dehghani et al., 2018 — the original recurrence-over-blocks idea.
