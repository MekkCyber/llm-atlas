# Variable-Width Transformer (><former)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Drop the assumption that all transformer layers have the same residual-stream width. The "><former" keeps early and late layers wide, narrows the middle, and joins them with a **parameter-free residual resizing** so no extra weights are introduced. Across 200M → 3B (dense + MoE) it consistently beats uniform-width baselines at matched loss while spending **22% fewer FLOPs** and using **15% less KV cache** memory.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_moe](_moe.md), [mla](mla.md), [looped-transformer](looped-transformer.md)

---

## What it is

For nearly every modern transformer, $d_{\text{model}}$ is constant across depth — the same residual width carries from layer 1 to layer $L$. This is a default, not a result: every layer gets an equal slice of parameters and FLOPs regardless of what it does.

The ><former ("bowtie") schedule says: **early layers (embedding shaping) and late layers (LM-head readout) need width; middle layers can be narrow**. The width profile dips in the middle and recovers, joined by a parameter-free linear resize on the residual stream.

---

## How it works

### Width schedule

For an $L$-layer model, width $d_\ell$ follows a bowtie:

$$
d_\ell = \begin{cases} d_{\text{wide}} & \ell \le \ell_{\text{down}} \\ d_{\text{narrow}} & \ell_{\text{down}} < \ell < \ell_{\text{up}} \\ d_{\text{wide}} & \ell \ge \ell_{\text{up}} \end{cases}
$$

Typical ratios reported in the paper: $d_{\text{narrow}} / d_{\text{wide}} \approx 0.5\text{–}0.7$, with the narrow band covering the middle ~50% of layers.

### Parameter-free residual resizing

When the residual stream needs to enter a narrower (or wider) layer, the resize is a **fixed, parameter-free projection**:

$$
r' = R \cdot r, \quad R \in \mathbb{R}^{d_{\ell+1} \times d_\ell}
$$

where $R$ is a deterministic matrix (e.g., low-rank random projection or block-diagonal selection) chosen so that **the resize introduces no trainable parameters**. Inverse resize on the way back out uses $R^\top$ (or the explicit inverse for invertible designs). Crucially, gradient flow and the residual identity property are preserved across the bottleneck.

### Where the savings come from

Per-layer FLOPs scale as $\sim L \cdot d^2$ for the FFN and $\sim L \cdot d \cdot n$ for attention. Narrowing the middle band shrinks $d^2$ for those layers; the parameter-free resize adds no FLOPs of its own. The narrower middle band also has smaller per-layer K and V, so the **KV cache** at inference shrinks proportionally.

---

## Why it matters

- **22% FLOP reduction at loss-matched scaling**, fit across decoder-only 200M → 2B dense and 3B MoE models. Direct, measurable efficiency win on the language-modeling Pareto frontier.
- **15% smaller KV cache and I/O** at inference — exactly the kind of saving that compounds for long-context serving.
- **No new parameters.** The bowtie schedule is a layout choice, not an architecture innovation in the parameter sense — easy to slot into existing training stacks.
- **Composes with MoE.** Gains hold under MoE; the narrower middle reduces expert routing cost as a side effect.

---

## Gotchas & tricks

- **The middle isn't useless — it's *bottlenecked*.** Analysis shows the narrow middle layers learn qualitatively different residual-stream representations than uniform-width models. This is the cost: less mid-layer feature capacity. The bowtie wins when readout/embedding capacity dominates, which is the regime for current decoder-only LMs.
- **Resize matrix choice matters.** Random projections work; orthogonal block-diagonal selections give cleaner gradient flow. A naive truncation breaks the residual identity and slows training.
- **Not all tasks benefit equally.** The paper measures perplexity and standard LM evals. Long-context retrieval (which leans on middle-layer attention) was not separately stressed; expect the bowtie to be less safe there.
- **Combines with [MLA](mla.md) cleanly.** MLA compresses KV at every layer; bowtie shrinks the *number of dimensions* being compressed at the middle layers. Multiplicative savings on KV are plausible but unmeasured in the paper.

---

## Sources

- Paper: *Variable-Width Transformers* — Zhaofeng Wu, Oliver Sieberling, Shawn Tan, Rameswar Panda, Yury Polyanskiy, Yoon Kim — MIT / MIT-IBM Watson AI Lab, 2026 — [arXiv:2606.18246](https://arxiv.org/abs/2606.18246).
