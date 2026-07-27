# Hybrid Linear–Softmax Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Pure linear attention scales as $O(N)$ but loses full-rank token interactions, hurting quality on long-sequence tasks (notably video). Pure softmax attention preserves quality but costs $O(N^2)$. **Hybrid Linear–Softmax Attention** interleaves gated linear-attention blocks with periodic gated-softmax "anchor" blocks at a fixed ratio (e.g. 3:1 in SANA-Video 2.0), restoring the full-rank interactions the linear blocks lack while paying softmax cost only 1 layer in 4.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](./multi-head-attention.md).
**Related:** [mla.md](./mla.md) · [transformer-block.md](./transformer-block.md)

---

## What it is

An attention-layer schedule that alternates two block types:

- **Gated linear-attention block**: $O(N)$ token mixing via kernel-feature-map linear attention with an input-dependent gate.
- **Gated softmax-attention block ("anchor")**: standard multi-head softmax attention at $O(N^2)$, also gated.

Blocks are interleaved on a fixed periodic schedule (SANA-Video 2.0 uses **3 linear : 1 softmax**), so long-sequence memory/compute is dominated by the linear blocks while the anchors restore the exact token interactions linear attention approximates.

## How it works

Each transformer layer is either linear or softmax, with the mix determined by the schedule. For a 24-layer model at 3:1, layers 4, 8, 12, 16, 20, 24 are softmax; the rest are linear. The gates on both variants condition on the input, letting the model modulate how much the block contributes at each position.

The "attention residuals" framing: linear attention is a low-rank approximation of softmax attention. The softmax anchor blocks add back the residual — the interactions the linear blocks miss — every 4 layers. This is cheaper than making every layer softmax (4× the cost) but preserves quality much better than making every layer linear (no restoration path).

## Why it matters

Video generation is where softmax attention's quadratic cost bites hardest — sequence lengths of $10^5$+ tokens are routine, and full softmax attention throughout is infeasible on a single GPU. Fully-linear architectures (Mamba, RWKV) trade quality for scaling. The hybrid schedule is a **practical middle ground**: single-GPU 720p video at DiT-level quality, at 5B and 14B scale, using conventional DiT infrastructure.

The pattern also composes: any transformer facing long-sequence compute pressure (video, audio, long-context language) can adopt it without rewriting the training stack.

## Gotchas & tricks

- **Ratio is task-dependent.** SANA-Video finds 3:1 works for 720p video. Language tasks may need different ratios; too few softmax anchors and quality collapses back to pure-linear levels.
- **Anchor placement matters.** Uniform periodic placement is a strong baseline; adaptive placement (put softmax where the loss says it's needed) is an obvious extension.
- **Gates are load-bearing.** Ungated linear attention is meaningfully worse in the hybrid schedule; the input-dependent gate is what lets the linear blocks specialize.
- **Distinct from MLA / GQA.** MLA and GQA reduce the *memory* of softmax attention (via latent projections or head grouping); hybrid linear–softmax reduces the *compute* of long-sequence mixing by replacing most layers with a cheaper operator entirely.

## Sources

- Paper: *SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation* — Chen, Yu, Li, Xue, Liu et al., NVIDIA, 2026 — [arXiv:2607.21553](https://arxiv.org/abs/2607.21553)
- Code: [github.com/NVlabs/Sana](https://github.com/NVlabs/Sana)
