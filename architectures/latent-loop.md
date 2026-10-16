# LatentLoop
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Looped MoE architectures reuse the same expert stack across multiple passes to amortize parameters. LatentLoop observes that **the two loop visits prefer different residual scales**, compresses the sequence on the first visit, keeps full resolution on the second, redirects the saved compute into model width, and learns per-layer writeback coefficients to mix the two visits' residuals asymmetrically.

**Prereqs:** [_moe.md](./_moe.md), [deepseek-moe.md](./deepseek-moe.md), [transformer-block.md](./transformer-block.md)
**Related:** [capacity-factor.md](./capacity-factor.md) · [aux-loss-free-balancing.md](./aux-loss-free-balancing.md) · [mla.md](./mla.md)

---

## What it is

A looped Transformer calls the same block (same weights) more than once in a forward pass — e.g. two passes through an MoE stack before the final head. Standard looped designs run both passes **symmetrically**: same sequence length, same compute, residuals summed.

LatentLoop's claim: the two visits are *not* doing the same job. The first visit prefers coarser-grained processing (plan-like), the second visit prefers fine-grained decoding. Treating them symmetrically wastes compute. The paper proposes asymmetric allocation.

## How it works

### Asymmetric sequence length

- **First visit:** compress the sequence to a shorter latent sequence (fewer effective tokens, e.g. via learned pooling or a compressor layer). The expert stack runs on this short sequence.
- **Second visit:** full-length sequence. The expert stack runs on full tokens.

The compute saved on the first visit's $O(L_{\text{short}} \cdot H)$ attention and FFN frees budget that can be invested elsewhere.

### Compute redirected into width

The saved budget is spent on wider experts or more experts in the stack. Across 15 configurations from 1.3B to 7B parameters, this trades sequence length (first visit only) for model width (always-on).

### Learned writeback coefficients

Instead of adding the two visits' residuals 1:1, LatentLoop learns per-layer coefficients $\alpha_1, \alpha_2$ that scale each visit's contribution:
$$h_{\text{out}} = \alpha_1 \cdot h^{(1)} + \alpha_2 \cdot h^{(2)}.$$
Freely learnable; the paper reports they diverge from 0.5/0.5 toward asymmetric settings consistent with the "different-scale" interpretation.

## Why it matters

- **First design point to break looped-model symmetry.** Prior parameter-sharing architectures (Universal Transformer, ALBERT, weight-tied LMs) treated iterations as identical. LatentLoop names and measures the asymmetry.
- **Compute savings reported:** 19.0% vs vanilla MoE, 7.9% vs the looped baseline, with lower validation loss and improved average over nine benchmarks.
- **Generalizes beyond MoE.** The asymmetric-residual-scale observation applies to any parameter-sharing design; the compress-then-decode pattern is portable.

## Gotchas & tricks

- **Compression quality matters.** If the first-visit compressor loses too much, the second visit can't recover; LatentLoop uses a learned compressor rather than fixed pooling.
- **Writeback coefficients can collapse.** With no constraints, $\alpha_2 \to 1$ and $\alpha_1 \to 0$ reduces the loop to a single pass. In practice the coefficients stay in middle regimes but are worth monitoring.
- **Interaction with routing.** MoE routing on short sequences has fewer tokens to balance; expect load-balancing loss behavior to differ between the two visits.
- **Second-visit cost dominates.** Full-length second visit still bounds throughput; savings come from the first visit and from the compute moved into width.

## Sources

- Paper: *LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation* — Chai, Chen, Zhu, Wu, Luo, Li, Cheng, 2026 — [arXiv:2610.19975](https://arxiv.org/abs/2610.19975). 15 configurations spanning 1.3B–7B parameters; results on 9 downstream benchmarks.
