# Group-Scaled Autoregressive Image Generation
*Depth — scaling AR group size quadratically with per-axis spatial compression recovers the AR speed gap for discrete image generation.*

**TL;DR:** Discrete autoregressive image generation faces a hard tradeoff: long token sequences preserve quality but are slow; aggressive spatial compression is fast but kills representation capacity. **G-STAR** (Group-Scaled Token cApacity for AR) argues and shows empirically that, under a fixed information budget, the AR *group size* should scale **quadratically** with per-axis spatial compression. With 32× (effective 64×) compression and a semi-autoregressive within-group decoder, this delivers 1024×1024 images in **2.5 s on one H200** — a 20× speedup over prior AR models at competitive quality.

**Prereqs:** [../pre-training/mtp.md](../pre-training/mtp.md), [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [../fundamentals/bpe.md](../fundamentals/bpe.md)

---

## What it is

A design rule for discrete-AR image generators:

- Choose per-axis spatial compression $c$ (e.g., 32×: one token represents a 32-pixel-wide patch per axis).
- Choose AR group size $g$ (how many image tokens are predicted per AR step).
- **Rule: $g \propto c^2$.** Because spatial compression is 2D, compressing $c$× per axis multiplies tokens per patch by $c^2$. Group size has to track that, or the sequence stops carrying enough information to reconstruct the scene.

G-STAR pairs this with a non-autoregressive *within-group decoder* that handles the intra-group dependencies, so each AR step emits a whole group in parallel.

## How it works

### Sequence structure

Image is tokenized at compression ratio $c$ per axis, giving an $(H/c) \times (W/c)$ grid of discrete tokens. Flatten and group into chunks of $g = c^2$ tokens. The AR model generates one *group* per step:

$$
p(x) = \prod_{k=1}^{K} p(x_{g \cdot k - g + 1 \,:\, g \cdot k} \mid x_{<\,g \cdot k - g + 1})
$$

where $K = HW / (g \cdot c^2)$ groups total.

### Within-group semi-autoregressive decoder

Within each group, G-STAR does *not* use another AR pass (that would cancel the speedup). Instead it uses a non-AR / bidirectional decoder — essentially a short MaskGIT-like iterative refinement conditioned on the group's AR context — to model the intra-group joint distribution.

Why $g = c^2$ specifically: a patch of linear compression $c$ contains $c^2$ pixels. Each token must retain enough channel capacity to encode its pixel-share. Keeping the aggregate information capacity constant across compression ratios means scaling group size with pixel count per patch.

## Why it matters

- Discrete AR image generators had lost the speed race to diffusion by a wide margin — multi-second per image at 1024px, not 2.5 s. G-STAR closes most of that gap while keeping the AR formulation.
- Matters for unified multimodal models (text + image AR in one backbone) that want to generate images without switching to a diffusion head: now you can keep AR and still generate 1024px in a few seconds.
- The $g \propto c^2$ rule is a clean scaling law — rare in the visual-generation space, where speedups are usually heuristic.

## Gotchas & tricks

- Rule assumes uniform 2D spatial compression. For aspect-ratio-irregular or non-grid tokenizations, "compression per axis" needs re-defining.
- The within-group decoder's iterative refinement is the quality bottleneck. More iterations → better fidelity → smaller practical speedup.
- Group boundaries show up as artifacts if group ordering is naïve raster scan. G-STAR uses a learned group traversal to minimize seams.
- Benchmark reporting: "speedup vs. prior AR" is against models with $g = 1$. Against modern diffusion, the comparison is more nuanced and task-dependent.

## Sources

- Paper: *G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation* — Huang et al., 2026 — https://arxiv.org/abs/2611.13004
- Related: *MaskGIT* — 2022 — within-group iterative refinement ancestor.
