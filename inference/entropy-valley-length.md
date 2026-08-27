# Entropy-Valley (EV) Length Selection
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Masked diffusion language models decode on a **fixed canvas** — target length must be chosen *before* denoising begins. Prior work focused on *unmasking order* and left length selection under-explored despite its direct impact on coverage and redundancy. **Entropy-Valley (EV)** is training-free: run an all-mask forward pass at each candidate target length, score each canvas by mean predictive entropy, and pick the one the backbone is most confident to fill. Introduced by Zhan et al. 2026 for masked-diffusion machine translation.

**Prereqs:** None (masked-diffusion basics)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Autoregressive LMs decide length by generating an end-of-sequence token — length emerges from the generation itself. Masked-diffusion LMs are different: they operate on a **fixed canvas** of masked positions and iteratively unmask them. Before the first denoising step, you must commit to a canvas size.

Standard practice is to pick length from training-corpus statistics (mean length or a length-prediction head trained separately). This is a coarse baseline — it ignores that the *specific input* might be well- or poorly-suited to a given canvas length.

EV asks: what if we let the backbone itself vote on which canvas it wants?

## How it works

### The all-mask entropy signal

For each candidate target length $L \in \{L_1, L_2, \ldots\}$:

1. Construct an all-mask canvas of length $L$ concatenated with the (fully visible) source.
2. Run a single forward pass of the masked-diffusion backbone.
3. Compute per-position predictive entropy $H(p(x_i \mid \text{source}, \text{all-mask}))$.
4. Score $S(L) = \text{mean}_i H(x_i)$ — mean predictive entropy across all positions.

Pick $L^* = \arg\min_L S(L)$: the canvas the backbone is most **prepared** to fill.

### Why entropy is the right signal

If the backbone is very uncertain across all positions of a candidate canvas, it doesn't have a strong prior over what should fill it — the length is a bad match to what the source affords. If entropy collapses into a "valley" at some $L$, the backbone has strong beliefs about what should live there. That valley is the length the model wants.

Training-free: no length head, no additional fine-tuning.

## Why it matters

- **Recovers 33–65% of oracle-length gain** on COMET-22 across En↔Zh and En→De MT — a large fraction of the ceiling from *knowing* the reference length, achieved with only forward passes.
- **Establishes what matters most in masked-diffusion decoding.** The paper's oracle-length diagnostic reveals that *how much target to generate* matters more than *which tokens to reveal first* — inverting the prior conventional wisdom that unmasking order was the key lever.
- **General decoding-time trick.** All-mask entropy is a cheap way to score any decoding decision that fixes a discrete choice ahead of denoising (length, structural template, position budget). Broadly reusable.

## Gotchas & tricks

- **Candidate length grid choice.** Too coarse a grid (e.g. only 5 lengths) misses valley minima; too fine wastes forward passes. A geometric grid around a corpus-based prior works well in practice.
- **Compute cost.** One extra forward pass per candidate length — cheap compared to N-step diffusion decoding, but not free.
- **Entropy is not calibrated across lengths uniformly.** Very short canvases naturally have lower total entropy just because there are fewer positions. Use *mean* entropy per position, and consider adding a small prior over reasonable length ranges.
- **Applies only to fixed-canvas methods.** Continuous-diffusion LMs or those with dynamic length don't benefit — the length decision they make is different.
- **Reference length is not always the best length.** Zhan et al. observe denoising-friendly lengths sometimes beat reference lengths in COMET-22. Don't treat "match reference" as the ceiling — EV can beat it in specific cases.

## Sources

- Paper: *Length-Adaptive Decoding for Masked Diffusion Machine Translation* — Zhan, Hou, Zhang, Gao, 2026. [arXiv:2608.22274](https://arxiv.org/abs/2608.22274).
- Related: prior masked-diffusion decoding work (unmasking-order focus).
