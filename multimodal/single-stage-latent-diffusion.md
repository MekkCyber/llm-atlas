# Single-Stage Latent Diffusion (UNIFIED)
*Depth — one diffusion transformer with fully shared weights that encodes, generates, and decodes, distinguished only by a mode token and the timestep.*

**TL;DR:** Latent diffusion ships three networks (encoder, generator, decoder). UNITE merged encoder+generator. UNIFIED (2026) folds the decoder in too. The key move: cast all three modes as **position-aligned sequence maps** over one shared backbone, and signal the mode with a learnable mode token plus the timestep. Trained from scratch in a single stage with only reconstruction and flow-matching losses — no adversarial loss, no DINO. Beats UNITE at matched params and reaches 2.08 gFID at 215M params on ImageNet-1K 256×256.

**Prereqs:** [README](README.md), [../architectures/transformer-block](../architectures/transformer-block.md)
**Related:** —

---

## What it is

A unification of the three networks that latent generative models carry (encoder: pixels → latent; generator: latent distribution; decoder: latent → pixels) into a **single diffusion transformer** with fully shared weights. Introduced as UNIFIED (Hsu et al., 2026). All three roles share every parameter; only a mode token and the current diffusion timestep differ.

## How it works

Core design choices:

1. **Position-aligned sequence maps.** Each of encoding, generation, and decoding is reformulated as a function $f: \text{sequence} \to \text{sequence}$ over the same positions. This lets one transformer stack handle all three without role-specific layers or heads.
2. **Mode token + timestep as mode selector.** A learnable embedding (the mode token) is prepended to the sequence; the diffusion timestep is injected as usual. The combination tells the backbone which role to play on this forward pass.
3. **Single-stage training.** From scratch, with reconstruction loss (for encoding/decoding modes) and flow-matching loss (for generation mode). No adversarial loss, no pretrained perceptual encoder (DINO/VGG). All three modes share the gradient.

Notable negative result: naively letting the shared backbone decode without the position-aligned-sequence framing doesn't work. The sequence-map framing is load-bearing.

## Why it matters

Three structural simplifications:

- **No VAE.** The encoder is the same transformer.
- **No adversarial loss.** Training stability improves and perceptual-loss distillation disappears.
- **One training stage.** Throughput and infrastructure complexity drop.

Reported quality: at matched parameters (~60M, ~115M, ~215M), UNIFIED improves gFID over UNITE on ImageNet-1K 256×256 and reaches 2.08 at 215M — the lowest among compared models under 250M params. On text-to-image at 512×512, UNIFIED-B (218M) hits GenEval 0.87, exceeding single-stage pixel diffusion models with roughly one billion parameters.

## Gotchas & tricks

- **Mode token design matters.** A single-token mode is cleaner than a set of role-specific prefixes; the paper's ablations show position-aligned framing + mode token is a combined effect.
- **Resolution transfer is untested beyond 512×512.** Scaling to video or high-res image generation is an open question.
- **Shared weights may limit per-role capacity.** The paper observes that UNIFIED's gains persist across sizes, but it's plausible that a decoder-specialized variant could still beat UNIFIED at very large scales.
- **Composability.** The mode-token design is architecturally orthogonal to flow matching vs. score matching; the paper uses flow matching but the recipe should carry over.

## Sources

- Paper: Single-Transformer Latent Diffusion — Hsu, Wang, Schwing et al., 2026 — [arXiv:2611.29064](https://arxiv.org/abs/2611.29064)
- Project page: https://haoyuhsu.github.io/Unified-LDM/
