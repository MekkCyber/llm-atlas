# MoE Vision Encoders
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Port fine-grained Mixture-of-Experts + aux-loss-free balancing (DeepSeek-style) into a **CLIP/SigLIP-shaped vision encoder** for VLMs. Fine-grained topologies (many small experts, high top-k) beat both dense and standard MoE, a specialised MoE kernel absorbs the added latency, and a frozen-image / trainable-temporal distillation pipeline adds video without degrading image priors. Matches a 1.7× larger dense encoder at 76% of its latency.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md), [../architectures/aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md)
**Related:** [../architectures/capacity-factor.md](../architectures/capacity-factor.md)

---

## What it is

Vision encoders for VLMs (CLIP, SigLIP, DINOv2) have been scaled *dense-first* — bigger ViT, more FLOPs per patch. MoE-ViE replaces the dense FFN in ViT blocks with a Mixture-of-Experts layer, following the same recipe MoE LLMs use, and demonstrates that vision-specific tuning of the topology (expert count, granularity, top-k) matters more than in LLMs.

## How it works

### Fine-grained MoE for patches

Instead of the usual 8–64 experts with top-2 routing, MoE-ViE uses a **fine-grained topology**: many small experts (>128) with a higher top-k (4–8). The intuition: image patches have more locally similar features than natural-language tokens, so a fine-grain expert can specialise on a narrower visual sub-distribution and be picked more often without over-concentrating routing mass.

### Aux-loss-free balancing (vision variant)

Adds a bias per expert to the router logits, adjusted online to keep load balanced — no auxiliary balancing loss on the training objective. The paper introduces a vision-encoder-specific variant that tolerates the patch-level distributional differences (background vs. object, uniform vs. texture-heavy patches).

### Latency: custom kernel

Naive MoE ViT is bandwidth-bound at inference — token-permute + expert-dispatch + un-permute reshuffle dominates. A specialised MoE kernel fuses these into a single pass that maintains sequence contiguity on the GPU, cutting the latency overhead that would otherwise erase the parameter-efficiency gain.

### Adding video without losing image knowledge

Two-stage: (1) freeze the image encoder trained above, (2) train temporal-mixing layers between frozen ViT blocks using **frame-level distillation** from the frozen encoder as a stability anchor. This preserves the image priors while adding video-specific structure — the encoder can be swapped between image-only and image+video inference without a quality drop.

## Why it matters

- **Ends the "just use SigLIP" default for VLMs.** MoE-ViE beats every compared dense encoder on both image and video benchmarks, even ones with 5× more activated parameters.
- **Sub-quadratic scaling regime.** Dense ViT-G at frontier scale is compute-prohibitive; MoE opens a curve where activation cost decouples from parameter count for vision, exactly as it did for LLMs.
- **VLM alignment gets easier.** A stronger, latency-competitive vision tower means fewer failure modes attributable to weak visual features — the LLM side does less work.

Reported: largest MoE-ViE matches a **SoTA dense encoder 1.7× its size at 76% of its latency**; when aligned with an LLM, it surpasses all compared encoders on image and video benchmarks. Code at `github.com/facebookresearch/moe_vie`.

## Gotchas & tricks

- **Granularity is a genuine hyperparameter.** Vision benefits from finer granularity than language; don't reuse LLM top-k=2 defaults.
- **Balance without an auxiliary loss.** Aux losses shift the vision loss landscape more than they help — the bias-adjustment scheme keeps balance without adding a competing objective.
- **Kernel overhead is real.** Without the specialised kernel, MoE-ViE loses its latency win — the token-permute step dominates. Ship the kernel with the weights.
- **Freeze then temporal-adapt.** Retraining the whole encoder on video regresses image benchmarks. Frozen-image + trainable-temporal beats joint training in this paper's ablations.
- **Batch effects.** MoE routing statistics need enough patches per batch to be reliable — vision batches are often smaller than LLM batches, so tune expert count with realistic batch size.

## Sources

- Paper: *MoE-ViE: Mixture of Experts Vision Encoder for Efficient Image and Video Understanding* — Zhang, Dong, Tran, Gschwind, Yang, Chen, Ahmadyan, Moon, Zhang, Kirmani, Damavandi, Kumar — Meta, 2026 — https://arxiv.org/abs/2608.17402
- Code: https://github.com/facebookresearch/moe_vie
- Related: SigLIP (Zhai et al., 2023), DeepSeek-MoE (Dai et al., 2024), aux-loss-free balancing (Wang et al., 2024).
