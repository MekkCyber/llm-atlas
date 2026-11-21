# HiUni
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A unified understanding-and-generation multimodal model built from a pretrained VLM that **keeps the same visual encoder and dense backbone for both tasks**, confining task-specific computation to the output side. Adds a **Pixel Diffusion Head** for generation plus a lightweight gating pathway (**HiGate**) that re-injects fine spatial detail lost through dense processing. Understanding scores stay close to the base VLM while generation and editing numbers remain competitive — evidence that a shared dense backbone can host both tasks without a routing / MoE split.

**Prereqs:** [README.md](README.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md), [posterior-matching-distillation.md](posterior-matching-distillation.md)

---

## What it is

Prior "unified multimodal models" (SEED, Show-o, Chameleon, Transfusion) either split the backbone into expert branches (one for understanding, one for generation) or interleave discrete image tokens with text tokens in a single shared vocabulary. HiUni keeps the dense backbone *unsplit* and makes generation an output-side head on top of it.

## How it works

Three structural pieces:

1. **Shared VLM backbone.** A pretrained vision-language model (encoder + dense decoder). Both understanding and generation read from the same residual stream.
2. **Pixel Diffusion Head.** The output side for generation: a diffusion head operating in pixel space, conditioned on the backbone's last-layer features. Generation is autoregressive at the backbone-token level but diffusion at the pixel level.
3. **HiGate pathway.** A thin gating pathway that re-injects high-resolution encoder features into the diffusion head at generation time. The dense backbone loses fine spatial detail under global pooling / long-range mixing; HiGate restores what's needed for crisp output without routing around the backbone.

Training:

- **Phase 1:** Freeze the backbone, train the diffusion head and HiGate on image-generation data.
- **Phase 2:** Unfreeze a late slice of the backbone and jointly fine-tune on a mix of understanding and generation data, with a balancing loss that prevents either task from dominating.

## Why it matters

- **One backbone, two heads.** Attractive for serving (one KV cache, one weight copy). Undercuts the "split into expert branches" argument that motivated Chameleon-Fuyu-style designs.
- **Understanding preservation.** Previous unified designs degraded understanding; HiUni reports scores near the frozen-VLM baseline.
- **Composable with modern diffusion.** The Pixel Diffusion Head can be swapped for any output-side diffusion family (flow-matching, posterior-matching distillation), so distillation techniques apply directly.

## Gotchas & tricks

- **HiGate bandwidth matters.** If HiGate is too weak, generated images are blurry; too strong and understanding regresses. The paper reports a schedule that ramps HiGate during Phase 2.
- **No MoE.** Route splitting is explicitly rejected — if your deployment target wants MoE sparsity, HiUni is the wrong starting point.
- **Pixel-space diffusion is slow.** Training is cheaper with latent diffusion heads; the paper argues pixel-space preserves fine-grained conditioning, but this is a cost-vs-quality call.
- **Author affiliation not disclosed on HF page.** Treat organisational provenance as unknown pending the arXiv listing.

## Sources

- Paper: *HiUni: Harmonizing Understanding and Generation in Shared Visual Space* — Zhenyu Zhang (project lead) et al., 2026 — primary source.
- Related: Chameleon (2024), Show-o (2024), Transfusion (2024) — competing unified-MLLM designs with different architecture choices.
