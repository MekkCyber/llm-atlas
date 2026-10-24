# GAFU (Generation-Aware Fine-grained Understanding)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In unified multimodal models (one network for both image understanding and image generation), the generation branch already carries rich fine-grained visual features — but the understanding branch doesn't use them. GAFU is a post-training step that compresses generation-side representations into a small set of **G tokens** and feeds them to the understanding path. Beats SFT on the same data across benchmarks and across unified-model architectures (2026).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [README.md](README.md)

---

## What it is

A representational-asymmetry fix for unified multimodal models. The generation branch is already strong at pixels and shapes (it has to be, to generate); the understanding branch, trained with different losses, is weaker at fine-grained visual detail. GAFU projects a few generation-side tokens into the understanding branch's attention, closing the gap without new data.

## How it works

- **G tokens.** A small learned bottleneck (a cross-attention module from generation-side features into a fixed-size token bank) produces N compressed G tokens per image.
- **Insertion.** G tokens are prepended to (or cross-attended into) the understanding branch's token stream. Only the G-token projection and the understanding branch's cross-attention layers are updated; both branches' main weights stay fixed.
- **Objective.** Standard understanding-side losses (VQA, captioning, grounding) on existing SFT data. No generation-side labels are added; the signal is purely the already-learned generation features.
- **Compactness.** N is small (the paper calls the tokens "compact"), making this near-free at inference versus feeding full generation-side features.

## Why it matters

- **Free capability from existing representations.** Unified models already paid the training bill for generation-side features; GAFU recoups that investment for understanding.
- **Architecture-agnostic.** Works across tested unified architectures — the mechanism is "project generation features into understanding attention," not a specific module.
- **Minimal parameters.** The only new params are the G-token projection and the cross-attention adapter; most of the model stays frozen.

## Gotchas & tricks

- **Needs a unified model.** GAFU is useless on separate understanding-only VLMs — there is no generation branch to extract from. For separate models, the equivalent step is distillation from a generation model, which is more expensive.
- **N matters.** Too few G tokens underfit; too many drown the understanding signal. The paper reports best N as a tuned hyperparameter.
- **Can't be applied to a locked unified model.** Even though main weights stay fixed, the generation-side features must be *accessible* — serving stacks that only expose understanding outputs can't do this.
- **Not a free lunch for alignment.** G tokens pass generation-side quirks (hallucinated textures, mode collapse) through to understanding; it's worth measuring whether known generation failure modes show up as understanding failures.

## Sources

- Paper: *What Generation Knows: Unlocking Fine-Grained Understanding in Unified Multimodal Models* — 2026 — [arXiv:2610.23987](https://arxiv.org/abs/2610.23987).
