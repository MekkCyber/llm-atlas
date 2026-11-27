# Looped Diffusion Transformer (DyLoMo-DiT)
*Depth — DiT variant that composes a weight-shared loop over middle layers with sparse experts, and schedules the loop count by diffusion noise level.*

**TL;DR:** DyLoMo-DiT applies two orthogonal sparsity axes to a diffusion transformer: **depth sparsity** (loop the middle layers to reuse parameters, as in [looped-transformer](looped-transformer.md)) and **width sparsity** (sparse MoE experts inside each loop). It then ties the number of loop passes to the current diffusion noise level — more passes when the denoising curve is steep (high-noise, early steps), fewer at the easy end. Reports lower ImageNet 256² FID than DiT and DiT-MoE at matched training compute, and higher GenEval than FLUX.1 at a smaller active-parameter budget.

**Prereqs:** [_moe](_moe.md), [looped-transformer](looped-transformer.md)
**Related:** [deepseek-moe](deepseek-moe.md) · [transformer-block](transformer-block.md)

---

## What it is

A diffusion transformer (DiT) in which the middle stack is a *looped* block with *sparse experts*. Standard DiTs scale by adding layers (depth) or widening FFNs; standard MoE-DiTs scale by adding experts (width). DyLoMo-DiT does both inside one looped stack, and additionally makes the loop count a function of the diffusion step.

## How it works

Three structural pieces:

1. **Weight-shared loop over middle layers.** The middle block set is applied $N_t$ times. Parameters stay small; effective depth grows.
2. **Sparse Mixture-of-Experts inside each looped layer.** Each token selects a few experts from a shared pool. Capacity scales via expert count without raising active parameters per token. Routing is standard top-k gating in the paper.
3. **Noise-conditional loop schedule $N_t$.** The diffusion scheduler emits the noise level $t$; a learned (or hand-set) schedule maps $t \mapsto N_t$. High-noise early steps get more loop iterations (denoising needs more computation when the distribution is far from the data manifold); low-noise late steps get fewer.

The per-step compute is therefore not uniform across the generation trajectory — it is skewed toward the early steps, matching where the denoising objective's curvature is highest.

## Why it matters

- **Transfers LLM-era sparsity ideas into DiTs.** Weight-shared loops + token-routed experts had already paid off in LLMs; DyLoMo-DiT ports them to image/video generation and reports FID / GenEval wins at matched budget.
- **Noise-conditional compute is diffusion-native.** Unlike language modeling (where every token gets the same compute), diffusion has a natural per-step budget axis. DyLoMo-DiT is one of the first DiTs to exploit it explicitly through loop count.
- **Beats DiT-MoE at matched active parameters.** The paper reports the text-to-image variant outscoring FLUX.1 on GenEval with fewer active parameters, suggesting looping adds genuine capacity on top of MoE's width.

## Gotchas & tricks

- **Schedule tuning matters.** A flat $N_t$ (same loop count at every step) recovers plain looped-MoE-DiT and is a strict loss vs. the noise-conditional schedule in the paper. If porting the idea, re-tune $N_t$ for your sampler (DDPM vs. DDIM vs. flow-matching schedules have different curvature profiles).
- **Expert routing + looping share instability failure modes.** Both are known to suffer from training collapse without balance losses. Use standard MoE load-balance losses ([../architectures/load-balancing-loss.md](load-balancing-loss.md), [../architectures/aux-loss-free-balancing.md](aux-loss-free-balancing.md)) and watch for routing collapse early in training.
- **Inference cost moves with $N_t$.** The KV-sharing trick that keeps looped LLMs cheap at inference is less useful for diffusion (each step is a full forward pass anyway). Instead, a lower $N_t$ at late steps gives the practical sampling speed-up.

## Sources

- Paper: *DyLoMo-DiT: Dynamic Looped Mixture-of-Experts Diffusion Transformers* — Li, Liu, Xiong, Zhang, 2026, arXiv 2611.30844.
- Related: *DiT-MoE* — Fei et al., 2024 — sparse-expert diffusion transformer baseline.
- Related: *Mixture-of-Loops Transformer* — 2026, arXiv 2611.34923 — looped-with-KV-sharing on the LLM side; see [looped-transformer](looped-transformer.md).
