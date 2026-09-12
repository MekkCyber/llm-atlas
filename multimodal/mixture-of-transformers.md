# Mixture-of-Transformers (MoT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A multimodal architecture that dedicates a separate transformer expert per modality — text, image (patch), image-generation — while letting all experts see one another through a **shared global self-attention**. Reads like an MoE where routing is by *modality* rather than by learned gate, and where each expert is a full transformer rather than an FFN block. Practical fit for **native unified models** (encoder-free / VAE-free): patch tokens go through the image expert, text tokens through the text expert, and generation tokens through the generation expert, but all share the same attention context.

**Prereqs:** [../architectures/_moe](../architectures/_moe.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [README](README.md), [native-unified-models](native-unified-models.md)

---

## What it is

Unified multimodal models want to *understand* an image, *reason* about it in text, and *generate* another image within one autoregressive stack. Prior designs either (a) glued three specialist models together with adapters, (b) put every modality through a single shared transformer (losing modality-specific inductive bias), or (c) shared attention but split FFN blocks per modality — the last of which is what MoT crystallizes into a clean design.

Structurally: at every layer, tokens are routed by their **modality tag** — not by a learned gate — into a per-modality set of parameters (attention Q/K/V projections and FFN). The global self-attention then mixes tokens across modalities within the same layer. Each modality has its own weights, but they attend jointly.

## How it works

Per transformer layer:
```
tokens (mixed modality) ─┬─► if modality = text  → Q_text, K_text, V_text, FFN_text
                         ├─► if modality = image → Q_img,  K_img,  V_img,  FFN_img
                         └─► if modality = gen   → Q_gen,  K_gen,  V_gen,  FFN_gen

                         → concat Q,K,V → shared multi-head self-attention → per-modality out-proj → per-modality FFN
```

Key properties:
- **Routing is by modality tag**, not by a learned gate — no load-balancing loss needed.
- **Attention is joint.** All tokens land in one attention block; each token's Q/K/V just come from its own modality's projections.
- **Total parameters scale as $M \times P_{\text{expert}}$** for $M$ modalities, but any single forward pass activates $\sim P_{\text{expert}}$ compute — sparse in the MoE sense.
- **Position/space handling is per-modality.** Text uses 1D positions; image patches use 2D positional encodings; generation-side positions can be different again.

## Why it matters

- **Preserves modality-specific inductive bias.** Text and image want different attention patterns and normalization scales; MoT gives each modality its own without giving up cross-modality attention.
- **Cheap capacity growth.** Adding a new modality = adding another expert. Existing modalities are untouched.
- **Enables native unified models.** SenseNova-U1.5 uses MoT to run understanding + generation + editing in one encoder-free / VAE-free stack. Without per-modality expert weights, the generation-side signal would corrupt the understanding representation.
- **Sparsity story.** Compute per token is bounded by a single expert; parameter budget spreads across all experts.

## Gotchas & tricks

- **Modality-tag correctness.** Wrong tags silently route tokens through the wrong FFN and only surface as generation glitches — assert-check the tags near the tokenizer boundary.
- **Attention re-scaling.** Different modalities have different attention entropy; add per-modality attention temperature or per-modality QK-Norm if scale mismatches show up in early training.
- **Load imbalance at fixed cost.** MoT doesn't need a routing loss, but modality *proportions* in the batch determine which expert gets more gradient; sample the mixture explicitly.
- **Cross-modality initialization.** Init experts by copying a strong text-transformer's weights, then perturb slightly for other modalities — cold init on the image expert underperforms.
- **Not a substitute for shared MoE.** MoT and gated MoE can coexist: MoT for modality specialization, gated MoE inside each expert for task-level sparsity.

## Sources

- Paper: *SenseNova-U1.5: Towards Native Unified Visual Intelligence* — SenseTime, 2026 — [arXiv:2609.11929](https://arxiv.org/abs/2609.11929) — the 8B-MoT native unified model demonstrating the design at scale.
- Related: earlier "Mixture-of-Transformers" line of work in multimodal pretraining exploring per-modality parameter blocks with shared attention.
