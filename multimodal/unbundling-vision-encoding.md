# Unbundling Multimodal Vision Encoding
*Depth — a controlled study of where and how VLM vision capacity should sit.*

**TL;DR:** A compute-matched ablation study of three conventions that VLM designers usually bundle: **(1) separate encoder pretraining**, **(2) dedicated vision parameters**, and **(3) placing them *before* the language backbone**. Training every variant from scratch on the same data at matched compute, the paper finds: (1) is unnecessary — joint training matches pre-encoder pretraining at matched compute; (2) is necessary — encoder-free VLMs still underperform on VQA even when the backbone is enlarged to match parameter counts; (3) is **not** necessary — vision-only modules embedded in the *early layers of the backbone* perform comparably to a front-loaded encoder and reduce time to first token.

**Prereqs:** [../architectures/transformer-block](../architectures/transformer-block.md)
**Related:** [../fundamentals/attention](../fundamentals/attention.md)

---

## What it is

A design-space ablation, not a new architecture. Decouples three choices VLM papers usually make together and measures each at matched compute:

- **Pretraining pathway** — is the vision encoder pretrained separately on CLIP-style objectives, or trained from scratch alongside the LM?
- **Capacity allocation** — does the model have dedicated vision parameters, or does the LM backbone shoulder all visual processing (encoder-free)?
- **Placement** — do dedicated vision parameters come *before* the LM backbone (classic CLIP → projector → LM), or can they live *inside* the backbone's early layers?

## How it works

- **Protocol.** Train every variant from scratch on the same text+image+interleaved data at matched *total training compute*. Each ablation changes one axis while holding the others fixed.
- **Metrics.** Downstream VQA accuracy, latency to first text token, scaling behavior across model sizes.
- **Starting points.** Both random-init and pretrained-LM starting points are tested; conclusions hold across both.

### Findings

1. **Separate encoder pretraining isn't needed at matched compute.** A jointly trained encoder performs about as well. Encoder-first pretraining only "wins" when its pretraining compute is free-ridden on top.
2. **Dedicated vision parameters do help.** Encoder-free VLMs underperform on VQA, and enlarging the backbone to match parameter counts doesn't close the gap — the capacity needs to be specialized for vision.
3. **Placement is flexible.** Early-backbone-layer vision modules match pre-encoder designs on VQA and reduce time to first token (because the encoder's serial step is folded into the backbone's parallel prefill).

## Why it matters

- **Challenges the CLIP-era default.** Most production VLMs still run a separate encoder stage. If early-backbone vision modules match performance with better latency, the industry default should shift.
- **Rebuttal to "encoder-free" purists.** The paper shows encoder-free is attractive but trades accuracy. Dedicated vision capacity matters; where it sits does not.
- **Clean compute-matched methodology.** Many VLM comparison papers mix uncounted pretraining compute; the matched-compute protocol here is what gives the conclusions their weight.

## Gotchas & tricks

- "Matched compute" accounting is sensitive to how encoder pretraining time is counted; the paper is explicit about including vs excluding it and shows both readings.
- Results cover standard VQA-style evaluation. Fine-grained localization, OCR-heavy tasks, and video have not been ablated with the same protocol.
- "Early-backbone vision module" is still a *module*, not a pure dense layer — it's typed to vision and sees image patches only. The paper is not claiming the backbone's generic layers suffice.

## Sources

- Paper: *Unbundling Multimodal Vision Encoding* — Hoang et al., NYU / Meta FAIR, 2026 — arXiv 2611.17334 — https://arxiv.org/abs/2611.17334
