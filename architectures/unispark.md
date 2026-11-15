# UniSpark — Hybrid Attention for Unified Multimodal Modeling
*Depth — a Mixture-of-Transformer-Experts backbone mixing full and linear attention for one-model understanding + generation.*

**TL;DR:** A 9B unified multimodal model that uses a **Mixture-of-Transformer-Experts (MoT)** backbone combining **full softmax attention** with **GatedDeltaNet linear attention**. Text is produced autoregressively; images are produced by **flow matching**. A **blockwise training protocol** keeps noisy visual targets out of later autoregressive text predictions while allowing clean visual content to serve as context. A four-stage curriculum stretches image generation up to 2K resolution. Reports MME-S 2470.3 and GenEval 0.83, with instruction-based editing and interleaved text-image generation in one checkpoint.

**Prereqs:** [_moe](_moe.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [multi-head-attention](multi-head-attention.md)

---

## What it is

A unified multimodal backbone built around three pieces that normally sit in different models:

- **MoT (Mixture-of-Transformer experts)** — the routing happens at *whole transformer-block* granularity, with experts specializing by modality (text vs visual) rather than by token cluster.
- **Hybrid attention** — some experts use full softmax attention (expensive, high-fidelity); others use **GatedDeltaNet linear attention** (cheap, long-range). The combination gets softmax quality where it matters without softmax cost everywhere.
- **Flow-matching image head** — images are produced with a flow-matching objective, not autoregressive discrete tokens.

## How it works

- **Routing.** Token's modality (text or image/patch) selects an expert. Within modality, standard MoT routing applies.
- **Attention mix.** Full-attention experts are concentrated in blocks that matter most for high-fidelity reasoning / alignment; linear-attention experts carry the bulk of long-context work.
- **Blockwise training protocol.** During training, clean image blocks can appear *as context* for later text prediction, but noisy (in-progress / denoising) image blocks are masked from text targets. This keeps the AR text loss clean while still letting the model use visual context autoregressively.
- **Four-stage curriculum.** Progressive resolution up to 2K, interleaved with task mixes (understanding, generation, editing).

## Why it matters

- **Hybrid attention at scale.** Linear attention has been hard to integrate with high-quality generation; UniSpark shows softmax + linear-attention experts can co-exist and still produce 2K images and state-of-the-art VQA.
- **MoT routing for multimodality.** Whole-block modality experts is a cleaner factorization than token-level MoE routing for unified models — specialization is predictable and interpretable.
- **One checkpoint for understanding + generation + editing.** Removes the "serve two stacks" cost of running a separate VLM and a separate image generator.

## Gotchas & tricks

- Linear attention's throughput advantage is realized only with a fused kernel for GatedDeltaNet; naive implementations give back most of the win.
- Flow-matching image head's output quality is sensitive to the sampler schedule; the paper's numbers assume their specific schedule.
- "Unified" here covers understanding, T2I, editing, and interleaved generation; it does not include video, which remains a separate stack.

## Sources

- Paper: *UniSpark: Exploring Hybrid Attention for Multimodal Understanding and Generation* — Liu et al., 2026 — arXiv 2611.16517 — https://arxiv.org/abs/2611.16517
- GatedDeltaNet (linear attention building block, cited).
