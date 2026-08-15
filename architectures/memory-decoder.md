# Memory Decoder
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A small, separately-trainable decoder that runs alongside a **frozen** large backbone to specialize it to new domains — a "memory branch" rather than a fine-tune. In Intern-S2-Preview, a **4B Memory Decoder lifts the frozen 397B backbone's Biology-Instructions average from 56.92 → 60.32** without modifying a single backbone weight.

**Prereqs:** [transformer-block](transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [../post-training/fine-tuning/README](../post-training/fine-tuning/README.md), [../post-training/_post-training](../post-training/_post-training.md)

---

## What it is

A specialization path for very large frozen backbones. Standard options are:

- **Full fine-tuning** — updates the backbone, prohibitively expensive at 397B and destroys generality.
- **PEFT (LoRA / adapters)** — cheap, but the update path is bolted onto the backbone's forward pass and inherits its structural constraints.
- **RAG** — external retrieval, but under-uses the model's parametric knowledge for the new domain.

Memory Decoder is a **fourth option**: an independently-trained small decoder whose outputs are fused with the frozen backbone's decoding. The "memory" holds domain knowledge; the backbone contributes reasoning and general competence. Both models are trained separately, and inference composes them.

## How it works

**Setup.** Given a frozen backbone $B$ (e.g., 397B) and a target domain corpus $D$:

1. **Train** a small memory decoder $M$ (e.g., 4B) on $D$. $M$ is a standard transformer decoder, trained on next-token prediction over $D$ from scratch or from a small pretrained init.
2. **Compose at inference.** For each output token, fuse $M$'s distribution with $B$'s distribution. Fusion can be:
   - **Logit-space interpolation** — $\log p = (1 - \alpha) \log p_B + \alpha \log p_M$, where $\alpha$ may be static, learned, or gated per-token.
   - **Hidden-state fusion** — $M$ contributes activations that are injected into $B$'s residual stream at chosen layers.
   - **Attention-key contribution** — $M$'s hidden states are added to $B$'s KV cache for cross-attention.

The Intern-S2-Preview paper reports the outcome (Biology-Instructions 56.92 → 60.32 with Intern-MemDec-4B against a frozen 397B) but does not disclose the exact fusion mechanism in the preview.

**Key property.** $B$ is never touched. Multiple memory decoders can be swapped in per domain — biology memory, chemistry memory, materials memory — each trained independently at 4B scale.

## Why it matters

- **Deploy-once, specialize-many.** One 397B backbone in memory, many small memory decoders on disk. Load the relevant one per task.
- **Training cost lives at 4B, not 397B.** For a lab that already ships a frontier general model, adding a new scientific vertical costs a 4B training run, not a 397B fine-tune.
- **Preserves general capability.** Frozen backbone means no catastrophic forgetting on general benchmarks — you can measure and guarantee the general behavior is unchanged.
- **Composable with fine-tuning of the small decoder.** If you already have a small SFT'd model, it can act as the memory decoder. Path from existing artifacts to memory-decoder deployment is short.
- Middle ground between RAG (retrieves text) and PEFT (updates the base): Memory Decoder is **parametric** but **external**.

## Gotchas & tricks

- **Fusion mechanism matters more than size.** Logit interpolation is the simplest but weakest — the memory can only correct final-layer decisions. Hidden-state fusion is stronger but requires backbone-internal access.
- **α calibration.** Static α is a compromise: too high, the small model dominates and quality drops on general tasks; too low, the memory contributes nothing on specialized tasks. Learned or gated α is better.
- **Latency.** Two forward passes per token unless the fusion is co-scheduled. On a shared GPU, memory decoder inference is small ($\sim$1% of backbone cost at 4B/397B), but the ergonomics of running two models need serving-stack support.
- **Not a general RAG substitute.** For factual queries about rapidly-changing information, retrieval still wins. Memory Decoder shines when the domain is stable and deep enough to justify parametric specialization.
- **Composability across memories is not automatic.** Running biology memory + chemistry memory simultaneously requires a fusion rule for multiple auxiliary decoders, which the current formulation doesn't address.
- **Interacts with quantization.** A quantized frozen backbone + full-precision memory decoder is the natural deployment shape; ensure logit ranges align.

## Sources

- Paper: *Intern-S2-Preview: Scientific Agentic Foundation Model* — Shanghai AI Laboratory, 2026 — [arXiv:2608.13505](https://arxiv.org/abs/2608.13505) — names Memory Decoder as a specialization path; reports Intern-MemDec-4B + frozen 397B on Biology-Instructions.
