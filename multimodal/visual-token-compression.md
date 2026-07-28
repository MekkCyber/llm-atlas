# Visual Token Compression (VisCo)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Compress the visual-token stream of a VLM by **reusing the VLM itself** as a parameter-sharing autoencoder. A small pool of learnable **memory tokens** replaces most raw visual tokens; the pretrained backbone acts as both encoder and decoder. Training-efficient (no external compressor to train from scratch), beats prior training-based and training-free baselines at all compression ratios, stays coherent at the single-token-per-image extreme, and *concatenating* memory tokens with the original visual tokens **improves** the base model.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** none yet

---

## What it is

A visual-token-compression scheme in the class of **intrinsic-encoder** methods — the compressor is the VLM itself, not a bolt-on module. Introduces a compact set of *memory tokens* that carry the image content forward; hierarchical information flows between encoding and decoding phases via parameter sharing.

## How it works

- **Memory tokens.** A small learned pool of query tokens (few → single-digit → single) sits inside the visual-token stream.
- **Parameter-sharing autoencoder.** The pretrained VLM backbone plays both encoder (compresses raw visual tokens into memory tokens) and decoder (reads memory tokens to answer downstream queries). Weights are shared; no new heavy module.
- **Training-efficient loss.** Because the backbone is reused, adaptation is limited to the memory-token embeddings and (optionally) a light adapter, not a full retraining pass.
- **Complementarity check.** Concatenating memory tokens with the *original* visual tokens is evaluated separately; the combined variant outperforms the base VLM — evidence that memory tokens capture information the raw tokens don't surface.

## Why it matters

Visual tokens dominate VLM prefill cost and KV footprint. Prior training-free compressors degrade at aggressive ratios; prior training-based compressors force the backbone to adapt to a foreign module and burn substantial compute. VisCo is training-efficient, monotonic in quality vs. ratio, and *strictly upgrades* the base model when kept alongside the original tokens — a rare all-upside change that lands well as VLMs move to long-video and document-heavy workloads.

## Gotchas & tricks

- **Single-token extreme is impressive but domain-dependent.** VLM tasks that need fine-grained localization degrade more than caption-style tasks; check per-task before deploying at max compression.
- **The parameter-sharing property is what makes it cheap.** Bolting on a separate autoencoder gives up the training-efficiency claim.
- **Concat-with-original mode is a distinct deployment.** It costs the raw-token budget *plus* the memory tokens — use it as a quality-first deployment, not the compression deployment.

## Sources

- Paper: *VisCo: Leveraging Large Language Models as Intrinsic Encoders for Visual Token Compression* — Zheng, Zou, Liu, Yu, 2026 — [arXiv:2607.12756](https://arxiv.org/abs/2607.12756).
