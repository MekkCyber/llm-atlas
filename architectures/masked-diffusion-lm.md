# Masked Diffusion Language Model (dLLM)
*Depth — non-autoregressive LMs that decode text by iteratively denoising masked tokens with bidirectional attention.*

**TL;DR:** A masked diffusion language model (dLLM) generates text by starting from a fully masked sequence and iteratively predicting the tokens beneath the masks, using a transformer with **bidirectional attention** over the whole (still-masked) window. Unlike AR LMs, prediction is parallel across positions and refined over many denoising steps; multiple in-flight tokens can be revised. Modern instances include LLaDA and Dream. dLLMs are competitive with AR LMs at similar scale on many benchmarks and have distinct strengths (parallel decoding, editing) and weaknesses (bounded reasoning across chunks, more expensive per token).

**Prereqs:** [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md)
**Related:** [register-tokens](register-tokens.md), [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

A generative LM whose forward process progressively **masks** tokens and whose reverse process is a bidirectional transformer trained to predict all masked positions from all unmasked ones. At inference, the model runs several denoising steps: each step commits (or resamples) a subset of positions based on model confidence, working the sequence from all-masked to fully written. It is the discrete-text analogue of masked-image diffusion, not of continuous-latent DiTs.

## How it works

1. **Training.** Sample a corrupted sequence by masking a random fraction of tokens; the transformer outputs a categorical distribution over the vocabulary at every masked position, and the loss is (weighted) cross-entropy over those positions.
2. **Inference.** Start from an all-mask sequence of the target length. Repeat for T denoising steps:
   - Forward pass; get per-position logits.
   - Pick a subset of masked positions to commit (usually the top-k by confidence, or scheduled per-position).
   - Optionally *re-mask* previously committed positions with low current confidence (`remask` schedules).
3. **Chunking.** Long generations proceed by chunks: fully denoise a chunk of length L, append it to the prompt, then start a new chunk. This is where dLLMs' long-horizon reasoning becomes fragile — see gotchas.
4. **Attention.** Fully bidirectional within the chunk, causal on prior chunks.

## Why it matters

- **Parallel decoding** — many tokens per denoising step, not one per forward pass.
- **Editing.** Because attention is bidirectional and any position can be re-masked, dLLMs naturally support fill-in-the-middle and revision.
- **Different failure profile.** Errors that AR LMs make left-to-right, dLLMs sometimes catch by re-masking; conversely, dLLMs make consistency errors AR LMs don't.

## Gotchas & tricks

- **Chunk boundary is where dLLMs lose long-horizon reasoning.** Each new chunk sees only prompt + prior chunks as text; internal working state does not survive. Solutions include register tokens ([register-tokens](register-tokens.md)) that carry hidden state across chunks.
- **Number of denoising steps trades quality for cost.** Fewer steps → faster, lower quality; more steps → AR-like cost.
- **Remask schedules matter.** Naive top-k confidence commits can lock in early errors; scheduled remasking (letting the model re-open a low-confidence position later) is a common trick.
- **Post-training is different.** SFT and RL both work but need masking-aware objectives; DPO on dLLMs is an active area.
- **Length is fixed per chunk.** dLLMs don't naturally do variable-length generation; length policies are picked at chunk boundaries.

## Sources

- Paper (LLaDA): *Large Language Diffusion Models* — Nie et al., 2024/2025.
- Paper (Dream): *Dream 7B* / *Dream-Coder* — release notes 2025–2026.
- Paper: *Register Tokens for Bounded-State Reasoning in Diffusion Language Models* — Ge et al., 2026 — [arXiv:2609.16372](https://arxiv.org/abs/2609.16372).
