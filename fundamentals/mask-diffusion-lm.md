# Mask Diffusion Language Models

*Depth — a non-autoregressive LM family that generates by iteratively unmasking tokens in parallel.*

**TL;DR:** A **mask diffusion model (MDM)** is a language model trained to recover masked tokens from a partially masked sequence, applied as a *diffusion* process: start with a fully masked sequence, then iteratively unmask tokens over T denoising steps until a clean sequence remains. Unlike autoregressive (AR) LMs that generate left-to-right one token at a time, MDMs decode many tokens per step in parallel — and their core operator (re-masking + denoising) supports native *local* edits without regenerating the whole sequence.

**Prereqs:** [attention](attention.md), [_tokenization](_tokenization.md)
**Related:** [../post-training/reasoning/reflective-masking.md](../post-training/reasoning/reflective-masking.md), [../multimodal/diffusion-mllm.md](../multimodal/diffusion-mllm.md)

---

## What it is

A discrete-token diffusion process over sequences. The "forward" process progressively masks tokens; the model learns the "reverse" process that denoises a masked sequence back to a clean one.

Two ingredients:

- **A discrete corruption process.** At training time, each token in a sequence is randomly replaced with `[MASK]` according to a schedule (often a fraction `t` drawn uniformly from [0, 1]).
- **A denoising model.** A bidirectional transformer that, given the corrupted sequence, predicts the original tokens for the masked positions in parallel.

At inference time, start from an all-`[MASK]` sequence (or a partially-specified prompt) and iteratively predict + commit a subset of tokens per step, optionally re-masking low-confidence positions. After T steps, the full sequence is materialized.

## How it works

Training objective (simplified):

```
Sample sequence x₀ from data
Sample t ~ Uniform(0, 1)
Build x_t by masking each token of x₀ independently with prob t
Loss = -E[Σ log p_θ(x₀ⁱ | x_t) over masked positions i] (averaged over t)
```

Inference (one common schedule):

```
Start with x_T = all-mask
For step in T, T-1, ..., 1:
    p_θ(· | x_step) over all positions
    pick a subset of positions to commit (highest-confidence, or by schedule)
    keep the rest masked
    → x_{step-1}
Return x_0
```

Architecturally a standard transformer with bidirectional attention. The "diffusion" framing is what gives the principled training objective and the parallel decoding semantics.

## Why it matters

Three practical properties make MDMs interesting as an alternative to AR LMs:

- **Parallel decoding.** Many tokens per denoising step → can be much faster wall-clock than AR generation if the sampler is tuned.
- **Native local editing.** Unlike AR, where editing a token midway through requires regenerating everything to its right, MDMs naturally re-mask a small window and denoise locally — see [reflective-masking](../post-training/reasoning/reflective-masking.md) for how this is exploited for reasoning.
- **Symmetric handling of prefix and suffix.** A prompt can be tokens on either end of the sequence; MDMs don't have AR's strict left-to-right inductive bias.

The downside: MDMs have historically lagged AR LMs in quality at the same parameter count, especially on long-form reasoning. Recent work (2025–2026) closes much of the gap.

## Gotchas & tricks

- **Sampling schedule matters more than for AR.** Number of denoising steps T, the per-step commit policy (top-k by confidence vs. random subset), and re-masking decisions all materially shift quality.
- **Confidence calibration is the bottleneck.** If the model commits low-confidence tokens early, errors propagate. Most production samplers commit only high-confidence positions per step.
- **Pretraining is closer to BERT than to GPT.** The training objective is masked LM with a *random* mask ratio sampled per example, not a fixed 15% as in BERT. This is what makes the model usable as a generative denoiser, not just a fill-in-the-blank model.
- **Pairs with iterative refinement.** The "re-mask and denoise again" loop is a natural inference-time scaling knob — see [reflective-masking](../post-training/reasoning/reflective-masking.md).

## Sources

- Foundational: *D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces* — Austin et al., 2021.
- Foundational: *MaskGIT: Masked Generative Image Transformer* — Chang et al., 2022 — the discrete-mask diffusion sampler many MDMs adopt.
- *Multi-Turn Reflective Masking Elicits Reasoning in Mask Diffusion Models* — Zhang et al., 2026 — https://arxiv.org/abs/2606.16700 — recent post-training recipe that turns MDMs into reasoners.
- *PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models* — Sun et al., 2026 — https://arxiv.org/abs/2606.19534 — multimodal extension.
