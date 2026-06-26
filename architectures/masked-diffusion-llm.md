# Masked-Diffusion Language Model
*Depth — a non-autoregressive LLM trained with a masked-diffusion denoising objective and fully bidirectional attention.*

**TL;DR:** Replace next-token prediction with a *masked-diffusion* objective: at each training step, mask a random fraction of tokens and train the model to predict all of them in parallel given full bidirectional context. Inference iteratively unmasks tokens. With enough scale (8B+, 12T+ tokens) this matches autoregressive baselines on most benchmarks while supporting parallel decoding, in-place edits, and length-flexible generation. iLLaDA (Min et al., 2026) is the strongest published evidence at 8B.

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_normalization](_normalization.md), [_lr-schedules](../pre-training/_lr-schedules.md), [_training-stability](../pre-training/_training-stability.md), [illada case study](../case-studies/illada.md)

---

## What it is

A Transformer LM whose training objective is **masked-token denoising at a continuum of noise levels**, instead of causal next-token prediction. Two distinguishing features:

1. **Fully bidirectional attention** — no causal mask. Every position attends to every other.
2. **Variable-rate masking** — the masking rate $t \in (0, 1]$ is sampled per batch (uniform or scheduled), and the model must reconstruct the masked tokens at *any* corruption level.

At inference, generation is iterative *unmasking*: start with a fully (or partially) masked sequence and progressively reveal tokens over $T$ refinement steps.

## How it works

### Training objective

Let $x = (x_1, \ldots, x_L)$ be a token sequence. Sample a masking rate $t \sim U(0, 1]$. Construct $x_t$ by independently replacing each $x_i$ with `[MASK]` with probability $t$. Train the model $\pi_\theta$ to minimize

$$
L(\theta) = \mathbb{E}_{t, x, x_t}\!\left[ \frac{1}{t} \sum_{i: x_{t,i} = \mathrm{MASK}} -\log \pi_\theta(x_i \mid x_t) \right]
$$

The $1/t$ weighting is the masked-diffusion ELBO correction — it accounts for the fact that high-$t$ batches contain more mask positions and would otherwise dominate the loss. The $\pi_\theta(x_i \mid x_t)$ is read from the model's output at position $i$ — the marginal over the vocabulary given the corrupted sequence.

### Inference: iterative unmasking

```
x ← all-MASK of target length (or pre-filled prompt)
for k = 1 .. K:
    p ← π_θ(· | x)                              # model marginals
    pick a subset S of currently-masked positions to commit
    sample x_S ~ p_S  (or take argmax)
    x[S] ← x_S
return x
```

Two scheduling choices:

- **How many tokens to commit per step.** Top-$k$ by confidence, or a deterministic schedule.
- **Which positions.** "Most confident first" tends to work best — commit positions where the model's marginal is sharp.

### Variable-length generation (iLLaDA)

Vanilla masked diffusion requires a fixed sequence length set at the start. iLLaDA's improvement: predict an end-of-sequence position dynamically. Either:
- A length-predictor head that estimates $L^*$ given the partial generation.
- An explicit `<EOS>` token that the model can unmask, terminating the sequence.

Variable-length generation closes much of the inference-cost gap vs autoregressive baselines on short outputs.

### Confidence-based MCQ scoring (iLLaDA)

For multiple-choice eval, score each option by the model's marginal log-likelihood under a single full-mask forward pass — no decoding required. Better-calibrated than scoring autoregressively because the bidirectional context sees the full option.

## Why it matters

- **Parallel inference.** $L$ tokens in $K \ll L$ steps. At long $L$, this is a meaningful inference-cost reduction once the per-step model cost is amortized.
- **In-place edits.** Editing a middle span is native (mask the span, re-unmask) — no costly KV-cache invalidation.
- **Bidirectional context for SFT.** Instruction tuning sees the full prompt + answer at once; gradients flow across the whole sequence.
- **Reopens architectural choices.** AR-only assumptions in attention design (causal masks, KV cache) don't apply; this enables fundamentally different serving systems.
- **First convincing scale.** iLLaDA-8B (12T tokens) closes most of the gap to Qwen2.5-7B without an AR teacher; +21.6 BBH, +14.9 ARC-Challenge over LLaDA at the same size.

## Gotchas & tricks

- **Mask-rate scheduling matters.** A pure uniform $t$ underweights low-noise regimes that matter at inference. Some recipes use a beta-shaped distribution skewed toward low $t$.
- **Bidirectional attention costs more memory.** No KV-cache reuse across decoding steps because the attention pattern changes when the mask changes.
- **Sampling determinism.** Top-confidence-first scheduling is greedy and deterministic. For diversity, mix in temperature on the commit step.
- **Don't bolt onto an AR model.** Masked diffusion needs the full bidirectional pretrain — distilling an AR teacher into a masked-diffusion student loses much of the benefit.
- **SFT works.** Standard supervised fine-tuning on the masked-diffusion objective (mask the assistant turn, condition on the system + user prompt) just works; iLLaDA SFTs for 12 epochs on a 25B-token instruction corpus.
- **No KV cache, but no autoregressive decode dependency either.** Different speed/quality tradeoff than AR — strong fit for long outputs decoded in few steps; weak fit for short outputs.

## Sources

- Paper: *Improved Large Language Diffusion Models* (iLLaDA) — Min, Xu, Huang, Song, Shan, Lin, Zhao, Li, Wen, 2026 — [arXiv 2606.25331](https://arxiv.org/abs/2606.25331).
- Predecessor: *LLaDA* (Large Language Diffusion Model) — Nie et al., 2025 — first 7B-scale masked-diffusion LM.
- Foundational: *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM) — Austin et al., 2021 — [arXiv 2107.03006](https://arxiv.org/abs/2107.03006).
- Continuous-time view: *Simplified and Generalized Masked Diffusion for Discrete Data* — Shi et al., 2024.
