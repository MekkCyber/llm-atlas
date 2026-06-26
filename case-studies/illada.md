# Case Study: iLLaDA

*An 8B fully bidirectional masked-diffusion LLM trained from scratch on 12T tokens, then SFT-ed on a 25B-token instruction corpus for 12 epochs. The strongest published evidence to date that masked-diffusion language models can scale into "regular LLM" territory and stay competitive with same-size autoregressive baselines.*

**Related concepts:** [masked-diffusion-llm](../architectures/masked-diffusion-llm.md) · [transformer-block](../architectures/transformer-block.md) · [_normalization](../architectures/_normalization.md) · [_lr-schedules](../pre-training/_lr-schedules.md) · [_training-stability](../pre-training/_training-stability.md) · [fp8-training](../pre-training/fp8-training.md) · [evaluation/math500](../evaluation/math500.md) · [evaluation/humaneval](../evaluation/humaneval.md)

---

## What this is

**iLLaDA** ("improved LLaDA") is an 8-billion-parameter language model released June 2026 by Renmin University of China and ByteDance Seed (Min, Xu, Huang, Song, Shan, Lin, Zhao, Li, Wen). It is *not* an autoregressive Transformer: every layer uses fully bidirectional attention, and the training objective is masked-diffusion denoising, kept end-to-end through pretraining *and* supervised fine-tuning.

The paper does two things:

1. **Scales masked-diffusion language modeling.** 12T-token pretrain at 8B parameters, 25B-token SFT corpus for 12 epochs — the first masked-diffusion LM evaluation at this scale.
2. **Closes the gap to autoregressive baselines.** Substantial improvements over the predecessor LLaDA at matched size on general, math, and code benchmarks, and competitive with Qwen2.5-7B on several tasks despite the non-autoregressive objective.

For the underlying technique see [masked-diffusion-llm](../architectures/masked-diffusion-llm.md).

---

## Architecture

| Component | Choice |
| --- | --- |
| Parameters | 8B |
| Attention | Fully **bidirectional** (no causal mask) |
| Positional encoding | RoPE (carried over from standard transformer LMs) |
| Normalization | Pre-norm RMSNorm (paper's standard substrate) |
| Vocab | Standard SentencePiece-style tokenizer |
| Special tokens | `[MASK]` (training + inference), `<EOS>` for variable-length termination |
| Inference path | Iterative unmasking with variable-length termination |

The model is, intentionally, a *standard* transformer with two changes:

- The attention mask is identity (bidirectional), not lower-triangular.
- The output head is interpreted as a per-position **marginal** over the vocab, not as a next-token prediction.

Everything else — RMSNorm, RoPE, SwiGLU FFN — matches the AR transformer recipe so that pretraining-stability findings transfer directly.

---

## Training recipe

### Stage 1 — pretraining (12T tokens)

- **Objective.** Masked-diffusion denoising at a continuum of mask rates $t \in (0, 1]$. Per-batch, sample $t$; mask each token independently with probability $t$; predict all masked tokens in parallel.
- **Loss weighting.** ELBO-correct $1/t$ weighting on the per-position cross-entropy so high-noise batches don't dominate.
- **Data mix.** Web + code + math + multilingual (mix not fully detailed in the abstract). 12T tokens is comparable to 8B AR training budgets (Llama 3-8B uses ~15T tokens).
- **Hyperparameters.** Schedule and optimizer details not in the abstract — paper text required.

### Stage 2 — supervised fine-tuning (25B tokens, 12 epochs)

- **Same objective.** Masked diffusion *throughout*. No pivot to AR for SFT.
- **Procedure.** Mask the assistant turn (and possibly part of the user turn for instruction-following robustness); condition on the system + user prefix and the still-visible response tokens; predict the masked tokens.
- **Why 12 epochs.** Masked diffusion appears to benefit from more passes over instruction data than AR SFT does, because each pass sees a different random mask pattern — effectively augmented coverage of the same examples.

The "kept the masked-diffusion objective throughout" point is the central design choice: alternative recipes pivot to AR at SFT and lose the inference benefits. iLLaDA does not.

---

## Inference

Two new inference-time techniques the paper highlights:

- **Variable-length generation.** Rather than fix the output length up front, the model can predict an `<EOS>` token during unmasking; once committed, the sequence terminates. Closes a long-standing inference-cost gap between masked diffusion and AR on short outputs.
- **Confidence-based MCQ scoring.** For multiple-choice evaluation, score each option by the model's marginal log-likelihood under a *single full-mask* forward pass — no decoding required. Better-calibrated than AR scoring because the bidirectional context sees the full option at once.

Detailed mechanism in [masked-diffusion-llm](../architectures/masked-diffusion-llm.md).

---

## Key results

Headline gains over the LLaDA predecessor at matched 8B size:

| Benchmark | iLLaDA-Base over LLaDA-Base |
| --- | --- |
| BBH | **+21.6 points** |
| ARC-Challenge | **+14.9 points** |
| Other general / math / code | Broad improvements across the board |

After instruction tuning:

| Benchmark | iLLaDA-Instruct over LLaDA-Instruct |
| --- | --- |
| MATH | **+14.5 points** |
| HumanEval | **+16.5 points** |

Versus the Qwen2.5-7B autoregressive baseline:

- Competitive on several benchmarks (specific list in the paper).
- The gap that remains is small enough that the architectural choice (masked diffusion) is no longer an obvious disadvantage.

---

## Why it matters

- **First convincing 8B masked-diffusion LM.** Before iLLaDA, the masked-diffusion track was viewed as a research curio that didn't reach AR parity. The 12T-token scale is what closes most of the gap.
- **No AR teacher, no distillation.** iLLaDA is trained masked-diffusion-from-scratch. This rules out the "you secretly need an AR teacher" hypothesis.
- **Reopens the inference story.** Parallel multi-token decoding, in-place edits, and length-flexible generation are native to masked diffusion. If the AR-parity trend holds at larger scale, serving stacks change shape.
- **Architectural fork in the road.** The default for the past five years has been "if it's a big LM, it's autoregressive." iLLaDA argues that's an empirical finding (driven by what scale was tried), not a theoretical necessity.

---

## Limitations and open questions

- **Wall-clock inference cost vs AR.** Bidirectional attention precludes KV-cache reuse across unmasking steps. Whether the parallel-decode advantage actually beats AR on real serving hardware (FlashAttention KV-cache + speculative decode) at 8B is unsettled.
- **Long-context behavior.** Bidirectional attention is $O(L^2)$ memory; AR with KV cache is $O(L)$ per step. Pretraining length and inference-time long-context degradation aren't reported in the abstract.
- **Tool use / agentic settings.** Most agent code paths assume causal decoding (streaming tool calls, partial JSON parsers). Masked-diffusion serving needs new abstractions.
- **No RL post-training reported.** The paper is SFT only. Whether RL-with-verifiable-rewards (RLVR / GRPO) on a masked-diffusion model works at all is open.
- **Reproducibility outside the lab.** Weights, code, and the full mix aren't covered in the abstract. Adoption depends on the release.

---

## Sources

- Paper: *Improved Large Language Diffusion Models* — Qiyang Min, Shaoxuan Xu, Zihao Huang, Yuxuan Song, Yong Shan, Yankai Lin, Wayne Xin Zhao, Chongxuan Li, Ji-Rong Wen — 2026 — [arXiv 2606.25331](https://arxiv.org/abs/2606.25331).
- Predecessor: *LLaDA* — Nie et al., 2025.
- Foundational masked-diffusion: *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)* — Austin et al., 2021.
- Continuous-time formulation: *Simplified and Generalized Masked Diffusion for Discrete Data* — Shi et al., 2024.
