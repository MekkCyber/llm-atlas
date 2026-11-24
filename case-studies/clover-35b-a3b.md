# Case Study: Clover-35B-A3B-Preview

*A 35B-total / 3B-active mixture-of-experts **diffusion** language model, targeting agentic coding. Reaches 73.6 on SWE-bench Verified at ~680 tokens/s on a single NVIDIA GB300. The interesting part is not the benchmark number — it's that it's the first open result where the diffusion-LM track competes with autoregressive frontier coders on a hard agentic eval.*

**Related concepts:** [block-diffusion-lm](../architectures/block-diffusion-lm.md) · [gated-delta-networks](../architectures/gated-delta-networks.md) · [_moe](../architectures/_moe.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [humaneval](../evaluation/humaneval.md) · [livecodebench](../evaluation/livecodebench.md)

---

## What this is

Clover-35B-A3B-Preview, released November 2026 by DiffuSpace. A **mixture-of-experts diffusion language model** with 35B total parameters and ~3B activated per token. Unlike mainstream autoregressive LLMs, Clover is trained with a **block-diffusion** objective: the model denoises blocks of tokens jointly rather than predicting one token at a time. The sequence mixer inside each block is a variant of **Gated Delta Networks**, adapted to tolerate the non-causal within-block attention that block diffusion requires.

The paper packages four bets in one:

1. Scale a **diffusion LM** to MoE frontier size.
2. Use **block diffusion** as the training objective, trading some perplexity for parallel decoding.
3. Use **Gated Delta Networks** as the block-level mixer rather than full softmax attention.
4. Target **agentic coding** as the primary evaluation, not perplexity or short-form QA.

A separate "decision-model" variant jointly predicts answers to grouped related questions, exploiting the diffusion objective's natural support for multi-span completion.

---

## Architecture at a glance

```
~35B total parameters, ~3B activated per token (MoE)
block-diffusion objective (non-autoregressive within a block)
block size: not fully disclosed in the preview report (likely 32–128 tokens)

sequence mixer:
  Gated Delta Networks (linear-time state-space variant with gating),
  adapted for non-causal within-block attention

MoE:
  routed experts + shared expert (deepseek-moe style)
  expert routing per block, not per token
  3B active ≈ ~8–9% activation ratio, similar to recent open MoEs

diffusion denoiser head:
  predicts clean-token distribution per-position within the noised block;
  iterative refinement across diffusion steps
```

Exact expert count, routing scheme, number of diffusion steps, and block size are not fully disclosed in the preview technical report. The headline configuration (35B total / 3B active) and the mixer family (Gated Delta Networks) are confirmed.

---

## Training recipe

The preview report is a system paper, not a full pretraining recipe dump — several training specifics are not enumerated. Known structure:

**Stage 1 — Diffusion LM pretraining.** The base model is trained from scratch (or from a published GDN-class checkpoint; the paper does not disclose which) with a block-diffusion objective on a general code+text corpus. The denoising loss per block is standard masked-prediction cross-entropy summed over diffusion timesteps.

**Stage 2 — MoE expansion / fine-tuning.** The MoE routing is trained alongside or post-hoc from the dense phase (details not explicit). The routing operates at block granularity — one routing decision per block rather than per token — which keeps the diffusion head cheap.

**Stage 3 — Agentic coding post-training.** SWE-bench-Verified and coding-agent-benchmark performance is attributed to a dedicated post-training phase on agentic coding traces. The exact mix of SFT / RL / DPO is not fully disclosed in the preview; the headline is "agentic coding" rather than generic instruction-following.

**Decision-model variant.** A separate fine-tuning run exploits block diffusion's ability to jointly predict multiple related answers: for a batched set of related questions, the model fills answer spans jointly, raising accuracy on each when answers are correlated. This is a diffusion-specific capability that AR models cannot replicate without heavy prompting.

Training-compute and token-count figures are not disclosed in the preview.

---

## Key results

| Benchmark | Score | Notes |
|---|---|---|
| SWE-bench Verified | **73.6%** | Primary headline; agentic coding |
| Inference throughput | **~680 tok/s** | 1× NVIDIA GB300 (single GPU) |
| Model size | 35B total, 3B active | MoE |

Context length, HumanEval, LiveCodeBench, and non-coding benchmarks are not in the preview highlights — the paper is a targeted preview and does not attempt a general-LLM leaderboard sweep.

---

## Why the throughput number matters

Block diffusion decodes **multiple tokens per step** inside a block. A sequence of `B` tokens takes `T` diffusion steps rather than `B` autoregressive steps; if `T ≪ B` the wall-clock wins. With a 3B-active MoE and a modern GPU, this lands at ~680 tok/s on one GB300 — competitive with the fastest AR coders at similar active-parameter counts, without speculative decoding or special inference tricks.

The trade-off is that block diffusion is lossy at long-horizon planning: each block is denoised given the prior blocks, but within-block structure can collapse to the modes of the training distribution. The paper's agentic-coding numbers are evidence that, at least for structured code synthesis, this is acceptable.

---

## Novel innovations

Two of Clover's design choices are standalone contributions worth documenting:

- **[block-diffusion-lm](../architectures/block-diffusion-lm.md)** — block diffusion as the LM training objective at MoE frontier scale.
- **[gated-delta-networks](../architectures/gated-delta-networks.md)** — Gated Delta Networks as the sequence mixer, adapted to tolerate non-causal within-block attention.

The MoE recipe borrows from [DeepSeekMoE](../architectures/deepseek-moe.md) (shared + routed experts, fine-grained experts) and is not independently novel.

---

## Open questions

- **Block size and diffusion steps.** The preview does not disclose the exact block size, number of diffusion steps, or noise schedule. These are the primary knobs controlling the quality / throughput trade-off.
- **How well it generalizes outside coding.** The report is focused on SWE-bench Verified. General reasoning, long-context, and multimodal results are not published.
- **Routing stability under diffusion.** MoE routing was developed on AR losses; the preview does not report whether router balance behaves differently under block-diffusion objectives.
- **Comparability to AR-MoE baselines.** The 73.6 SWE-bench number is reported without a size-matched AR-MoE baseline from the same post-training recipe. Without that, how much of the gain is "diffusion" vs "the specific post-training mix" is unclear.

---

## Sources

- Paper: *Clover-35B-A3B-Preview Technical Report* — Chen et al., DiffuSpace, 2026 — [arXiv:2611.24636](https://arxiv.org/abs/2611.24636)
