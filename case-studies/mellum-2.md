# Case Study: Mellum 2

*JetBrains' open-weight 12B-total / 2.5B-active MoE model specialized for software engineering. The interesting part isn't a single innovation — it's that **every architectural choice was ablated against per-token inference cost on commodity GPUs**, then stacked into a coherent recipe (MoE + GQA + SWA + MTP-as-draft + layer-selective YaRN), trained with Muon under FP8 hybrid precision on a 3-phase WSD curriculum, and post-trained with SFT + RLVR into both Instruct and Thinking variants.*

**Related concepts:** [_moe](../architectures/_moe.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [mtp](../pre-training/mtp.md) · [fp8-training](../pre-training/fp8-training.md) · [wsd-schedule](../pre-training/wsd-schedule.md) · [rlvr](../post-training/rlvr.md) · [sliding-window-attention](../architectures/sliding-window-attention.md) · [yarn](../architectures/yarn.md) · [muon](../pre-training/muon.md)

---

## What this is

**Mellum 2**, released 2026 by JetBrains, arXiv 2605.31268. A decoder-only Mixture-of-Experts transformer: **12B total parameters, 2.5B activated per token**, 64 experts (top-8), 10.6T pre-training tokens. The successor to the completion-focused 4B dense Mellum, and the first JetBrains release positioned as a general-purpose software-engineering model — code generation, editing, debugging, multi-step reasoning, tool use, function calling, and conversational programming. Released under Apache 2.0, with **base**, **instruct**, and **thinking** checkpoints.

The argument the report builds isn't "we beat the big labs"; it's **"we designed an end-to-end recipe optimized for per-token inference cost on commodity GPUs, validated each choice by ablation, and stayed competitive with 4B–14B baselines at the cost of a 2.5B dense model."** Every architectural decision has a measured inference-cost justification.

---

## Architecture at a glance

```
12B total / 2.5B active per token

MoE FFN per layer
  ├─ 64 routed experts, top-8 active
  └─ DeepSeek-MoE-style fine-grained pattern

Attention
  ├─ Grouped-Query Attention (GQA), 4 KV heads
  └─ Sliding Window Attention on 3 of every 4 layers
     (the 4th layer uses full attention)

Output head
  └─ Single Multi-Token Prediction (MTP) head
     ├─ used as auxiliary pre-training loss
     └─ kept at inference as the built-in speculative-draft model

Context window
  ├─ 32K during pre-training
  └─ Layer-selective YaRN extension to 128K
```

Each component was selected for *inference* cost on commodity GPUs:

- **MoE for capacity at low active FLOPs.** 12B effective capacity at 2.5B per-token compute. See [_moe](../architectures/_moe.md), [deepseek-moe](../architectures/deepseek-moe.md).
- **GQA-4 for KV-cache compression.** 4 KV heads (vs full MHA's all-heads) reduces KV-cache memory by the head ratio.
- **SWA on 3-of-4 layers** ([sliding-window-attention](../architectures/sliding-window-attention.md)) caps the attention cost on most layers at $O(W)$ for window size $W$, while the 1-of-4 full layers preserve global mixing.
- **MTP head ([mtp](../pre-training/mtp.md)) doubling as speculative draft** — earns its weights at pre-training and again at inference. No external draft model.
- **Layer-selective YaRN ([yarn](../architectures/yarn.md))** extends only the full-attention layers' positional encoding to 128K, since the SWA layers' effective range is already window-bounded.

---

## Training infrastructure

### Optimizer: Muon, with FP8 hybrid precision

Muon ([muon](../pre-training/muon.md)) is the optimizer — a recent matrix-aware optimizer that has emerged as a competitive alternative to AdamW for transformer pretraining. Muon's update rule operates on parameter matrices rather than vectors, exploiting low-rank structure in transformer gradients.

Compute lives in FP8 ([fp8-training](../pre-training/fp8-training.md)) with hybrid-precision components retained in BF16/FP32 (norms, attention internals, embeddings, LM head, MoE gating). The exact set of high-precision components is documented in the report; it broadly matches DeepSeek-V3's recipe.

### LR schedule: Warmup-Hold-Decay with linear decay to zero

A three-phase WSD ([wsd-schedule](../pre-training/wsd-schedule.md)) schedule with **linear** decay to zero. Linear decay (vs cosine) makes the final-phase checkpoint reusable — the report uses this property to fork the stable checkpoint for the post-training pipeline without re-running the bulk of pretraining.

---

## Pre-training recipe

### 10.6T tokens, three-phase code-mixture ramp

| Phase | Code/math ratio | Notes |
| --- | --- | --- |
| Phase 1 | ~23% | Diverse web data dominant; establishes general knowledge |
| Phase 2 | ~42% | Code/math ramped up; broad reasoning skills consolidate |
| Phase 3 | ~59% | Curated code and mathematical content dominant; specialization |

The progressive ramp avoids the "all-code-from-scratch" failure mode (poor general knowledge) and the "general-then-code" failure mode (loses general fluency during the code-only phase). The 23 → 42 → 59 trajectory is the report's empirically-chosen Pareto path.

### Context extension via layer-selective YaRN

After the 32K pretraining, [YaRN](../architectures/yarn.md) extends to 128K. The "layer-selective" twist: YaRN's RoPE-frequency interpolation is applied only to the full-attention layers (1-of-4). The SWA layers are left as-is — their effective range is already bounded by the window. The result is a 128K-context model without YaRN's interpolation artifacts on the bulk of layers.

---

## Post-training

Two stages on top of the pretrained base, producing both released checkpoint variants:

### Stage 1 — SFT

Supervised fine-tuning on curated instruction-following + code-task data. Standard recipe.

### Stage 2 — RLVR

[Reinforcement learning with verifiable rewards](../post-training/rlvr.md) — rule-based verifiers (test pass, format match, math answer match) drive a GRPO-style update. The verifiable reward shape is well-suited to the code and math tasks Mellum 2 specializes in.

### Two released variants

- **Instruct** — answers directly. SFT + RLVR with shorter-CoT data and standard answer-format reward.
- **Thinking** — emits an explicit reasoning trace before its final answer. Trained on long-CoT data with reasoning-format rewards.

Both checkpoints share the same base; the two post-training tracks diverge at SFT.

---

## Inference mode: MTP as built-in speculative draft

The MTP head trained as an auxiliary pre-training objective is *kept* at inference and reused as the speculative draft model. The pattern: at position $t$, the MTP head predicts $t+2$ from the main backbone's hidden state plus the embedding of $x_{t+1}$. The main model verifies; on match, both tokens are accepted. This is the same trick DeepSeek-V3 uses ([deepseek-v3](deepseek-v3.md)) and is one of MTP's two motivations alongside the denser pretraining signal.

No external draft model. The 12B-total / 2.5B-active model already carries its own draft.

---

## Key takeaways

1. **Design the architecture against the inference budget.** Mellum 2's interest is methodological: every component (MoE, GQA, SWA, MTP, YaRN) was ablated for inference cost on commodity GPUs. The result is a 2.5B-active model that ships at 4B–14B baseline quality.

2. **The "MoE + GQA + SWA + MTP-as-draft + layer-selective YaRN" stack is reusable.** None of the individual pieces are new; the report's contribution is that they compose without quality regression and at the *budget targets* a developer-tools vendor needs.

3. **MTP-as-draft pays twice.** Once as a pretraining signal (denser supervision per position, stronger hidden states), once as the inference speculative draft (~1.8× speedup at 85–90% acceptance, per the broader MTP-as-draft pattern).

4. **Layer-selective YaRN is the natural extension under SWA.** Pure-YaRN on every layer pays an interpolation cost that the windowed layers don't need; restricting to full-attention layers gives back most of the cost while preserving the long-context behavior.

5. **WSD with linear decay enables cheap post-training divergence.** Stable-phase checkpoint → multiple short decay-phase runs → SFT/RLVR forks for Instruct and Thinking from a near-shared base. Linear-decay WSD is the schedule that makes this cheapest.

6. **Open-weight tech reports from developer-tool vendors are filling a niche.** Mellum 2 sits next to DeepSeek-V3 and OLMo 2 as a fully-documented open release, but optimized for the **inference-cost-on-commodity-GPUs** regime that code-completion products actually run in.

---

## What's still opaque

- **Exact ablation tables for each architectural choice.** The report says every choice was validated by ablation, but only summary results are published — the per-ablation numbers (e.g., SWA frequency, YaRN layer set) are not fully tabulated in the public version.
- **Muon hyperparameters at this scale.** Muon's published recipes are mostly at smaller scale; Mellum 2's specific schedule and second-moment treatment at 12B / 10.6T tokens is documented but not framework-released.
- **MoE balancing strategy.** Whether Mellum 2 uses aux-loss, sequence-wise, or aux-loss-free balancing is mentioned but not fully detailed in the public report.
- **Post-training data sources.** SFT and RLVR data composition is summarized; exact mixtures and sizes are not disclosed.

---

*Pairs well with:* [deepseek-v3](deepseek-v3.md) for the larger-scale MoE+FP8+MTP reference, and [olmo-2](olmo-2.md) for the contrast in openness scope (OLMo 2 fully opens training trajectory; Mellum 2 opens weights + a detailed recipe but not intermediate checkpoints).
