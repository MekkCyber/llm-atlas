# Case Study: Kwai Keye-VL-2.0

*A 30B-total / 3B-active MoE multimodal foundation model from Kwai (Kuaishou) targeting long-video understanding and agentic intelligence. The interesting part is not the topline scores — it's that the paper bundles a multimodal-adapted DeepSeek Sparse Attention, a 13-teacher cross-modal distillation recipe, a four-stage context extension curriculum to 256K, and Video-RL with rule-verifiable temporal rewards, all on top of an MoE backbone.*

**Related concepts:** [dsa-multimodal](../multimodal/dsa-multimodal.md) · [mopd](../post-training/mopd.md) · [video-rl](../post-training/video-rl.md) · [_moe](../architectures/_moe.md) · [deepseek-moe](../architectures/deepseek-moe.md) · [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) · [multi-head-attention](../architectures/multi-head-attention.md) · [grpo](../post-training/grpo.md) · [rlvr](../post-training/rlvr.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What this is

Kwai Keye-VL-2.0-30B-A3B, released June 2026 by Kwai/Kuaishou. A vision-language Mixture-of-Experts model: 30B total parameters, 3B activated per token. Supports lossless 256K-token context over interleaved video + image + text, with hour-level video as a first-class input. Trained for video understanding, temporal grounding, reasoning, STEM, and agentic tool/code/search workflows. Open-source release.

The paper bundles four innovations that each stand alone:

- **DeepSeek Sparse Attention adapted to GQA-based multimodal.** Brings DSA's MQA-style "Lightning Indexer" + sparse aggregation into a GQA stack, enabling 256K context at sub-quadratic cost without dropping video frames.
- **Cross-Modal Multi-Teacher On-Policy Distillation (MOPD).** Thirteen RL-trained domain teachers (math, code, OCR, grounding, counting, video, tool use, …) provide token-level feedback on the student's on-policy rollouts, with an overlap-set advantage and a token-category-aware scaling that down-weights formatting tokens.
- **Four-stage context-extension curriculum** that grows from 32K to 256K with a 1:1 long-to-short data ratio in the final stage.
- **Video-RL with rule-verifiable temporal rewards.** Temporal IoU for grounding, LLM-as-Judge for dense captioning, and a synthetic *FrameForge* video benchmark with timestamp / counting / reasoning rule rewards.

Topline numbers: Video-MME-v2 (512 frames) **42.4** vs Qwen3.5's 28.5; LongVideoBench **74.1** vs 61.6; TimeLens ActivityNet **58.5**, QVHighlights **70.1**; MMMU **80.0**; LiveCodeBench v6 **64.2**; τ²-Bench **82.6**.

---

## Architecture at a glance

```
MoE backbone: 30B total, 3B active per token (~10:1 sparsity)
Attention:    GQA + DSA (DeepSeek Sparse Attention) for multimodal
  ├─ MQA-style Lightning Indexer: global index scores, one set per token
  └─ GQA Sparse Aggregation: same sparse index set reused across GQA groups
Context:      256K tokens, lossless (no frame dropping)
Sparse cost:  O(L·k) instead of O(L²), with k = 2048

Vision:       ViT encoder + projector → MoE LM
Parallelism:  heterogeneous ViT-LM parallelism (different schedules per part)
```

The MoE follows the DeepSeekMoE fine-grained + aux-loss-free pattern (the paper does not disclose router-bias hyperparameters at the level DeepSeek-V3 does, but the design family is the same).

DSA-for-multimodal is the only architectural innovation; the rest of the stack reuses MoE + ViT + projector patterns from prior open multimodal models. See [dsa-multimodal](../multimodal/dsa-multimodal.md).

---

## Training infrastructure

- **Heterogeneous ViT-LM parallelism.** ViT and LM backbone have very different parameter / activation footprints; running them under different parallel-strategy schedules (more DP on ViT, more EP on the MoE LM) improves utilization.
- **Custom DSA kernels.** Lightning Indexer + sparse aggregation need fused kernels to realize the asymptotic O(L·k) advantage on real hardware.
- **Scalable video I/O.** Hour-long video at training time requires streaming decoders and async prefetch; the paper details a custom video data pipeline.

---

## Training recipe

### Stage 0 — Projector initialization

Train only the ViT→LM projector while the rest is frozen. Standard multimodal-LLM warmup.

### Stage 1 — General multimodal pre-training

- **1T tokens, 32K context.**
- Interleaved image + text + short-video data.
- Builds general vision-language grounding and basic temporal reasoning.

### Stage 2 — Task-oriented capability injection

- **2T tokens, 64K context.**
- Long-video-heavy mixture with task-targeted distributions (grounding, captioning, video QA, OCR, counting, STEM).
- This is where most of the capability acquisition happens.

### Stage 3 — 256K long-context extension

- 256K context, **1:1 long-to-short** data ratio.
- Trains the DSA Lightning Indexer + sparse aggregation pattern on real long-video sequences while keeping short-context capability through the balanced mixture.

### Post-training

- **SFT.** Standard mixed-task instruction tuning across math, code, video QA, agent tool-use, etc.
- **MOPD.** Cross-Modal Multi-Teacher On-Policy Distillation — 13 RL-trained domain teachers, student generates on-policy rollouts, router picks the relevant teacher per prompt, advantage computed only on the Top-k overlap between teacher and student distributions. Formatting tokens down-weighted via a token-category-aware schedule. See [mopd](../post-training/mopd.md).
- **General RL with GSPO.** Group Sequence Policy Optimization with four reward terms: Format, Outcome, Process, ContextRL. The Context-RL component rewards correct use of long-context information.
- **Video-RL.** Temporal IoU on grounding tasks; LLM-as-Judge on dense captioning; rule-verifiable rewards on synthetic FrameForge videos (timestamps, counting, reasoning). See [video-rl](../post-training/video-rl.md).

---

## Evaluation snapshot

Selected benchmarks (Keye-VL-2.0-30B-A3B vs Qwen3.5 comparison from the paper):

| Benchmark | Keye-VL-2.0 | Qwen3.5 |
| --- | --- | --- |
| Video-MME-v2 (512 frames) | **42.4** | 28.5 |
| LongVideoBench | **74.1** | 61.6 |
| TimeLens ActivityNet | **58.5** | — |
| TimeLens QVHighlights | **70.1** | — |
| MMMU | 80.0 | 80.4 |
| LiveCodeBench v6 | **64.2** | 62.8 |
| OJBench | **71.5** | 70.2 |
| τ²-Bench (tool use) | **82.6** | 81.2 |

The gap on Video-MME-v2 and LongVideoBench is the headline result — direct evidence the DSA + 256K-context recipe pays off on long-video.

---

## Why it matters

- **Long video moves from "expensive special case" to "first-class context."** A 3B-active MoE that handles 256K interleaved video losslessly brings hour-scale video into reach for product use.
- **A complete cross-modal post-training recipe.** MOPD is one of the first end-to-end recipes for combining many domain experts into one multimodal generalist via distillation rather than mixing data — a recipe likely to recur.
- **First adaptation of DSA outside text.** DeepSeek-V3.2 introduced DSA for text-only LLMs; Keye-VL-2.0 is the first paper to bring it to GQA-based multimodal stacks with custom kernels and a 256K eval.
- **Open weights.** The model checkpoints are released, giving the community a strong long-video baseline.

---

## Sources

- Paper: *Kwai Keye-VL-2.0 Technical Report* — Bin Wen, Changyi Liu et al., Kwai/Kuaishou, 2026 — [arXiv 2606.10651](https://arxiv.org/abs/2606.10651).
- Background: *DeepSeek-V3.2 — Sparse Attention* — DeepSeek 2025 (DSA precursor).
- Background: *DeepSeekMoE* — Dai et al. 2024 — MoE design family Keye-VL-2.0 inherits.
