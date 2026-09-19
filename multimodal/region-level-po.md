# Region-Level Policy Optimization (Vision-RL2)
*Depth — treating region selection as an RL action for token-efficient fine-grained MLLM perception.*

**TL;DR:** Fine-grained MLLM perception is usually improved by raising input resolution, which inflates visual-token count and prefill cost. Vision-RL2 factors perception into two resolutions: **coarse localization** and **fine recognition**. A lightweight proposal network (distilled from the base MLLM's attention) proposes coherent image regions as **actions**; the MLLM then does per-region recognition. Trained with subtractive + additive RL objectives, Vision-RL2 beats the base model's largest-budget accuracy with ~4× fewer visual tokens across 6 benchmarks and 4 MLLM architectures.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

A post-training method for multimodal LLMs that turns *visual tokenization* into a policy problem. Instead of tokenizing the whole image at high resolution, an **RL-trained proposal policy** picks a set of coherent image regions; the base MLLM then processes each region at moderate resolution.

Two novel pieces:

- **Attention-distilled proposal network.** A small side head is trained to imitate the base MLLM's attention over the image, giving a cheap prior on where to look. No external detector needed.
- **Region-as-action RL.** The proposal policy is trained with RL where its actions are region proposals. The reward is downstream task performance (verifiable, when applicable). Objectives combine a **subtractive** term (penalize proposals that don't help) and an **additive** term (reward proposals that unlock correct answers).

---

## How it works

```
image ─► proposal net (distilled from attention) ─► candidate regions
                                                              │
                                          RL selects a subset (policy π)
                                                              │
                                          MLLM processes each region at moderate res
                                                              │
                                                    downstream task answer
                                                              │
                                                    verifiable reward → GRPO
```

The subtractive-plus-additive objective is the key: naive selection reward would over-select (adding regions doesn't hurt); the subtractive component prunes proposals that add no marginal value; the additive component keeps regions that unlock a correct answer.

Because the proposal network is a lightweight side head, the whole system runs at ~1/4 the visual-token count of the base model at matched accuracy.

---

## Why it matters

- **Token cost is the MLLM inference bottleneck.** Raising input resolution multiplies both vision-encoder and language-model prefill cost. Vision-RL2 replaces the "raise resolution" default with region-level selection.
- **Cross-architecture transferability.** Reported gains hold across four MLLM architectures — the recipe isn't tied to a particular vision encoder.
- **First principled RL over visual tokenization.** Prior work either uses fixed pruning heuristics or attention-based static reduction. Treating region selection as a *policy* trained with RL is a new direction.

---

## Gotchas & tricks

- **Proposal distillation is load-bearing.** A poor prior blows up the RL search space. Distill carefully from the base MLLM's own attention.
- **Reward must be verifiable.** RL needs a clean signal. On open-ended VQA, region selection is undertrained; on tasks with verifiable answers (grounding, counting), it works.
- **Subtractive-plus-additive shaping.** Dropping either half regresses to trivial policies (select-all or select-none).
- **Moderate per-region resolution matters.** Too low → recognition fails; too high → defeats the point. Tune per-region resolution jointly with region count.
- **Not for real-time.** The proposal step adds latency; the win is on token cost / batched throughput, not on per-request wall clock.

---

## Sources

- Paper: *Region-Level Policy Optimization for Fine-grained MLLM Perception* — Shi et al., 2026 (U. Sydney / City U. HK) — [arXiv:2609.19745](https://arxiv.org/abs/2609.19745).
