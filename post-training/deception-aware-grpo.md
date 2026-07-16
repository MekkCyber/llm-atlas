# Deception-Aware GRPO
*Depth — GRPO with adversarial reward shaping that penalizes answers which are visually consistent but numerically wrong.*

**TL;DR:** Standard GRPO on VLMs rewards accuracy. But charts can lie — inverted axes, distorted scales — and a VLM that trusts its visual read gets the wrong answer while feeling correct. Deception-Aware GRPO adds an **adversarial reward term** that specifically penalizes rollouts whose answer matches the visual illusion but disagrees with an independent numeric check (e.g., OCR'd data). Introduced by ChartCynics (2026); yields **~29 absolute points** over the Qwen3-VL-8B base on misleading-chart benchmarks.

**Prereqs:** [grpo](grpo.md)
**Related:** [_rewards](_rewards.md)

---

## What it is

A specialized GRPO reward-shaping recipe for perception + verification tasks where the two sources of truth (what the model sees vs. what's actually there) can disagree by adversarial design. The base RL algorithm is unchanged; the reward function is what carries the specialization.

Sits alongside other GRPO variants that shape the reward for domain-specific failure modes (e.g., length penalty in Kimi k1.5, format rewards in R1). "Deception-aware" refers specifically to shaping against **perceptual-adversarial** failures — the kind where the input itself is designed to fool perception.

## How it works

The reward is a sum of components:

1. **Base correctness reward.** Standard — did the final answer match the ground truth? (rule-based, when available).
2. **Deception penalty.** If the answer matches what a visual-only pass through the chart would predict but disagrees with an independent numeric check (OCR-extracted data, or a data-path branch of the model), apply a large negative reward.
3. **Consistency bonus.** If the vision path and the data path agree *and* the answer matches, apply a small positive reward.

Concretely, ChartCynics runs two paths internally: a **Diagnostic Vision Path** that ROI-crops and inspects structural anomalies (inverted axis, distorted scale), and an **OCR-Driven Data Path** that reads numeric ground truth from the chart. The deception penalty fires when a rollout's answer follows the vision path uncritically over the OCR data path — the exact behavior misleading charts are designed to elicit.

Training pipeline: **Oracle-Informed SFT** first (distill a reasoning trace that reconciles the two paths), then Deception-Aware GRPO to push the policy off the visually-fooled equilibrium.

## Why it matters

- **Adversarial multimodal is a real deployment concern.** Anywhere users can craft inputs to fool perception (misleading charts, doctored screenshots, deepfake docs), a correctness-only reward reinforces exactly the wrong behavior.
- **Shaping rewards to penalize specific failure modes works.** A single well-shaped reward term (~29-point gain) matches or beats architectural changes for this class of task.
- **Template for other perception-vs-verification splits.** Any task with two independent evidence sources — vision + OCR, VLM + calculator, image + retrieval — can plug into the same recipe.

## Gotchas & tricks

- **Requires an independent verification source.** OCR is the paper's; other domains need a comparable "second opinion" that's less foolable than the primary perceptual channel.
- **Shaping strength matters.** Too weak: doesn't move the policy off deception. Too strong: policy over-trusts the second source even when it's wrong (bad OCR on stylized fonts).
- **Composes with format / length rewards.** Multi-term GRPO rewards are additive; deception penalty stacks with the usual shaping suspects.
- **Not a replacement for training data.** The penalty only fires when the model produces a fooled answer to sample from; if fooled rollouts are rare in the training distribution, the penalty rarely activates. Needs adversarial data curation.

## Sources

- Paper: *Navigating the Mirage: A Dual-Path Agentic Framework for Robust Misleading Chart Question Answering* — Zhang et al., HKUST, 2026 — arXiv 2603.28583.
- Related: [grpo](grpo.md) — base algorithm; deception-aware GRPO differs only in the reward function.
