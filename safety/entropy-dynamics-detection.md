# Entropy-Dynamics Jailbreak Detection
*Depth — training-free jailbreak detector built on per-layer, per-position entropy trajectories read through the logit lens.*

**TL;DR:** Static aggregates of prompt-level entropy (mean, variance) carry almost no jailbreak signal. But *how* per-token entropy evolves across layers and positions does — measured by monotonic rank-based trend scores. Crucially, the signal is concentrated in **intermediate layers** (not the final one), and is architecture-consistent across Llama, Qwen, and Gemma without any training. The Nikolenko et al. (2026) detector is an external safety filter that cannot be evaded by output-shaping attacks.

**Prereqs:** [_jailbreaks](_jailbreaks.md), [interpretability/logit-lens](../interpretability/logit-lens.md)
**Related:** [cot-monitoring](cot-monitoring.md), [competing-objectives](competing-objectives.md), [mismatched-generalization](mismatched-generalization.md)

---

## What it is

A black-box (per-model) jailbreak classifier that requires no fine-tuning. Input: a prompt. Procedure: forward-pass through the frozen LLM, compute per-token logit-lens entropy at each layer, summarize the entropy trajectory with a monotonic-trend statistic per layer, classify on the layer that maximizes class separability.

## How it works

1. **Forward pass.** Push the prompt through the model. Capture residual-stream states $h_\ell[t]$ for layers $\ell \in [1, L]$ and positions $t \in [1, T]$.
2. **Logit lens.** Project each $h_\ell[t]$ through the model's unembedding to get a vocabulary distribution.
3. **Per-position entropy.** Compute the Shannon entropy of each distribution, yielding $E_\ell \in \mathbb{R}^T$ — one entropy trajectory per layer.
4. **Monotonic trend score.** For each layer, compute a rank-based trend statistic on $E_\ell$ (e.g. Mann-Kendall): does entropy *rise* or *fall* monotonically across positions?
5. **Select discriminative layers.** Pick the layer (or set of layers) where the trend statistic best separates benign vs jailbreak prompts on a small calibration set.
6. **Classify.** Threshold the trend score at the selected layer(s).

The headline empirical finding: the most discriminative layers are in the **middle of the network**, not at the final layer. The final-layer trajectory is degraded because the output head re-normalizes the harmful-intent representation away.

## Why it matters

- **Training-free.** No labeled jailbreak set required for training — just a small calibration set to pick the layer.
- **Architecture-consistent.** Works across Llama, Qwen, and Gemma without re-tuning.
- **Robust to output-shaping attacks.** Detection happens in mid-network representations, *before* any decoded token. Attacks that hide harm in the output (style-injection, encoding obfuscation) don't help the prompt evade detection.
- **Interpretability claim.** Provides evidence that jailbreak-relevant structure lives in intermediate representations and is *suppressed* by the final head — a constraint on how safety training works.

## Gotchas & tricks

- **Calibration layer is model-specific.** The "best" layer varies by model family — Llama vs Qwen pick different mid-network depths.
- **Sequence-length sensitivity.** Trend statistics are noisy at very short prompts; require ≥20–30 tokens for stable estimates.
- **Confound with reasoning prompts.** Long, deliberate prompts can show similar trajectories to jailbreaks. Use a held-out benign-CoT calibration set to set the threshold.
- **Logit lens is required.** For closed-weight models you can't read intermediate residuals, so this detector is open-weight-only.
- **Pairs with classical filters.** Use as an additional layer in a defense-in-depth stack — keyword filters, output classifiers, constitutional classifiers — not as a replacement.
- **Not adversarially robust by default.** A motivated attacker with white-box access could craft prompts that flatten the entropy trajectory. Adversarial robustness is an open question.

## Sources

- Paper: *What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics* — Nikolenko, Papucci, Manchingal et al., 2026 — [arXiv 2606.25182](https://arxiv.org/abs/2606.25182).
- Related primitive: [interpretability/logit-lens](../interpretability/logit-lens.md).
- Companion: *Do Thinking Tokens Help with Safety?* — Ri, Panigrahi, Arora, 2026 — [arXiv 2606.25013](https://arxiv.org/abs/2606.25013) — adjacent claim that safety decisions are formed early in hidden representations.
