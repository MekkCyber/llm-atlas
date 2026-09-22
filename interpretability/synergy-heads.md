# Synergy heads (causal per-head disentanglement in MLLMs)
*Depth — identifying and calibrating the attention heads that jointly bind visual and textual streams in multimodal LLMs.*

**TL;DR:** Most attention-based hallucination-mitigation methods look at *attention weights* — a proxy that doesn't cleanly reflect what information a head actually carries. HEAL (Qin et al., 2026) applies **causal noise intervention** to filter out causally redundant heads, then uses a **counterfactual difference-in-differences** on the survivors to classify each head by its information-flow role. This surfaces a distinct class — *synergy heads* — whose information-distribution drift correlates with hallucination, even though the modality-specific heads look statistically unchanged. Injecting dynamic calibration factors into synergy-head value vectors reduces hallucinations across multiple MLLMs without retraining.

**Prereqs:** [attention.md](../fundamentals/attention.md), [multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

A per-head diagnostic and edit for multimodal transformers. Two questions are answered mechanistically rather than empirically: (1) *which heads causally matter* for a given output, and (2) *what kind of information each mattering head carries* (modality-specific vs. joint).

The name **synergy head** describes heads in the second class — heads that combine information from both visual and textual streams, and whose combined output drifts when the model hallucinates.

## How it works

**Step 1 — Causal noise intervention.** For each head, replace its output with iid noise and measure the change in the model's prediction. Heads whose ablation doesn't move the prediction are *causally redundant* and are dropped from further analysis. This is stronger than attention-weight thresholding: a head with sharp attention can still be redundant, and a head with diffuse attention can still be pivotal.

**Step 2 — Counterfactual difference-in-differences.** For each surviving head, compare its output across paired conditions:

- Full inputs (image + text prompt).
- Text-only ablation (image replaced with a null/noise image).
- Image-only ablation (text replaced with a null prompt).

A difference-in-differences on these conditions decomposes each head's contribution into modality-specific components and a joint (synergy) component. Heads with a large joint component are the *synergy heads*.

**Step 3 — Calibration.** The key observation: hallucinations correlate with distributional drift in synergy heads' output, not with anything unusual in modality-specific heads. HEAL injects dynamic calibration factors — a per-token multiplicative scale — into the *value vectors* of synergy heads, steering the joint output back toward the factual-evidence distribution.

## Why it matters

- **Mechanism over correlation.** Prior hallucination fixes rewire attention weights or add contrastive decoding heuristics. HEAL grounds the fix in a causal analysis of *which heads* actually carry the bug, giving a much smaller, more interpretable intervention surface.
- **Transferable diagnostic.** The "causal noise intervention → DiD decomposition" recipe is not specific to hallucination. Any task where you suspect multiple modalities (or streams) interact jointly is amenable; safety and refusal circuits are obvious candidates.
- **No retraining required.** Value-vector calibration is applied at inference. HEAL reduces hallucinations across multiple MLLMs with a single mechanism that ports across model families.

## Gotchas & tricks

- **Redundant-head filtering matters.** Skipping the causal-noise step and going straight to DiD produces noisy synergy-head assignments — many statistically-joint heads turn out to be causally silent. Both stages are load-bearing.
- **DiD needs matched ablations.** Replacing an image with pure noise is a stronger intervention than replacing it with a semantically-neutral gray patch; the choice affects which heads score as modality-specific vs. synergy. The paper uses noise ablation; other choices are defensible.
- **Calibration factor is dynamic, not static.** A single global scale on synergy-head values won't work — the factor is a function of the current token's context. Static-factor ablations in the paper underperform.
- **Not a full fix for hallucination.** HEAL reduces hallucination rates but doesn't eliminate them — synergy-head calibration addresses one causal path, not every one. Composes with contrastive decoding.

## Sources

- Paper: *MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads* — Qin, Chen, Liu, Xing, Wang, Han, 2026 — [arXiv:2609.09206](https://arxiv.org/abs/2609.09206).
- Related: standard MLLM hallucination-mitigation methods (attention re-weighting, contrastive decoding).
