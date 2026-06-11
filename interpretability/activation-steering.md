# Activation steering

*Depth — controlling model behaviour by modifying internal activations at inference.*

**TL;DR:** Pick a direction $v$ in the model's residual stream that corresponds to a behaviour (refusal, sycophancy, a stylistic feature, a topic) and add or subtract a scalar multiple of $v$ at a chosen layer during inference. The model's downstream behaviour shifts accordingly — at zero training cost, with the same weights, no prompt engineering. The directions can be hand-picked (mean difference between two activation sets), discovered via [SAE](sparse-autoencoders.md) features, or learned with small probes.

**Prereqs:** [attention](../fundamentals/attention.md), [sparse-autoencoders](sparse-autoencoders.md)
**Related:** [alignment-gating](../safety/alignment-gating.md) · [refusal-suppression](../safety/refusal-suppression.md)

---

## What it is

LLMs encode many human-interpretable behaviours as approximately-linear directions in their residual stream. Activation steering exploits this: at inference, find the direction $v$ associated with a behaviour, then at chosen layer $\ell$ replace $h_\ell \leftarrow h_\ell + \alpha \cdot v$ for some scalar $\alpha$. Positive $\alpha$ amplifies the behaviour; negative $\alpha$ suppresses it.

The recipe is shockingly cheap — no fine-tuning, no LoRA — and produces large behavioural effects. The TTS-LM result reproduced in the cited paper (laughter probability 0.02 → 0.79 from a single feature edit) is typical.

## How it works

### Finding the direction

Three common approaches, in increasing order of automation:

1. **Difference-of-means.** Collect activations on (target-behaviour, neutral) pairs at layer $\ell$; subtract the means → $v$.
2. **Probe weights.** Train a linear probe to classify behaviour from activations; the probe's weight vector is $v$.
3. **SAE feature.** Pick a feature from a trained SAE that the labelling pipeline tagged with the target behaviour; the corresponding decoder direction is $v$.

SAE features have become the default in 2024–2026 because the SAE training already discovered the relevant directions and labelled them.

### Applying the edit

At inference, for each token, modify the residual stream:

$$ h_\ell' = h_\ell + \alpha \cdot v $$

Variants:
- **One layer or all layers** — single-layer edits usually suffice for downstream behaviour.
- **Token-position scoping** — edit only at the first generated token (anchors the behaviour) or at every generated token.
- **Token-mask scoping** — edit only on specific token types (e.g. only on responses, not on input context).

### Steering strength $\alpha$

The dominant knob. Too small → no effect; too large → output collapse (the model starts generating $v$-aligned token-junk). The right range is task-dependent and modest — $\alpha$ values that yield clear behavioural shifts also leave the rest of the output coherent.

## Why it matters

- **Cheapest control mechanism in the toolbox.** No weights changed, no extra forward pass, no prompt engineering.
- **Causal interpretability.** Successful steering is positive evidence that the chosen direction *is* the behaviour's representation — not just correlated with it.
- **Production-relevant.** Audio LMs control speaker gender, accent, speech rate via single-feature edits. LLMs control refusal strength, tone, topic adherence.
- **Safety primitive.** Refusal-suppression and refusal-amplification both reduce to steering on the [refusal direction](../safety/refusal-suppression.md). Both red-team and blue-team uses.

## Gotchas & tricks

- **Direction generalization is partial.** A direction found from one context (e.g. assistant responses) may not steer cleanly on another (e.g. system-prompt responses). Validate where you'll use it.
- **Composing edits is brittle.** Editing two directions simultaneously sometimes works, sometimes interferes destructively. Compose in increasing strength and watch for output collapse.
- **Single-direction is a rough model.** Some behaviours are carried by multiple directions or by interactions; single-vector steering captures the dominant component but not all.
- **Strength curves are not monotone.** For some behaviours, increasing $\alpha$ keeps shifting the output; for others, the effect saturates and then breaks. Sweep.
- **TTS / multimodal directions work the same way.** The TTS SAE paper shows phoneme, laughter, gender, and accent are all steerable in the residual stream of a speech LM. Recipe transfers.
- **Doesn't fix misalignment robustly.** Steering refusal *up* dampens unsafe behaviour but doesn't remove the underlying capability — the model can still be jailbroken to bypass the steered residual. See [alignment-gating](../safety/alignment-gating.md) for a learned alternative.

## Sources

- Paper: *Steering Language Models With Activation Engineering* — Turner et al., 2023 — foundational.
- Paper: *Refusal in Language Models is Mediated by a Single Direction* — Arditi et al., 2024 — canonical safety application.
- Paper: *Scaling Monosemanticity* — Templeton et al., Anthropic, 2024 — SAE-derived steering at Claude scale.
- Paper: *Interpreting and Steering a Text-to-Speech Language Model with Sparse Autoencoders* — Koriagin et al., T-Tech, 2026 — steering for speech control — [arXiv 2606.10029](https://arxiv.org/abs/2606.10029).
