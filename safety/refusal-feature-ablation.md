# Refusal Feature Ablation (RFA)
*Depth — attack that projects a single linear refusal direction out of an open-weight model's residual stream.*

**TL;DR:** Refusal Feature Ablation is a family of open-weight attacks that identify a **single linear direction** in the residual stream associated with refusal behavior and then edit the model to no longer represent it — typically by projecting it out of the weights. The attack is fast, requires no fine-tuning, preserves general capability, and reliably produces "abliterated" checkpoints of released models. Variants include the original Arditi et al. abliteration, RepE-style projection, and 2026's Heretic weight-level attack.

**Prereqs:** [../interpretability/README.md](../interpretability/README.md), [_attacks](_attacks.md)
**Related:** [refusal-suppression](refusal-suppression.md), [decoy-direction-optimization](decoy-direction-optimization.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

A **weight-level attack** against safety-trained open-weight LLMs. Given the model and a small paired dataset of harmful/harmless prompts, RFA (a) estimates the direction in activation space along which the model's residual stream differs most between "refuse" and "comply" cases, then (b) applies a rank-1 edit — usually zeroing that direction in the writers to the residual stream (attention output projection, MLP down-projection) — so the refusal feature can no longer be written. The base model's other capabilities are approximately preserved.

## How it works

1. **Contrastive estimation.** Collect two small sets of prompts — one that reliably triggers refusal, one that gets compliance. Run both through the model; average residual-stream activations per layer.
2. **Refusal direction.** Compute `r = mean(refuse) − mean(comply)` at a chosen layer. Normalize.
3. **Weight edit.** For every layer that writes to the residual stream (`W_out` of attention, `W_down` of MLP), project the output orthogonal to `r`:
   `W' = W − r r^T W`.
4. **Sanity-check.** Test on held-out harmful prompts. If ASR is high enough, ship the "abliterated" checkpoint.

Refinements: multi-direction ablation, layer-selective edits, or the *Heretic* attack, which operates at the weight level directly.

## Why it matters

RFA is the reference weak-attack against open-weight safety. It is:
- **Cheap.** Minutes of CPU/GPU for a 70B; no gradient step through the safety objective.
- **Capable-preserving.** General benchmarks barely move.
- **Transferable.** Abliterated checkpoints circulate widely.

Any open-weight release with a refusal-tuned safety layer must assume RFA-style attacks will be applied on day one. Defenses that only obscure the refusal behavior with fine-tuning are cheaper to defeat than the fine-tuning cost.

## Gotchas & tricks

- **Single direction is a strong assumption.** Multi-basis refusal representations (e.g. after adversarial training) blunt the attack — but the attacker's estimator can adapt.
- **ASR ≠ full jailbreak.** RFA removes refusal but does not remove *awareness*. The model can produce refusals from context (e.g. a system prompt), and can still be caught by output-side guardrails.
- **Contrastive dataset drives the direction.** Poorly chosen prompts pick up the wrong feature (topic or style). The attacker's craft is in the dataset.
- **Adaptive attacks.** If the defender jitters weights, the attacker just re-runs the contrastive estimate on the jittered checkpoint. Static obfuscation loses.
- **Defenses that work.** See [decoy-direction-optimization](decoy-direction-optimization.md) — inject a competing high-magnitude nonlinear feature so the attacker's estimator locks onto a decoy.

## Sources

- Paper: *Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration* — Muhamed, Diab, Smith — 2026 — [arXiv:2609.16204](https://arxiv.org/abs/2609.16204).
- Origin: Arditi et al., *Refusal in LLMs is Mediated by a Single Direction*, 2024.
- Attack variant: Heretic weight-level attack, 2026.
