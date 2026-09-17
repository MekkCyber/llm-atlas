# Decoy Direction Optimization (DDO)
*Depth — post-hoc weight-editing defense against Refusal Feature Ablation attacks.*

**TL;DR:** DDO is a fast, no-finetuning defense against open-weight refusal-ablation attacks. Instead of hiding the true refusal circuit, DDO **injects a high-magnitude, nonlinear "decoy" signal** into MLP neurons so that a contrastive attacker's estimator locks onto the decoy and ablates a harmless orthogonal direction — leaving the real safety mechanism intact. Reduces standard RFA ASR to <10% across six model families and cuts Heretic weight-attack ASR from 88.7% → 18% at 30–450× lower cost than trained defenses.

**Prereqs:** [refusal-feature-ablation](refusal-feature-ablation.md), [_attacks](_attacks.md)
**Related:** [refusal-suppression](refusal-suppression.md), [../interpretability/README.md](../interpretability/README.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

A weight-editing defense that ships with an open-weight checkpoint. It runs once, on CPU minutes, and produces a checkpoint that:
- Behaves identically to the original on benign prompts.
- Refuses at the same rate on harmful prompts.
- Is significantly harder to abliterate.

It does not require access to safety fine-tuning data or new training runs.

## How it works

**Attacker model.** RFA-style attacks use a **contrastive estimator** on paired harmful/harmless activations to find a refusal direction. The estimator is dominated by the direction of **largest activation difference** between the two sets.

**DDO's move.** In selected MLPs, DDO adds a **high-magnitude, nonlinear decoy feature** that:
1. Fires strongly on a decoy dimension the attacker's estimator will detect.
2. Is *orthogonal* to the true refusal circuit.
3. Contributes ~0 to output logits (it fires but its effect is nulled by a cancelling edit downstream).

Because the decoy is high-magnitude, contrastive estimators prefer it over the true refusal direction. The attacker ablates the decoy — safely — and the true safety circuit remains functional. The paper proves a **spectral bound** formalising when the decoy dominates the estimator.

## Why it matters

Open-weight releases can't rely on fine-tuning as a defense: any fine-tuned safety layer can be abliterated for less compute than it took to train. DDO shifts the game to *making the refusal circuit hard to find*, not hard to remove.

- **<10% ASR** under standard RFA across six model families.
- Under Llama-3-8B-Instruct **adaptive multi-phase attacks**, comparable to expensive trained defenses (65% vs. 58% worst-case ASR).
- **Heretic weight-level attack** ASR: **88.7% → 18%**.
- **30 to 450× lower optimization cost** than trained baselines.

## Gotchas & tricks

- **Only fights ablation-style attacks.** DDO does nothing about prompt-side jailbreaks (roleplay, low-resource, etc.) — pair it with prompt-side defenses.
- **Decoy has to be nonlinear.** A linear decoy is easy for the attacker to decompose and factor out.
- **Adaptive attackers can co-target.** If they know DDO is applied, they can compute *two* directions and ablate both — the paper's spectral bound still holds under some parameter regimes but degrades if the decoy magnitude drops.
- **Downstream cancellation matters.** If the decoy's output effect leaks into logits, benign behavior degrades. Verify on general benchmarks after applying DDO.
- **Layer selection.** DDO is applied only in a subset of MLP layers; picking the wrong layers wastes the decoy's magnitude.

## Sources

- Paper: *Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration* — Muhamed, Diab, Smith — 2026 — [arXiv:2609.16204](https://arxiv.org/abs/2609.16204).
- Attack this defends against: [refusal-feature-ablation](refusal-feature-ablation.md).
