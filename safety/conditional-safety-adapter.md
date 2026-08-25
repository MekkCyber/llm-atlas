# Conditional Safety Adapter
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A safety LoRA whose activation strength is controlled by a **hidden-state gate** conditioned on the input. On benign prompts the adapter is mostly off (backbone runs near-unchanged); on harmful prompts it engages. Trims the utility tax that comes from globally-applied safety SFT or standard LoRA. Introduced as **CLEAR** by Wang et al. (2026); improved HarmBench robustness with less utility degradation than SFT/LoRA baselines.

**Prereqs:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [refusal-suppression.md](./refusal-suppression.md) · [_jailbreaks.md](./_jailbreaks.md) · [_attacks.md](./_attacks.md)

---

## What it is

Traditional safety tuning changes the model's response distribution *globally*: SFT on refusal data or LoRA fine-tuning fires on every input, including benign ones. This costs measurable utility — overrefusal, style shift, degraded reasoning on non-adversarial prompts.

A **conditional safety adapter** decouples "when to safety-correct" from "how to safety-correct":

- The *how*: a low-rank adapter (LoRA-style) trained to refuse harmful continuations.
- The *when*: a lightweight per-input gate over hidden states that continuously scales the adapter's contribution.

Benign input → gate ≈ 0 → backbone runs unmodified. Harmful input → gate ≈ 1 → adapter fully applied. Continuous scaling (not binary) is important for smooth training and calibration.

## How it works

CLEAR's setup:

```
h_L      = frozen backbone hidden state at layer L
g(h_L)   = small MLP → sigmoid ∈ [0, 1]           # the gate
Δh_safe  = LoRA_safe(h_L)                          # the safety correction
h_L'     = h_L + g(h_L) · Δh_safe                  # conditional application
```

Training minimizes safety loss on harmful prompts (adversarial dataset like HarmBench) while adding a regularizer that keeps `g(h_L) ≈ 0` on benign prompts (utility dataset). The frozen backbone is untouched. The gate and the safety LoRA are jointly trained.

## Why it matters

- **Utility-preservation.** On the same HarmBench robustness level, CLEAR reports substantially smaller utility drop than SFT or globally-applied LoRA safety tuning.
- **Architectural, not filter-based.** Contrast with output-side guardrails (Llama Guard, Prompt Guard) — those are external moderators; CLEAR is a change to the *model's own forward pass* that adapts to input.
- **Composable with jailbreak defenses.** Nothing prevents pairing CLEAR with input filters or output moderators. The gate is a *first line*, not the only one.
- **Cheap to add to an existing model.** The safety LoRA and the gate together are small (typical LoRA rank 8–32 + a two-layer MLP). No retraining of the backbone.

## Gotchas & tricks

- **Gate calibration is the failure mode.** Miscalibrated gates either overrefuse (gate too eager) or leak harmful completions (gate too lax). The utility-regularizer weight is the primary knob.
- **Adversarial inputs targeting the gate.** A jailbreak that phrases a harmful request in a benign-looking hidden state defeats the gate. Robustness of the gate needs its own adversarial evaluation, separate from the adapter's.
- **Layer choice for the gate matters.** Applying at a very early layer (before content is disambiguated) makes the gate unreliable; applying too late means the harmful continuation is already in flight. Mid-network is typical.
- **Interaction with system prompts.** Safety-related instructions in the system prompt should either be redundant with CLEAR (belt + suspenders) or handled inside the gate's training — otherwise the two systems can disagree.
- **Not a substitute for safety RLHF.** Best paired with, not instead of, a broadly safety-tuned base — CLEAR provides fine-grained per-input control on top of a baseline of safety behavior.

## Sources

- Paper: *CLEAR: Continuous Latent Adapter Routing for Utility-Preserving LLM Safety Alignment* — Wang, Jiang, Liao, Koyejo, 2026 — introduces the gate + safety-LoRA architecture.
- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021 — the LoRA baseline being conditioned.
- Paper: *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming* — Mazeika et al., 2024 — the safety eval used.
