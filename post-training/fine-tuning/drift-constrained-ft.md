# Drift-Constrained Fine-Tuning
*Depth — instruct-model fine-tuning with a pre-declared behavioral-drift budget and direction-only updates.*

**TL;DR:** Uncontrolled SFT of instruct models improves the target task while silently degrading other capabilities. Drift-constrained fine-tuning (Yuan et al., 2026) reframes SFT as picking a **direction** inside a **pre-declared drift budget** relative to the reference model. Selective, layer-targeted updates advance the target task while staying inside the budget, preserving reasoning and general-capability benchmarks across Qwen3 scales and 100+ languages.

**Prereqs:** [../dpo.md](../dpo.md), [README](README.md)
**Related:** [merged-lora](merged-lora.md), [../_post-training.md](../_post-training.md)

---

## What it is

A constrained-optimization view of instruct-model SFT. Instead of "minimize target-task loss and hope for the best", the method commits *upfront* to a maximum behavioral drift from the reference (base instruct) model, then chooses the update direction that maximizes target-task gain inside that budget. The budget is defined on model outputs (behavioral drift), not on weights.

## How it works

1. **Fix a reference model** `M_0` (the base instruct model to be tuned).
2. **Declare a drift budget** `ε` — the maximum tolerated divergence of the tuned model's outputs from `M_0` on a held-out capability probe (typically a mix of reasoning, general knowledge, and multilingual prompts). This is a **behavioral** budget: `KL(M_θ ‖ M_0)` on the probe, or a preference-model score on it.
3. **Direction selection.** Parameterize updates as a per-layer scaling of gradient direction; project out any component that would push the model outside `ε` on the capability probe.
4. **Layer-targeted updates.** In practice, most direction is spent on a small number of layers; the paper's ablations show restricting updates to the identified layers preserves capability better than uniform LoRA.
5. **Optimize.** Standard target-task loss, subject to the budget-projected direction.

## Why it matters

- **Predictable degradation.** The budget is set before training, so the operator knows the worst-case drift on general capability.
- **Retains reasoning.** The paper reports better reasoning and general-capability retention than unconstrained SFT across Qwen3 model sizes.
- **Multilingual retention.** Across 100+ languages, drift-constrained tuning holds prior performance while adding the new task.
- **Composable with LoRA.** The projection can wrap around a LoRA update; drift-constrained LoRA is a strict improvement on plain LoRA when capability preservation matters.

## Gotchas & tricks

- **The probe defines the budget.** If your probe is narrow, you'll be surprised by drift on axes the probe doesn't cover. Use a diverse probe (reasoning, code, multilingual, instruction following).
- **Budget-projection cost.** Computing the projection each step is expensive if the probe is large; the paper uses a small held-out subset per step and a full evaluation between epochs.
- **Direction ≠ magnitude.** The framework says only direction matters, but the effective update magnitude still comes from the learning rate. Overshooting inside the budget can still cause quality issues on the target task.
- **Layer selection is a hyperparameter.** The paper's identified target layers vary by model family; don't reuse them across families without re-selecting.
- **Not a substitute for RLHF/DPO.** Drift-constrained SFT preserves the *reference*; if the reference is under-aligned, DFO won't fix that.

## Sources

- Paper: *Drift-Constrained Optimization: Only Direction Matters in Fine-Tuning Instruct Models* — Yuan, Gao, Tu, Liu, Huang, Qiao — 2026 — [arXiv:2609.13680](https://arxiv.org/abs/2609.13680).
