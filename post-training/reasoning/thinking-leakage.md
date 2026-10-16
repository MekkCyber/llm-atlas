# Thinking Leakage
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Hybrid-reasoning models (e.g., Qwen3 series) support a **think mode** and a **no-think mode**. When GRPO trains them in no-think mode — explicitly instructed to skip CoT — most of the accuracy gain comes from the model silently reactivating its think-mode circuit, not from learning a new no-think strategy. The paper shows this causally via weight-scaling, activation-subspace patching, and head-ablation experiments.

**Prereqs:** [../grpo.md](../grpo.md), [long-cot-rl.md](./long-cot-rl.md), [long2short.md](./long2short.md)
**Related:** [length-penalty.md](./length-penalty.md) · [../rlvr.md](../rlvr.md) · [../../interpretability/README.md](../../interpretability/README.md)

---

## What it is

A diagnostic finding, not a method. The paper exposes a specific interpretability failure in a popular RL-training setup for hybrid-reasoning models.

**Setup.** A hybrid-reasoning model exposes two modes:
- *Think mode* — emits an explicit `<think>...</think>` CoT, then an answer.
- *No-think mode* — emits only the answer.

GRPO training in no-think mode has been reported to improve no-think accuracy. The implicit interpretation: the model is "learning to reason implicitly."

**Finding.** The accuracy gain under no-think GRPO is not implicit reasoning — it is the model routing the forward pass through think-mode circuitry while suppressing the surface CoT.

## How it works

The paper runs three complementary interventions:

### 1. Weight-scaling

Decompose the GRPO-updated weights as $W_{\text{new}} = W_{\text{base}} + \Delta$. Scale $\Delta$ by $\alpha \in [0, 2]$ and measure accuracy. Finding: no-think accuracy tracks the magnitude of think-mode-correlated subspaces inside $\Delta$ — the "learned" delta is largely a think-mode activation delta.

### 2. Activation-subspace patching

Find a subspace of residual-stream activations that is maximally active in think mode and absent in no-think mode. Patch activations in this subspace between the two modes at inference time:
- Patch think → no-think: no-think accuracy *jumps up*.
- Patch no-think (post-GRPO) → base model's no-think residual: accuracy *drops*.

Both directions implicate the think-mode subspace as the mechanism of improvement.

### 3. Head ablation

Ablate attention heads that are known to be think-mode-selective. No-think accuracy under GRPO drops disproportionately — the heads carrying the "leaked" reasoning are the same heads the think mode uses.

### Mechanism via GRPO advantage

The paper links the effect to the GRPO advantage sign/scale. The policy improvement in no-think mode is funded by the same reasoning circuits the model would use in think mode — GRPO shapes the no-think policy toward think-circuit activations because those circuits correlate with correct answers.

## Why it matters

- **"No-think RL" has been pitched as a way to make models reason implicitly.** This paper is clear evidence against that interpretation.
- **Methodological warning.** Measuring no-think accuracy without controlling for mode-switching inside the forward pass is misleading. Any paper claiming "implicit reasoning" must now rule out leakage.
- **Deployment-relevant.** If no-think models secretly re-route through think-circuits, their inference cost is not actually cheaper (same compute, same cache pressure), and their latency gains may vanish under careful measurement.
- **Interpretability × post-training.** One of the first clean examples of a post-training behavior that is only visible under causal interp methods — not from benchmark numbers alone.

## Gotchas & tricks

- **Specific to hybrid-reasoning models.** Models without an explicit two-mode distinction don't have this leakage structure.
- **Qwen3 1.7B / 4B / 8B.** The results replicate across three sizes; whether they hold on larger models (14B+) is open.
- **The intervention stack is portable.** Weight-scaling + subspace patching + head ablation is a reusable diagnostic recipe for any "the model learned X implicitly" claim.
- **Prevention is unclear.** The paper diagnoses leakage; mitigations (e.g., aggressive think-circuit ablation during no-think training, or anti-think regularizers) remain for follow-up work.
- **Related to [long2short](./long2short.md).** Both target "reason more with less output"; thinking-leakage is one explanation for why long2short-style compression can look stronger than it is.

## Sources

- Paper: *The Thinking Leakage in Reinforcement Learning for LLMs* — Liu, An, Aliev, Li, Yang, Ghanem (KAUST / SJTU), 2026 — [arXiv:2610.14270](https://arxiv.org/abs/2610.14270). Replicates across Qwen3 1.7B, 4B, 8B.
