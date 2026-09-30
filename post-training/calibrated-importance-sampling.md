# Calibrated Importance Sampling (CIS)
*Depth — a confidence-aware truncation of the training-inference importance ratio for LLM RLVR.*

**TL;DR:** In LLM RLVR pipelines, rollouts come from an inference engine while gradients come from a training engine, and the two assign different token probabilities. Importance sampling corrects for this but blows up variance when the ratio is large. CIS models the mismatch as an additive log-odds displacement (empirically invariant to token confidence) and truncates *that* with a single constant, which maps back to an importance-ratio cap that tightens as token confidence rises.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md), [ppo](ppo.md)
**Related:** [training-inference-mismatch](training-inference-mismatch.md), [_rl](_rl.md)

---

## What it is

A drop-in replacement for the plain importance ratio (or PPO/GRPO clip term) used to correct off-policy LLM updates when generation and gradient computation happen in different numerical environments. Instead of clipping the ratio itself, CIS clips the *pre-softmax logit displacement* that causes the ratio to be off.

## How it works

- **Empirical characterization.** Log the per-token logit at both engines. The training-inference discrepancy is well-fit by an additive displacement in log-odds whose distribution is approximately invariant to token confidence — the same magnitude of displacement whether the token was near-certain or near-uniform.
- **Confidence-aware truncation.** Cap the positive log-odds displacement at a single scalar `τ`. Because `IS = softmax(logits) / softmax(logits + δ)`, a fixed cap on `δ` maps to an importance-ratio cap that *tightens* for high-confidence tokens (where a small `δ` produces a large ratio) and *loosens* for low-confidence tokens.
- **Where it plugs in.** Replace `IS(a|s)` (or the min/clip pair in PPO/GRPO) with the CIS-corrected estimator. Nothing else in the update changes.

## Why it matters

The training-inference numeric gap has been the quiet source of instability in every serious LLM-RL system (RLHF, GRPO, RLVR). Prior fixes — clipping the ratio, matching temperatures, forcing bit-identical kernels — either over-correct confident tokens or fail on MoE stacks. CIS gives a principled, single-hyperparameter correction with a bounded second moment (trading unbounded IS variance for a small controlled bias). Reported result: top five-benchmark average on all three MoE models tested, across five math-reasoning benchmarks.

## Gotchas & tricks

- **MoE amplifies the gap.** Routing decisions can flip between engines, producing large `δ` on the affected tokens. CIS handles this by construction, but you should also verify router determinism when possible.
- **`τ` is the only knob.** Small `τ` → high bias, low variance; large `τ` → low bias, high variance. The paper reports the confidence-invariance holds empirically, so `τ` transfers across model sizes reasonably well — but validate on your specific engine pair.
- **Don't stack.** CIS already implements a cap; adding a PPO-style ratio clip on top double-truncates and biases updates further.
- CIS is a *variance* fix, not a *reward* fix. If your RM is miscalibrated, CIS won't save you — see [_rewards](_rewards.md).

## Sources

- Paper: Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where It Arises and How to Correct It — Yu, Zhao, Li, Yang, Jenkins, Zhang, Killian, 2026 — [arXiv:2609.32444](https://arxiv.org/abs/2609.32444)
