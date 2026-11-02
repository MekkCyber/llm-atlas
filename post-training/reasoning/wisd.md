# WISD

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Weakening-Induced On-policy Self-Distillation. Build a *weaker* copy of a reasoning model by cheap, in-place operations — low-bit quantization or magnitude pruning — then train the original model on the log-probability contrast between the two. No labels, no verifiers, no stronger teacher. Reported average gains on math/code suites: +4.0 (quantization-weakening) and +5.5 (pruning-weakening).

**Prereqs:** [long-cot-rl](long-cot-rl.md), [../grpo](../grpo.md)
**Related:** [../rlvr](../rlvr.md), [../rejection-sampling](../rejection-sampling.md), [../../quantization/_number-formats](../../quantization/_number-formats.md)

---

## What it is

Standard distillation needs a *stronger* teacher. For frontier models that doesn't exist. WISD inverts the setup: cheaply degrade the current model into a weaker version of itself and use the gap between the two models' log-probs as a *label-free* on-policy reward signal.

The weakening operator must satisfy: (i) stay on the same loss manifold (so the weakened model still produces coherent CoT), (ii) degrade reasoning quality (so there is a gap to teach from), (iii) be cheap to apply at every rollout step. Low-bit quantization and magnitude pruning satisfy all three.

## How it works

Per RL step:

1. **Weaken.** Produce a weakened policy `π_weak` by quantizing (e.g. 4-bit weight-only) or pruning (unstructured magnitude mask) the current policy `π`. Both are deterministic given `π`.
2. **Roll out.** Sample trajectories under `π`.
3. **Score contrast.** For each rollout token, compute `log π(y|x) − log π_weak(y|x)`. Positive values are tokens where the full model is more confident than the weakened one — these are the *reasoning-critical* tokens.
4. **Reinforce.** Use the contrast as a per-token advantage in a GRPO-style update of `π`.

No ground truth, no verifier, no reference solutions needed. Compatible with the standard GRPO rollout loop.

## Why it matters

- Breaks the "no stronger teacher" wall for frontier reasoning models. Current post-training depends on verifiers (RLVR), human preferences (RLHF), or a stronger teacher model (distillation); WISD gives a fourth signal that scales with model capability instead of being bottlenecked by it.
- Weakening operators are already shipped in every quantization and pruning library. Operationally cheap: no new training infra, no reward model.
- Reframes quantization and pruning as *information probes*, not just deployment tools: the gap they open up is itself a measure of what matters for reasoning.

## Gotchas & tricks

- Weakening too hard produces incoherent outputs; the contrast then has no signal. The paper uses low-bit quantization (preserves token-level structure) and moderate magnitude pruning. Avoid 2-bit or >70% pruning.
- Reward can collapse if `π` and `π_weak` drift apart too fast; recompute `π_weak` from the live weights each update.
- Pruning gives larger gains than quantization in the paper but is more sensitive to the pruning ratio — tune the ratio per model size.
- Length is not controlled directly; combine with [length-penalty](length-penalty.md) if rollouts get verbose.
- Verifier-based RL still wins when a verifier exists and is cheap; WISD is for the regime where it doesn't.

## Sources

- Paper: *Weaken to Strengthen: Weakening-Induced On-Policy Self-Distillation for Large Reasoning Models* — Wu, Han, Cai, MIT / NVIDIA, 2026 — [arXiv:2610.38681](https://arxiv.org/abs/2610.38681).
- Related: *DeepSeek-R1* — DeepSeek, 2025 — RLVR-style reasoning RL that WISD aims to complement when verifiers are absent.
