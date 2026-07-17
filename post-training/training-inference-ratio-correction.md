# Training-Inference Ratio Correction
*Depth — closing the numerical gap between the inference engine that generates rollouts and the training kernel that updates on them.*

**TL;DR:** In a modern RL loop, rollouts are produced by a fused-kernel inference engine (vLLM-style, FP8/BF16, custom attention) and the RL loss is computed by a training kernel with different numerics (FP32 accumulate, framework attention, mixed precision). At small scale the divergence is noise; at trillion parameters it becomes a *systematic bias* — the training update is taken against a policy distribution that differs from the one the inference engine actually samples from. **Training-inference ratio correction** reconciles the two by multiplying the RL objective by the ratio of the two implementations' per-token probabilities. Ring-Zero (Inclusion AI, 2026) documents this as one of three load-bearing fixes for zero-RL at 1T.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md)
**Related:** [clipped-importance-sampling](clipped-importance-sampling.md), [../pre-training/fp8-training.md](../pre-training/fp8-training.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Two "policies" coexist in a scaled RL loop:

- **π_inf** — the distribution the inference engine actually samples from, using its own attention implementation, its own precision (often FP8/BF16), and its own KV-cache math.
- **π_train** — the distribution the training kernel would produce for the same input, using training-side attention, training-side precision (typically higher), and different rounding.

Nominally these are the same policy (same weights). Numerically they aren't — different attention kernels have different rounding, precision casts differ, and small per-token probability drift compounds across long sequences.

At small scale, the drift is dominated by other sources of noise. At 1T with long CoT and off-policy rollouts, the drift becomes a *systematic bias*: the RL loss is being optimized against π_train while the rollouts were drawn from π_inf.

**Training-inference ratio correction** treats this as a special importance-sampling problem: multiply per-token contributions by `π_train(a_t | s_t) / π_inf(a_t | s_t)` to correct for the mismatch.

## How it works

1. **Log the inference-time probabilities.** Have the inference engine record its per-token log-probs alongside each generated token. This is cheap — the engine already computes them.
2. **Recompute the training-time probabilities.** During the training forward pass, compute the same tokens' log-probs under the training kernel with the same weights.
3. **Form the ratio.** `r_t = π_train(a_t | s_t) / π_inf(a_t | s_t)`. In log space: `log r_t = logp_train - logp_inf`.
4. **Apply as a correction weight.** Multiply the per-token RL objective by `r_t` (or its clipped variant, when the ratio is unstable — see [clipped-importance-sampling](clipped-importance-sampling.md)).

This is layered on top of the standard on-vs-off-policy IS correction. Structurally similar, targets a different mismatch:

- Standard IS corrects for `π_new` vs `π_snapshot` (temporal drift).
- Training-inference ratio corrects for `π_train` vs `π_inf` (implementation drift).

## Why it matters

- **Removes a scale-emergent bias.** At small scale you don't need this; at 1T you do. Ring-Zero calls it out as required for stable convergence.
- **Cheap to implement.** Inference engines already compute per-token log-probs; storing them adds trivial memory. The training-side recompute happens on the forward pass regardless.
- **Composable with everything else.** Slots into the RL loss the same way any per-sample weight does.
- **Sets the ceiling on other numerics work.** Once this correction is applied, the *residual* mismatch is dominated by whatever precision + kernel differences remain — a well-scoped target for further hardening.

## Gotchas & tricks

- **Ratio can spike.** If the inference kernel is much lower precision than the training kernel, `r_t` can be large. Combine with clipping (see [clipped-importance-sampling](clipped-importance-sampling.md)) or truncated IS.
- **Log-prob storage adds bandwidth.** Long CoT generation × trillion-scale batches = non-trivial log-prob tensors. Compress (FP16 is enough for log-probs) or store in a separate high-bandwidth buffer.
- **Which precision is "right"?** Convention: the *training* kernel is treated as the reference. Rollouts get corrected toward it. Reversing this convention gives a different bias direction and typically worse convergence.
- **Interacts with speculative / draft-based inference.** If inference uses draft models or speculative decoding, the effective sampling distribution is more complex. Log-probs from the *final accepted tokens* are what matter for the correction.
- **Does not fix reward-model mismatch.** This correction is about *policy* sampling numerics, not about reward-model numerics. Reward-model precision should be handled separately.
- **Validate before scaling.** At 70B–200B the effect is measurable but small. Run a sanity ablation before committing engineering effort — if your training-inference gap is already small (e.g., both FP32), the correction is a no-op.

## Sources

- Paper: *Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning* — Cao, Liu, Zhan, Lan, Li, Yan, Peng, Dong, Zhang, Wang, Kong, Wen, Zhao, Zhang, Zhou, 2026 — [arXiv 2607.12395](https://arxiv.org/abs/2607.12395). Introduces training-inference ratio correction as part of the 1T zero-RL stability recipe.
