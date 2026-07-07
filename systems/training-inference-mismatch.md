# Training-Inference Mismatch
*Depth — the two-engine problem in modern LLM RL and the family of fixes for it.*

**TL;DR:** Modern LLM RL uses one engine for rollouts (vLLM, SGLang) and another for gradients (Megatron, FSDP). Even with `load_weights` after every step, the two engines disagree on token log-probs for the same trajectory — different kernel implementations, different attention layouts, different precision policies. This turns every rollout into an off-policy sample against the training engine, and every gradient step into an off-policy update against the deployed policy. Fixes come from four angles: infra-side (masking, resync), policy-side (importance weighting, MIPU), objective-side (target the inference policy directly), and reference-side (train and infer with matched kernels).

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/ppo.md](../post-training/ppo.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [partial-rollouts](partial-rollouts.md), [../post-training/mipu.md](../post-training/mipu.md), [ray](ray.md)

---

## What it is

Any production LLM RL system runs two model implementations at once:

- **Inference engine.** vLLM / SGLang / TRT-LLM. Optimized for high-throughput autoregressive decoding: paged KV cache, continuous batching, custom attention kernels, aggressive quantization (FP8, INT8), speculative decoding.
- **Training engine.** Megatron / DeepSpeed / FSDP. Optimized for gradient computation: tensor / pipeline / expert parallelism, activation checkpointing, higher precision on sensitive ops, no paged cache.

Both hold the same parameters $\theta$. When you feed the same token sequence to each, they should produce identical log-probs. They do not. Root causes:

1. **Kernel drift.** FlashAttention-2 vs. paged FlashAttention have different numerics under FP16/BF16 reduction order.
2. **Precision drift.** Inference engines often run in FP8 or FP16 on sensitive matmuls; training runs BF16 or FP32.
3. **Layout drift.** Fused vs. unfused RMSNorm, GQA head sharding vs. replication, different rope implementations.
4. **Sampling drift.** Rollouts pass through a top-p / top-k sampler in the inference engine but the training engine sees only the resulting tokens; slight numerics in the pre-sampler logits mean the rollout distribution is *never* exactly $\pi_\theta^{train}$.

## The four families of fix

### 1. Infra-side masking

Accept the mismatch; mask off the worst-affected tokens from the gradient. [partial-rollouts](partial-rollouts.md) is the canonical example — segments from stale iterations don't contribute gradient even though they contribute to reward. Cheap; doesn't fix the objective.

### 2. Policy-side importance weighting

Re-weight training samples by $\pi_\theta^{train}(a|s) / \pi_\theta^{inf}(a|s)$ so gradient estimation is unbiased under the mismatch. Standard PPO ratio clipping already does part of this — the mismatch just widens the ratio distribution. Requires computing both engines' log-probs on the same trajectory, which doubles compute for the accepted rollouts.

### 3. Objective-side

Change what "improvement" means. [../post-training/mipu.md](../post-training/mipu.md) redefines the RL objective to be *inference-engine* improvement, and gates each candidate update on an inference-side gap proxy. Attacks the root — the target policy is the deployed one — but adds an acceptance check per step.

### 4. Reference-side (kernel parity)

Make the two engines numerically identical. Force the training engine to use the inference engine's attention kernel, or vice versa. Expensive to maintain (custom kernels drift out of sync with framework updates) and often defeats the point of using specialized engines. Some teams do this partially: use the same RMSNorm implementation, same rope, same GQA layout — but not the same attention.

## Why it matters

- **Universally present.** Every serious LLM RL stack hits this. It's not a research artifact.
- **Silent.** Reward curves can look healthy while the deployed model regresses because the training-engine metric is not the deployment metric.
- **Grows with scale.** Bigger models are more sensitive to numerics; longer contexts amplify kernel drift; quantized deployment widens the gap further.
- **Interacts with everything.** Speculative decoding, chunked prefill, KV-cache reuse — every serving optimization adds another drift source between rollout and gradient.

## Gotchas & tricks

- **Sanity check by re-scoring.** After each rollout, re-score the trajectory with the training engine and compute the log-prob delta. A sudden jump in this delta is your early-warning signal.
- **Ratio clip is not enough.** PPO's ratio clip was designed for on-policy drift within a training run, not for the ~1e-2 systematic drift between two engines. Tighten the clip range under known mismatch or you eat variance.
- **Weight sync frequency doesn't fix it.** Even syncing every step, the two engines produce different log-probs at the *same* $\theta$. Sync fixes staleness, not kernel drift.
- **Quantized deployment is the worst case.** If you train BF16 and deploy FP8/INT8, the deployed policy can be materially different from the training-engine policy at exactly the same $\theta$. Objective-side fixes are the only clean answer here.
- **Kernel parity is a moving target.** Frameworks update, kernels evolve, precisions change. Any "we made them numerically equal" story goes stale within a release cycle.

## Sources

- Paper (objective-side framing): *The Mirage of Optimizing Training Policies* — Liang et al., 2026 — [arXiv:2606.29526](https://arxiv.org/abs/2606.29526)
- Paper (infra-side): *Kimi k1.5* — Moonshot AI, 2025 (Sec. 2.6.2 — partial rollouts as an infra-side response)
- Practitioner posts: OpenRLHF, veRL issue trackers document the mismatch empirically.
