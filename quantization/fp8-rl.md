# Full-Pipeline FP8 Reinforcement Learning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Running both the actor's forward/backward pass *and* the rollout worker in FP8 is 1.5–2× faster than BF16, but naively-quantized FP8 RL diverges mid-training with entropy surges and garbled outputs. The root cause is that compounded FP8 quantization noise distorts the PPO/GRPO importance ratio and pushes high-negative-advantage tokens outside the trust region, where their gradients get clipped to zero — so pathological outputs never get penalized. **Calibrated Clipping** rebalances the clipping quantiles between FP8 and BF16, restoring stability at BF16-parity quality.

**Prereqs:** [fp8.md](./fp8.md), [../pre-training/fp8-training.md](../pre-training/fp8-training.md), [../post-training/ppo.md](../post-training/ppo.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [_number-formats.md](./_number-formats.md) · [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

FP8 pre-training landed as a production regime (DeepSeek-V3, Llama 4, others). The natural next step — run RL post-training in FP8 too — has been blocked by two failure modes:

1. **Train-inference mismatch.** The rollout worker (inference-style) and the actor (training-style) quantize differently, and their logprobs drift apart. Prior work addressed this with importance-sampling corrections (TIS-style).
2. **Compounded quantization noise.** *This* paper's contribution: even after fixing (1), full-pipeline FP8 RL still diverges — but from an *asymmetric* clipping failure, not from a mismatch.

## How it works

### The failure mechanism

PPO/GRPO clip the importance ratio $r_t = \pi_\theta(y_t)/\pi_{\theta_{\text{old}}}(y_t)$ to $[1-\epsilon, 1+\epsilon]$ (typically $\epsilon = 0.2$). Under BF16 the clipping quantiles are approximately symmetric around 1.

Under end-to-end FP8, quantization noise perturbs $\pi_\theta$'s logits so the distribution of $r_t$ becomes *skewed*: the lower tail (where $\pi_\theta$ underestimates a token's probability) stretches out much further than the upper tail. Tokens with **negative advantage** (bad tokens the policy should learn to *avoid*) often land in that lower tail. The clip zeros their gradient — the policy never gets the "don't do this" signal — and the bad behavior accumulates. Observationally: mid-training entropy surges, then garbled outputs.

### Calibrated Clipping

The proposed fix aligns the FP8 clipping bounds with the BF16 distribution of $r_t$ *dynamically*:

1. On a small BF16 reference batch, measure the quantile of $r_t$ that corresponds to the lower clipping bound.
2. Set the FP8 lower bound to match that same quantile of the FP8 $r_t$ distribution.
3. Rebalance the upper bound to preserve the intended overall clip fraction.

The scheme is dynamic (re-calibrates during training as the distribution shifts) and drop-in for GRPO or DAPO.

## Why it matters

- **1.5–2× RL throughput.** Doubles effective RL compute on Hopper/Blackwell hardware.
- **Unlocks FP8 for the full training→post-training pipeline** — the last regime where FP8 was still risky.
- **Diagnostic value:** frames the FP8 failure as an *importance-ratio distortion* problem, not just a numerical-precision problem. Suggests other RL-quantization failures may have similar asymmetric-clipping roots.

## Gotchas & tricks

- **Symptom to watch for.** A slow entropy climb followed by output degradation, without an obvious loss spike, is the tell. If you only monitor loss you can miss it until responses go visibly bad.
- **TIS-style corrections are still necessary** for train-inference mismatch; Calibrated Clipping composes with them, doesn't replace them.
- **Granularity of FP8 scaling matters.** The paper shows the fix works across multiple FP8 scaling granularities (per-tensor, per-channel, per-tile), but tighter tiles reduce the underlying noise so the calibration correction is smaller.
- **8B–32B scale validated.** Not yet reported for 100B+ dense models or MoE; the compounded-noise story is likely to hit MoE routers hard.
- **Reward-model scoring** in the RL loop can stay in higher precision — the paper's fix is about the actor's clipping behavior, not the reward path.

## Sources

- Paper: *Towards Full Pipeline FP8 Reinforcement Learning for LLMs* — Chen et al., ByteDance Seed / UW-Madison, 2026 — introduces Calibrated Clipping, validates on GRPO and DAPO at 8B–32B. [arXiv:2609.22870](https://arxiv.org/abs/2609.22870)
- Paper: *DeepSeek-V3 Technical Report* — FP8 training recipe (tile scaling, forced FP32 accumulation) that this paper extends into RL.
- Related: TIS-style train-inference correction methods for RL fine-tuning under mixed precision.
