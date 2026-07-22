# Group Entropy-Controlled Policy Optimization (GEPO)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** GEPO is a lightweight extension to [GRPO](grpo.md) that fixes a subtle bias when RL is run on heterogeneous task mixtures (math + code + IF + science). GRPO's group-mean/std advantage normalization becomes *entropy-dependent* across such mixtures, making advantages statistically non-comparable between prompt groups. GEPO fixes this by attenuating each group's advantages asymmetrically based on that group's rollout entropy: positive advantages are damped in low-entropy groups (avoid over-exploitation), negative advantages are damped in high-entropy groups (preserve exploration), with adaptive thresholds from historical entropy statistics.

**Prereqs:** [grpo.md](./grpo.md), [ppo.md](./ppo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md) · [long-cot-rl.md](./reasoning/long-cot-rl.md) · [_rewards.md](./_rewards.md)

---

## What it is

A drop-in modification to GRPO's advantage estimator that makes advantages comparable across heterogeneous task mixtures. Nothing else about the RL loop changes: same clipped ratio, same KL to reference, same group-relative baseline. Only the *sign-conditioned scaling* of $A_i$ is new.

## How it works

For each group of $G$ rollouts from the same prompt, GEPO:

1. Computes the standard GRPO advantage: $A_i = (r_i - \bar r) / \sigma_r$.
2. Estimates the group's rollout entropy $H_g$ from the sampled tokens (available for free from the policy's own logits).
3. Compares $H_g$ against a running percentile band $[\tau_\text{lo}, \tau_\text{hi}]$ derived from historical group entropies seen during training.
4. Applies an **asymmetric multiplicative scale** $s^+, s^-$:
   - If $H_g$ is in the low-entropy tail: shrink positive advantages ($s^+ < 1$) — the policy is already confident here, don't push exploitation further.
   - If $H_g$ is in the high-entropy tail: shrink negative advantages ($s^- < 1$) — the policy is still exploring, don't punish rollouts that just happened to underperform.
5. Feeds the shaped advantage into the usual PPO-clip objective.

The thresholds $\tau$ adapt over training as more entropy data accumulates, so early-training exploration isn't punished by late-training statistics.

## Why it matters

Multi-domain RL is the default recipe post-DeepSeek-R1, and everyone hits the same problem: math prompts and code prompts induce very different rollout entropies under the same policy, and GRPO's z-score normalization silently amplifies that gap. Global or per-token entropy controls (entropy bonuses, KL penalties) are too coarse. GEPO's per-group asymmetric shaping is targeted at exactly the bias GRPO introduces, and the paper reports consistent wins over both baseline GRPO and prior entropy-controlled methods across 13 benchmarks and two base models.

## Gotchas & tricks

- The historical-percentile thresholds need enough warmup that they stabilize. Early training may still see raw GRPO behaviour.
- The paper's ablations suggest asymmetry is the load-bearing part — symmetric shrinking (scale both signs equally) recovers less of the win.
- Because GEPO leaves everything else identical, it composes cleanly with RLVR-style verifier rewards; the paper's evaluation is all RLVR.

## Sources

- Paper: *Group Entropy-Controlled Policy Optimization* — Guangran Cheng, Chengqi Lyu, Songyang Gao, … Wenwei Zhang, Kai Chen (Shanghai AI Laboratory), 2026 — [arXiv:2607.16850](https://arxiv.org/abs/2607.16850) · [HF](https://huggingface.co/papers/2607.16850)
