# MIPU — Monotonic Inference Policy Update
*Depth — an LLM-RL objective that targets the deployed inference policy, not the training-engine policy.*

**TL;DR:** LLM RL uses two separate engines — an inference stack (vLLM, SGLang) that generates rollouts and a training stack (Megatron, FSDP) that computes gradients. Even with `load_weights` synchronized, kernel differences produce inconsistent token probabilities. Prior work treats this as a plain off-policy nuisance; MIPU reframes it as a *wrong-objective* problem. What we actually want to improve is the **inference** policy — the one that runs in production. MIPU builds candidate updates and selectively accepts them using an inference-side gap proxy, so every accepted step provably improves the deployed policy under a training-inference mismatch.

**Prereqs:** [ppo](ppo.md), [grpo](grpo.md), [_rl](_rl.md)
**Related:** [rlvr](rlvr.md), [../systems/training-inference-mismatch.md](../systems/training-inference-mismatch.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

LLM RL frameworks generate rollouts on an inference engine and compute gradients on a training engine. In practice, even with the *same* parameters loaded on both sides, the two engines assign slightly different log-probabilities to the same token sequences — different kernel implementations, different FP8/BF16 rounding, different attention layouts. Every rollout is therefore already off-policy with respect to the training engine, and every training step is off-policy with respect to the inference engine.

Prior fixes (importance sampling, forced re-scoring, partial-rollout masking) stabilize the training-engine policy under the mismatch. MIPU points out that this is optimizing the wrong thing: a gradient step that helps the training-engine policy does not necessarily help the inference-engine policy, and the inference-engine policy is the one that ships. MIPU introduces the *Monotonic Inference Policy Improvement* (MIPI) objective and a two-step accept/reject algorithm that enforces it.

## How it works

Let $\pi_\theta^{train}$ and $\pi_\theta^{inf}$ be the training-engine and inference-engine policies at the same parameters $\theta$. Under mismatch, $\pi_\theta^{train} \neq \pi_\theta^{inf}$ for the same $\theta$.

**MIPI objective.** Each update $\theta \to \theta'$ is valuable only if the **inference-engine** value function improves:

$$
J_{inf}(\theta') \geq J_{inf}(\theta),
\quad J_{inf}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta^{inf}}[R(\tau)]
$$

not merely if $J_{train}(\theta') \geq J_{train}(\theta)$.

**MIPU two-step.**

1. **Sampler-referenced candidate update.** Rollouts came from $\pi_\theta^{inf}$, so treat the inference engine as the reference distribution when computing the policy-gradient update. This produces a *candidate* $\theta'$ — no acceptance yet.
2. **Inference-side gap proxy.** Before committing $\theta'$, evaluate an inference-side gap proxy: an approximation of $J_{inf}(\theta') - J_{inf}(\theta)$ that uses the inference engine's log-probs on a small held-out batch. If the proxy is above threshold, accept $\theta'$; otherwise reject and shrink the step (or discard).

The check resembles PPO's clip step, but instead of a training-side trust region, the trust region is measured on the deployed policy. Rejection is not just a stability trick — it's the mechanism that keeps the objective monotonic.

## Why it matters

- **Reframes what "correct RL" means in production.** The deployed policy is now a first-class training target; the training engine is a means. This is the first policy-side attack on train-inference mismatch (prior work is either infra-side masking or importance-weighting).
- **Compatible with existing algorithms.** MIPU is a wrapper: rollouts are still PPO/GRPO/RLVR-shaped, gradients still flow through the training engine. Only the acceptance criterion changes.
- **Empirically stabilizes RL under high mismatch.** At two model scales with deliberately amplified mismatch, MIPU improves average reasoning performance and training stability over baselines.
- **A general principle beyond LLMs.** Any system with a *proxy-vs-deployed* policy split — safety filters, distilled deployment models, quantized serving — has the same misalignment. The pattern generalizes.

## Gotchas & tricks

- **Cost of the inference-side proxy.** Every candidate needs an inference-engine forward pass. Batch the proxy over accumulated candidates or compute it only every $k$ steps.
- **Rejection rate as a health signal.** If MIPU rejects most steps, the training and inference engines have drifted too far — a signal to re-sync weights or re-check kernel parity.
- **Threshold calibration.** The accept/reject threshold on the proxy is the key hyperparameter. Too strict → training stalls; too loose → the acceptance test degenerates into "always accept" and MIPU reduces to vanilla off-policy PPO.
- **Interacts with partial rollouts.** [partial-rollouts](../systems/partial-rollouts.md) mask stale segments from the gradient; MIPU adds an orthogonal filter on the *whole* step. They compose cleanly.

## Sources

- Paper: *The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning* — Liang et al., Tianjin U. / Alibaba, 2026 — [arXiv:2606.29526](https://arxiv.org/abs/2606.29526)
