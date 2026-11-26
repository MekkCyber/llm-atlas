# iTDM: Instantaneous Trajectory Distribution Matching
*Depth — diffusion RL with a behavior-velocity baseline that stays well-centered under CFG, off-policy data, and imperfect self-consistency.*

**TL;DR:** Forward-diffusion RL methods (DGPO, DiffusionNFT, AWM) can be shown to reduce to reward-weighted regression centered on the *policy* velocity. That centering breaks under classifier-free guidance rollouts, off-policy updates, or imperfect self-consistency, which destabilizes training. iTDM trains an auxiliary denoiser to track the *behavior* velocity and subtracts it as a baseline, keeping the advantage estimator correctly centered. Scales to the 5.6B FLUX.3 Klein Base model with ~3× faster convergence on GenEval2.

**Prereqs:** [ppo.md](ppo.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [rlvr.md](rlvr.md) · [_rewards.md](_rewards.md)

---

## What it is

Reinforcement learning for diffusion models trains the denoiser to generate samples that score well under some reward. The policy-gradient form of this requires a baseline to keep variance manageable. Prior methods implicitly use the current policy's velocity (the denoiser's current output) as the baseline, which is correct only when rollouts are on-policy and self-consistent. iTDM replaces that implicit baseline with an *explicit* auxiliary denoiser trained to model the behavior policy's velocity — the distribution that rollouts actually come from.

## How it works

Let `v_θ(x_t, t)` be the main denoiser's velocity (the parameters being trained) and `v_φ(x_t, t)` an auxiliary "behavior" denoiser. The behavior denoiser is trained online to match the velocity of the rollout distribution (which may differ from `v_θ` because of CFG, off-policy replay, or stale self-consistency). The reward-weighted update to `v_θ` is centered on `v_φ`:

```
loss_θ  =  E_{x_t ~ rollouts} [ w(x_t) · || v_θ(x_t, t) - v_target ||² ]
loss_φ  =  E_{x_t ~ rollouts} [ || v_φ(x_t, t) - v_rollout(x_t, t) ||² ]
```

where `w(x_t)` is the reward weight (positive for high-reward trajectories) and `v_target` is defined relative to `v_φ`, not `v_θ`. Updating `v_φ` to track the rollout distribution restores the "match the behavior baseline" property that the variance-reduction argument needs, even when CFG and off-policy data are in play.

## Why it matters

- **Role analogous to PPO's value function.** iTDM is to diffusion RL what the value network is to language-model PPO: a baseline that keeps the advantage estimator well-centered.
- **Robust under CFG rollouts.** Classifier-free guidance at inference pushes rollouts off the policy's velocity manifold; iTDM's behavior-tracked baseline absorbs the drift.
- **Scales.** Results on FLUX.3 Klein Base (5.6B) show the method is not a toy. ~3× faster convergence on GenEval2 vs. DiffusionNFT, and SOTA across compositional generation, text rendering, and preference alignment.
- **Clean theoretical story.** The paper derives the implicit-baseline failure mode of prior methods and the explicit-baseline fix from the same loss form — a tight unification.

## Gotchas & tricks

- **Auxiliary denoiser cost.** `v_φ` is a second forward pass per rollout. If `v_φ` is the same size as `v_θ`, you're doubling RL compute. The paper uses a smaller `v_φ` with shared early blocks; this is where most of the practical savings come from.
- **Staleness of `v_φ`.** The auxiliary denoiser is trained online but lags the rollout distribution by one step. Too aggressive LR on `v_φ` causes oscillation; too slow and the baseline misses drift.
- **CFG scale matters for the fix.** Small CFG values barely change the velocity; large CFG values create large baseline mismatches. The fix is most load-bearing at the CFG scales used in production image generation (5–10).
- **Not a drop-in for text diffusion.** The derivation assumes continuous-latent diffusion; discrete diffusion (text-latent, token diffusion) needs re-derivation.
- **Reward-weighted regression analogy generalizes.** Any RL method that reduces to RWR on the policy velocity is susceptible. iTDM's recipe (track behavior with an auxiliary head, subtract) applies more broadly than the paper's immediate targets.

## Sources

- Paper: *Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching* — Luo (HKUST), 2026 — [arXiv:2611.30986](https://arxiv.org/abs/2611.30986).
