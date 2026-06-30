# Diffusion-GRPO
*Depth — Group Relative Policy Optimization adapted to flow-matching diffusion policies.*

**TL;DR:** GRPO ported from token-level LLM policies to **velocity-field policies** in flow matching / rectified flow. Per prompt, sample a group of `G` full sampling trajectories, score each with a composite reward, and update the velocity-field policy using the GRPO advantage (group-normalized reward), with PPO-style ratio clipping applied **per diffusion timestep** and a KL anchor to the supervised teacher.

**Prereqs:** [grpo](../post-training/grpo.md), [_rl](../post-training/_rl.md), [_rewards](../post-training/_rewards.md)
**Related:** [dpo](../post-training/dpo.md)

---

## What it is

A drop-in RL post-training algorithm for diffusion / flow-matching generators. It addresses the structural mismatch between LLM-style RL (per-token policy, per-token value) and diffusion (continuous velocity field over a multi-step trajectory) without abandoning the GRPO simplifications that made it the modern default for reasoning RL.

## How it works

**Trajectory rollouts.** For a prompt `p`, sample `G` complete sampling trajectories under the current policy `v_θ`. Each trajectory `τ_g = (x_T, x_{T-1}, …, x_0)` produces a final image, which gets a composite scalar reward `R(τ_g)` (see [_rewards](../post-training/_rewards.md)).

**Group-normalized advantage.**

```
A_g = (R(τ_g) - mean(R)) / std(R)        # over the G rollouts for this prompt
```

No learned value model. Just like LLM GRPO, this avoids the brittle value-network step that PPO requires.

**Policy gradient on the velocity field.** Express the trajectory log-likelihood under `v_θ` as a sum over diffusion timesteps; the policy gradient becomes a sum of per-timestep terms. The GRPO update per timestep `t`:

```
L_t = -min( r_t · A_g,  clip(r_t, 1-ε, 1+ε) · A_g )    # per-step PPO ratio
r_t = exp(log π_θ(x_{t-1}|x_t) - log π_θ_old(x_{t-1}|x_t))
```

Summing over `t` gives the full trajectory loss; averaging over the group gives the prompt loss.

**KL anchor.** Add `β · KL(π_θ ‖ π_ref)` against the supervised teacher to prevent drift — same role as in LLM RLHF/GRPO.

## Why it matters

- **No value model**, unlike PPO; tractable at image scale where a per-step value would be its own modeling problem.
- **Uses composite reward directly**, unlike DPO, which collapses the reward into a pairwise preference and ignores the multi-step trajectory structure.
- **Matches the per-step structure of flow matching** — the PPO ratio is applied where the model actually predicts (each velocity step), not at the final-image level.
- **Same recipe as LLM-RLHF.** Teams that already operate GRPO infra for LLMs can reuse most of the orchestration (group rollouts, group-normalized advantages, KL anchor) for diffusion.

## Gotchas & tricks

- **Velocity-norm inflation.** RL consistently inflates `‖v_θ‖` by 5–15% across diffusion RL methods; pair with a training-time hinge penalty on `‖v_θ‖ > ‖v_ref‖` to suppress it (covered separately in the 2026-06-29 KG update).
- **Per-timestep clipping**, not just per-trajectory clipping. A noisy single timestep can dominate the gradient if you clip only at the trajectory level.
- **Sampling cost.** Each prompt needs `G` full trajectories — sampling is the dominant cost. Use short schedules (few-step samplers) during RL, even if the deployed model uses more steps.
- **Reward over-weighting.** Composite rewards with one dominant component cause GRPO to over-optimize that component; tune weights so group normalization is meaningful across components.

## Sources

- Paper: *Qwen-Image-2.0-RL Technical Report* — arXiv:2606.27608 — https://arxiv.org/abs/2606.27608
- See also: [qwen-image-2 case study](../case-studies/qwen-image-2.md) for the full pipeline context.
