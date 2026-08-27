# On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of maximizing a scalar reward directly, convert the reward signal into **explicit intermediate targets** that the trainable policy fits via distillation, while a **frozen behavior policy** supplies query states and gets refreshed via **exponential moving average**. The recipe cleanly decouples *target construction* (how good a supervision signal we build from the reward) from *fitting* (how well the policy realizes it), letting each be analyzed and debugged separately. Formalized for diffusion alignment by Zhou et al. 2026 as **DiffusionOPSD**; the same skeleton is applicable to LLMs and any policy where per-step supervision would be useful but the reward is a single scalar at the end.

**Prereqs:** [_rl.md](_rl.md), [_post-training.md](_post-training.md)
**Related:** [dpo.md](dpo.md), [rlvr.md](rlvr.md), [rejection-sampling.md](rejection-sampling.md), [cot-reward-model.md](cot-reward-model.md)

---

## What it is

Reward-driven post-training has two flavors:

- **Policy gradient** (PPO, GRPO, RLVR) — reward gradients update the policy directly, per rollout.
- **Distillation** (SFT-from-teacher, DPO-style preference learning) — build target outputs from a teacher or preference signal, minimize a supervised loss to those targets.

Policy gradient is simple but debugging is hard: a bad update could mean bad reward, bad advantage estimate, bad policy step, or any combination. Distillation is easy to debug but caps at the teacher.

**On-policy distillation splits the difference.** A frozen behavior policy rolls out. Reward gradients construct **explicit positive/negative targets** for the *intermediate* states along those rollouts. The trainable policy fits those targets via a supervised loss. Periodically an EMA update refreshes the behavior policy toward the trainable one.

## How it works

### The outer loop

```
freeze behavior_policy ← policy
loop:
    trajectories = behavior_policy.rollout(prompts)
    query_states, anchors = extract(trajectories)
    for anchor a in anchors:
        gradient_of_reward = ∇_x R(a)
        pos_target = clip(a + η · +gradient_of_reward)
        neg_target = clip(a - η · +gradient_of_reward)
    for a few inner steps:
        loss = fit(policy(query_state), pos_target) + repel(policy(query_state), neg_target)
        policy.update(loss)
    behavior_policy = EMA(behavior_policy, policy)
```

Three things separate this from vanilla RL:

1. **Targets are constructed and *detached*.** They are not `stop_grad(policy(x)) + gradient`; they are built from reward gradients, then supervision is against those fixed targets.
2. **Inner-loop is *supervised*, not RL.** A finite number of fitting steps, no policy-gradient variance.
3. **EMA-refreshed behavior policy.** The trajectory distribution slowly follows the trainable policy — the whole thing stays on-policy without needing importance corrections.

### Why the decoupling matters

Once targets are decoupled from fitting, you can measure them separately:

- **Target-construction quality** — do the constructed positive targets actually earn higher reward than the anchors, in a same-query controlled comparison?
- **Realization quality** — after a fixed number of fitting steps, how close is the policy to the target?

Zhou et al. observe that *larger target-construction gains don't necessarily translate to larger realized gains after one fitting update* — a diagnostic that vanilla policy gradient hides inside the update.

## Why it matters

- **Diffusion alignment.** Endpoint reward tells you nothing about *how the intermediate denoising prediction should change*. Building explicit clean-output targets at sampled query timesteps fixes that. DiffusionOPSD hits SOTA on 19/20 reward-matched settings and cuts training GPU-hours 40–63% versus DiffusionNFT.
- **LLM alignment analogue.** The same skeleton — build detached per-token targets from reward gradients, fit via supervised loss, EMA-refresh — is a natural way to *dense-supervise* any RLHF/RLVR pipeline. Related to OPDVR (which turns sampled-token OPD into an RLVR method) and to SecOPD (token-level distillation against a clean-input teacher).
- **Diagnosable failure modes.** Bad alignment can be traced to bad targets, bad fitting, or both — a lever the field has been missing.

## Gotchas & tricks

- **Target clipping is load-bearing.** Bounded positive/negative targets around the anchor prevent runaway targets when the reward gradient is large or noisy. Without clipping, targets can diverge and fitting collapses.
- **Detachment is not optional.** Backpropagating through target construction turns this back into a policy-gradient method with all its variance.
- **EMA decay rate ↔ off-policy risk.** Fast EMA = trainable policy quickly changes the behavior distribution, at risk of instability. Slow EMA = safer but caps how far the trainable policy can move per iteration.
- **Reward gradient availability.** The method assumes $\nabla R$ is accessible or estimable — trivial for differentiable reward models, requires score-function estimation for black-box rewards.

## Sources

- Paper: *On-Policy Self-Distillation in Diffusion Models* — Zhou et al., 2026 — introduces DiffusionOPSD. [arXiv:2608.24646](https://arxiv.org/abs/2608.24646).
- Related: *SecOPD* (Peng et al., 2026) — token-level on-policy distillation for prompt-injection defense.
- Related: *OPDVR* (Lin et al., 2026) — reformulates sampled-token OPD as an RLVR method via ReLU-gated rewards.
