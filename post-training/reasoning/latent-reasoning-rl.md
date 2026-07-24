# Latent Reasoning RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Latent reasoners (Coconut-style) carry intermediate CoT as continuous vectors, matching explicit token CoT at much shorter horizons. But they're stuck at imitation because outcome-reward RL requires a tractable per-step likelihood and a stopping rule — neither of which continuous-vector trajectories provide. SLPO (2026) introduces a surrogate policy density over latent transitions + a correctness-supervised stopping head, unlocking GRPO-style RL for latent CoT.

**Prereqs:** [long-cot-rl.md](./long-cot-rl.md), [../grpo.md](../grpo.md), [../rlvr.md](../rlvr.md)
**Related:** [length-penalty.md](./length-penalty.md), [../_rl.md](../_rl.md), [../_rewards.md](../_rewards.md)

---

## What it is

Explicit CoT decodes each intermediate step as a token — cheap to score under the model's usual token likelihood, easy to plug into PPO/GRPO. Latent CoT (Coconut and successors) skips detokenization: the reasoning trajectory is a sequence of continuous vectors passed forward in the residual stream. This is compute-efficient — no vocabulary softmax on intermediate steps — but breaks the RL setup:

1. **No per-step likelihood.** The transition between latent vectors is a deterministic (or near-deterministic) function of the model, not a sampled action from a categorical distribution. There's no `log π(a|s)` to compute a policy gradient over.
2. **No stopping interface.** Explicit CoT can stop when it emits `</think>`. Latent reasoners run for a *fixed* number of steps determined at training time, which prevents variable-horizon test-time scaling.

Both problems have to be solved together to bring RLVR to latent reasoning.

## How it works (SLPO)

**Surrogate density.** SLPO defines an empirical surrogate policy density over the latent transitions — a Gaussian (or similar) fit around the deterministic mean, treating the residual noise from denoising / sampling steps as the source of stochasticity. This surrogate gives a computable `log π(z_t | z_{t-1})` for trajectory-level credit assignment.

**Stopping head.** A small correctness-supervised head is trained jointly with the base model. Its target is "would this latent trajectory produce a correct final answer if halted here?" — a binary signal from the outcome verifier. Outcome-reward RL then refines the stopping head into a *variable-horizon* halting policy: the model learns to allocate more latent steps to harder instances.

**RL loop.** Once the surrogate density + stopping head are in place, the rest of the pipeline is standard GRPO with verifiable rewards. Sample G latent rollouts per prompt, run the verifier on the final decoded answer, compute group-relative advantages, update.

## Why it matters

- **Test-time compute scaling for latent CoT.** Before SLPO, latent reasoners were fixed-horizon and imitation-trained. Now they get the same "spend more compute on harder problems" behavior explicit CoT gets via RL, but at latent-space cost.
- **Cheaper Pass@k at inference.** Latent trajectories are shorter and cheaper; combined with variable-horizon halting, they can match or beat explicit CoT on Pass@k under a compute budget.
- **Reopens latent CoT as a serious research direction.** The RL barrier was the main reason latent reasoning stalled after Coconut. If SLPO generalizes, latent CoT becomes a candidate default for long reasoning.

## Gotchas & tricks

- The surrogate density is a *fit*, not the true transition kernel — bias in the surrogate translates to bias in the policy gradient. Paper argues empirically the bias is tolerable; stress-test on your own domain.
- The stopping head has to be trained *before* RL, not just jointly with it — a randomly-initialized head produces meaningless halting behavior and every rollout will look similar to the verifier.
- Standard GRPO gotchas still apply (see [../grpo.md](../grpo.md)): KL penalty tuning, group size G, entropy collapse.

## Sources

- Paper: *SLPO: Scaling Latent Reasoning via a Surrogate Policy* — You, Liu, Li, Li, 2026 — [arXiv:2607.19691](https://arxiv.org/abs/2607.19691)
- Prior art: *Coconut* (Chain of Continuous Thought) — the imitation-trained latent reasoner baseline SLPO extends.
