# PACT (Policy-Aligned Critic Training)
*Depth — an actor-then-critic update order with IS-corrected critic learning for LLM RL.*

**TL;DR:** Standard actor-critic RL (PPO / GAE) trains the critic under the old policy and then uses it to score updates for the new one — so once the policy drifts, the critic's estimates lag and errors compound. PACT reverses the update order (**actor first, critic second**) and applies importance-sampling correction to the critic's loss, so the value function tracks the policy that is currently being optimized. Beats GRPO by 8.8 pts on agentic math and PPO by 13.2 pts, and reaches 67.4% on SWE-bench Verified.

**Prereqs:** [ppo](ppo.md), [_rl](_rl.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [_rewards](_rewards.md)

---

## What it is

An actor-critic training procedure for LLM RL that fixes a specific misalignment between the actor and the critic: after a policy update, the value network's estimates were computed under the previous policy and are stale. PACT formalizes token-level credit assignment through three axioms (**Completeness**, **Prefix Consistency**, **Neutrality**) that uniquely determine the token-level credit signal, then uses that characterization to argue that RLOO signals and on-policy distillation are, in expectation, aligned with token-level credit — motivating a proper critic-alignment step.

## How it works

Standard PPO cycle: rollout → compute advantages using critic → update actor → update critic.

PACT changes two things:

1. **Actor-first update order.** Update the actor with the current advantages, *then* recompute the critic against the freshly-updated policy.
2. **Importance-sampling correction on the critic loss.** When training the critic on the same rollouts, reweight each token by $\pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$ so the critic's fit is against the distribution the *new* actor will sample from.

Approximate token-level credit sparsity (derived from the three axioms under bounded outcome rewards) means most tokens contribute little to the true credit signal, so the intermediate critic errors that dominate GAE estimates matter disproportionately — the IS correction is precisely aimed at reducing that error.

## Why it matters

The field mostly abandoned value networks after GRPO, on the grounds that value learning is unreliable at LLM scale. PACT is a targeted counterargument: fix the alignment problem and the critic pulls its weight. On agentic math averaged over four benchmarks it reaches **72.87%** (vs GRPO 64.07%, PPO 59.71%). SWE-bench Verified: **67.4%** vs 65.0 (PPO) / 65.4 (GRPO) / 63.6 (SAO). Puts actor-critic back on the LLM-RL menu for long-horizon and agentic tasks where value information genuinely helps.

## Gotchas & tricks

- Requires storing per-token log-probs under both $\pi_\theta$ and $\pi_{\text{old}}$ for the critic loss — small memory hit vs GRPO but negligible next to the value network itself.
- The three-axiom framework is descriptive, not a constraint on the algorithm: it explains *why* PPO / GAE / RLOO / OPD look aligned in the limit but diverge under policy drift.
- Reported gains are largest on agentic tasks and SWE-bench, where trajectory length is long and terminal reward is far from per-token decisions — the regime critic-based estimation was always supposed to help.

## Sources

- Paper: *PACT: From Credit Assignment to Critic Alignment* — Fu et al., 2026 — [arXiv:2609.26355](https://arxiv.org/abs/2609.26355)
