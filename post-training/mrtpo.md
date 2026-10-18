# MRTPO (Multi-Reward Token-level Policy Optimization)
*Depth — token-level credit assignment per reward in multi-reward RLVR.*

**TL;DR:** When an RL policy is trained against several verifiable rewards at once (tool-call correctness, format, individual instruction constraints), standard methods like GDPO/GRPO normalize each reward separately but still spread its advantage uniformly across every token. MRTPO instead gives each reward's advantage only to the tokens that reward *actually depends on*, measured on the fly by replaying the response with and without that reward's verifier and taking per-token JS distance between next-token distributions. No auxiliary model, no token labels.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [_rewards.md](_rewards.md)
**Related:** [ppo.md](ppo.md), [cot-reward-model.md](cot-reward-model.md)

---

## What it is

A principled per-reward token mask that comes from the policy itself. Multi-reward RLVR usually plugs several rule-based verifiers (format regex, SQL execution, constraint-match) in parallel; each contributes a scalar reward. The core question is "which tokens *made* this reward fire?" MRTPO answers it without extra labels.

## How it works

For sampled response tokens `y_t` and each reward `r_k`:

1. Compute two next-token distributions: `π(·|y_<t, with-verifier-of-r_k)` and `π(·|y_<t, without-verifier-of-r_k)`. "Verifier context" means whatever cue the policy was given about this reward (schema snippet, format spec, constraint list).
2. Let `w_{t,k} = JSD(π_with, π_without)` be the per-token Jensen–Shannon distance.
3. Normalize `w` over `t`, then multiply reward `r_k`'s advantage at step `t` by `w_{t,k}`.
4. Sum across rewards; use the token-weighted advantage in the usual policy-gradient update (GRPO-style group-relative advantage works).

## Why it matters

Multi-reward is the standard RLVR setting (format + correctness + constraints), and the universal failure mode is one reward's gradient drowning out others. Reward-specific token credit is a cheap, auxiliary-free fix: no extra model, no human labels, just two forward passes per reward.

## Gotchas & tricks

- Needs a verifier context the model can condition on (schema prompt, format spec). Rewards with no explicit cue (pure execution reward) have no "with/without" to replay — fall back to uniform.
- Two forward passes per reward multiply the cost; MRTPO works best with 2–5 rewards.
- Reported wins on text-to-SQL, multi-constraint IF, and tool-calling with GRPO/GDPO baselines; convergence is faster, not only final-accuracy higher.

## Sources

- Paper: *Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL* — Wang et al., NUS/UIUC/USC et al., 2026 — [arXiv:2610.21439](https://arxiv.org/abs/2610.21439)
