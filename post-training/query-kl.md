# Query-KL (QKL)
*Depth — a query-side KL regularizer for LLM policy optimization, introduced by ERPO.*

**TL;DR:** In GRPO/PPO for LLMs, the KL term usually lives on the action distribution — it clamps per-token drift and forces a stability–exploration tradeoff. Query-KL (QKL) moves the KL to the **query distribution** the current policy induces during rollouts, plus a static per-query weight derived once from a reference. This bounds the drift that actually destabilizes long-horizon training without over-restricting exploration inside each rollout.

**Prereqs:** [grpo](grpo.md), [ppo](ppo.md), [rlvr](rlvr.md)
**Related:** [_rl](_rl.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

A KL regularizer applied at the *query* level rather than the token/action level. During RL post-training the policy induces a distribution over which rollouts get sampled (via temperature, curriculum, and its own reward-shaped preferences). QKL bounds how far this induced distribution can drift from a reference; a reference-derived per-query weight focuses the penalty on queries whose reference-quality actually matters.

## How it works

- **Action-side Policy-KL** (baseline): per-token `KL(π_θ(·|s) || π_ref(·|s))` averaged over the rollout, added to the RL loss.
- **QKL:** treat the aggregate *query distribution* induced by the policy as the object. Bound `KL(p_θ(q) || p_ref(q))` — cheap to estimate from rollout statistics.
- **Reference-derived weight `w(q)`**: pre-computed once from the reference policy, static during training. Downweights queries where drift is expected and safe (easy queries), upweights queries where drift signals blow-up (hard reasoning, long-horizon).
- Combined with the standard rollout advantage (GRPO-style groups) so exploration inside a query is unrestricted; only the *choice of query* is regularized.

## Why it matters

Long-horizon RL post-training is bottlenecked by *stability*, not raw reward. Action-side KL either kills exploration (large coefficient) or lets the policy diverge (small coefficient). QKL sidesteps this by putting the constraint where drift shows up empirically — the query mixture — leaving per-rollout exploration free.

## Gotchas & tricks

- The reference-weight `w(q)` is static; if the reference is weak on hard queries you may under-regularize where you need it most. Choosing the reference matters.
- Estimating `p_θ(q)` from rollout counts is noisy for rare queries; combine with a light per-query prior.
- ERPO reports the largest gains at **high decoding temperature** and **long training horizons** — the regimes where GRPO/PPO variants blow up. At short horizons and low temperatures the gain over standard Policy-KL is small.

## Sources

- Paper: *Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization* — Meng et al., 2026 — [arXiv:2608.23311](https://arxiv.org/abs/2608.23311)
