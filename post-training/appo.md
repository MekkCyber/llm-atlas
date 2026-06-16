# APPO — Agentic Procedural Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agentic RL extension of GRPO that *doesn't* trust tool-call boundaries. Decision points are scored by a **Branching Score** combining token uncertainty with the policy-induced likelihood gain of the best continuation; rollouts branch at high-score positions; credit is redistributed across the branched rollouts by a **procedure-level advantage scaling** rather than uniformly per token. Reported as ~+4 points average across 13 agentic benchmarks over strong GRPO baselines.

**Prereqs:** [grpo.md](grpo.md), [ppo.md](ppo.md), [_rl.md](_rl.md)
**Related:** [rlvr.md](rlvr.md), [rejection-sampling.md](rejection-sampling.md), [partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Token-uniform credit assignment (every token in a rollout shares the same advantage) is fine for short single-turn RLHF but is the wrong granularity for multi-turn tool-using agents. The intuitive fix — branch at tool-call boundaries — turns out empirically to be wrong: the pilot study shows pivot tokens (positions where a different choice would change the outcome) are scattered throughout the sequence, not concentrated at tool calls. Token entropy alone is also unreliable as a proxy.

APPO answers two questions: *where to branch* and *how to assign credit after branching*.

---

## How it works

### Branching Score

For each token position $t$ in a rollout, compute

$$
B_t = H(\pi_\theta(\cdot \mid s_t)) \cdot \big(\log \pi_\theta(\hat{c}_t \mid s_t) - \log \pi_\theta(\bar{c}_t \mid s_t)\big)
$$

where $H$ is the token entropy, $\hat{c}_t$ is the best continuation under the current policy and $\bar{c}_t$ a sampled alternative. The product filters out the dominant failure mode of pure-entropy branching: positions where the model is confused but downstream outcomes are similar regardless. Only positions with both high entropy *and* large continuation-likelihood gap get branched.

### Branching and rollout collection

The top-$k$ branching-scored positions in each rollout are selected as branch points; an alternative continuation is rolled out from each. The original rollout plus its branches form a *procedural group* — multiple trajectories sharing a common prefix that diverge at meaningful pivot points.

### Procedure-level advantage scaling

Group-relative advantages (GRPO-style) are computed within each procedural group, *then* scaled by a per-procedure factor that accounts for how much each branch's prefix contribution differs from the original. Credit is concentrated where the branched outcomes actually differ. The remainder of the update is a standard PPO-clipped objective with KL to the reference model.

---

## Why it matters

- **Right granularity for agentic RL.** Most agentic RL inherits token-uniform credit from RLHF. APPO is one of the first concrete proposals that finds pivot tokens automatically.
- **No external verifier required.** Branching Score is computed from the policy itself — no learned process reward model, no rule verifier. Compatible with sparse outcome rewards.
- **Drop-in over GRPO.** Same outer loop, same KL regularizer, same reference-model machinery. The change is in how rollouts are collected and how their advantages are aggregated.
- **+4 points across 13 benchmarks** over GRPO baselines while keeping tool-call counts similar — gains come from credit assignment, not from longer rollouts.

---

## Gotchas & tricks

- **Top-$k$ branch count is the main knob.** Too few branches: little signal differs from GRPO. Too many: per-prompt rollout cost balloons. The paper reports modest $k$ (single-digit per rollout).
- **Continuation likelihoods need a temperature.** $\hat{c}_t$ and $\bar{c}_t$ must come from sampling; greedy decoding makes the second term collapse to zero.
- **Branching at the end of a sequence is wasted.** Late-position branches have no procedural divergence. In practice, restrict branching to the first ~half of the rollout.
- **Procedure-level scaling can underweight the original rollout.** If many branches succeed where the original fails, the original's credit shrinks — usually fine, occasionally destabilizing. Pair with a small clip on the per-procedure factor.

---

## Sources

- Paper: *APPO: Agentic Procedural Policy Optimization* — Wang et al., USTC · AMAP Alibaba, 2026 — [arXiv 2606.12384](https://arxiv.org/abs/2606.12384).
- See also: [grpo.md](grpo.md) for the group-relative baseline APPO extends, and [partial-rollouts.md](../systems/partial-rollouts.md) for the systems pattern that makes mid-sequence branching tractable.
