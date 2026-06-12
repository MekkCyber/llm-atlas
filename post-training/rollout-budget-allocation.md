# Rollout Budget Allocation (TRACE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In RL with verifiable rewards (RLVR), much of the rollout budget is wasted on prompts where every rollout returns the same reward — when all $G$ samples succeed or all fail, group-normalized advantages collapse to zero and the policy update for that prompt is uninformative. **TRACE** (Tsinghua / Tencent, 2026) treats rollout budgeting as a first-class objective: a unified allocator distributes a *fixed total budget* across prompts, attempts, and trajectories to maximize **reward contrast per token**. On Qwen3-14B Multi-Hop QA, +2.8 points average accuracy over competitive baselines at matched sampling cost.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md)
**Related:** [rl-prompt-curation.md](rl-prompt-curation.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [_rl.md](_rl.md)

---

## What it is

GRPO and other group-normalized RL algorithms compute an advantage that's *zero* whenever all rollouts in a group share a reward (formally: $\sigma_r = 0$ in the group-normalization step). The policy update for that prompt contributes nothing to the gradient. This happens often:

- **Trivial prompts** — every rollout succeeds. The model already knows.
- **Impossible prompts** — every rollout fails. The model can't recover here either.
- **Saturated prompts** — early in training the prompt has signal, but after the model masters it, all rollouts succeed.

Standard practice samples $G$ rollouts per prompt uniformly across the dataset. TRACE allocates *more* rollouts to prompts where reward contrast is informative (mixed success/failure) and *fewer* to degenerate prompts, under a fixed total budget.

---

## How it works

### Three allocation levels

TRACE unifies three rollout-budget questions:

1. **Prompt-level:** which prompts should get more attempts than others?
2. **Attempt-level:** within a prompt, when should we stop adding rollouts (we've seen enough contrast) vs keep sampling (still uniform)?
3. **Trajectory-level:** for agentic settings, how many steps deep should a trajectory go before the budget cuts it off?

A single allocation policy answers all three by estimating the **expected information gain per token spent** at each branch point. Branches with high expected contrast get more budget; branches with degenerate outcomes get cut.

### Information-gain proxy

Reward contrast is operationalized as the **variance of rewards** observed so far for that branch. Branches with low observed variance (early signs of degenerate outcome) get downweighted; branches with high observed variance get upweighted. Concretely:

```
for each prompt p in batch:
    initial_rollouts = small number, e.g. 4
    observed_variance = var(rewards on initial rollouts)
    extra_rollouts = allocate proportional to observed_variance / token_cost
```

The variance estimate stabilizes as more rollouts accrue, so the allocator can dynamically expand high-signal prompts and shrink degenerate ones within a single batch.

### Composes with GRPO

TRACE allocates the rollouts; GRPO (or any group-normalized algorithm) computes advantages and updates the policy on whatever rollouts the allocator produced. There's no change to the policy-update math — just to how rollouts are distributed.

---

## Why it matters

- **Same compute, more learning signal.** Rollout cost is the binding constraint of agentic RL. TRACE keeps total cost fixed while increasing the *useful* gradient contribution per dollar.
- **+2.8 points** average accuracy on Qwen3-14B Multi-Hop QA over competitive baselines **at equal sampling cost** — a clean apples-to-apples win.
- **Composable with rollout systems.** TRACE allocates budget; partial rollouts ([../systems/partial-rollouts.md](../systems/partial-rollouts.md)) and MTP speedups (Bebop) make each allocated rollout cheaper. Stack all three.
- **Generalizes to multi-step agentic settings.** Trajectory-level allocation is the same problem: where to spend depth in a tool-use rollout.

---

## Gotchas & tricks

- **Cold-start bias.** Variance estimates from the first 2–4 rollouts are noisy; the allocator can prematurely cut a prompt that would have shown contrast on rollout 5. Use a minimum-rollout floor before applying the allocator's signal.
- **Adversarial against curriculum.** TRACE prefers prompts where the model is currently uncertain. That's roughly the curriculum sweet spot, but it can also stall progress on prompts the model is *about to* master if rewards are noisy — adapt the variance threshold over training.
- **Independent of reward type.** Works for RLVR (rule rewards), preference-RM rewards, and PRM-style step rewards. The variance signal is reward-agnostic.
- **Don't conflate with [rl-prompt-curation](rl-prompt-curation.md).** Prompt curation chooses *which prompts enter the dataset*; TRACE chooses *how to spend the rollout budget across the chosen prompts*. Composable; not the same problem.

---

## Sources

- Paper: *TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning* — Zou, Wang, Qu, et al. (Tsinghua / Tencent), 2026 — [arXiv 2606.11119](https://arxiv.org/abs/2606.11119).
- Paper: *DeepSeekMath / GRPO* — Shao et al., 2024 — the group-normalized advantage that goes to zero on degenerate prompts.
