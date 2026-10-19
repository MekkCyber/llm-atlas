# MRTPO — Multi-Reward Token-level Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In RL with multiple verifiable rewards (correctness + format + tool-call validity + …), GRPO/GDPO broadcast every reward to every token with the same weight. MRTPO instead measures, per reward and per token, how much that reward's verifier *affects* the model's next-token distribution — using a counterfactual replay — and uses that influence as a soft mask over the per-reward advantage.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [_rewards.md](./_rewards.md)
**Related:** [_rl.md](./_rl.md), [ppo.md](./ppo.md)

---

## What it is

Stacking multiple verifiers in RLVR is standard practice (R1 Stage 2 sums accuracy + format + language; Tülu 3 adds more). GRPO treats the sum $R = \sum_r R_r$ as a single scalar, assigns one group-relative advantage $A_i$ to the trajectory, and broadcasts $A_i$ to every token. This under-fits: a format reward depends on a few tokens (opening tag, closing tag) and should not spread credit across the whole chain-of-thought.

MRTPO assigns, for each reward $R_r$, a *per-token* mask $m_r(x_t)$ measuring which tokens that reward actually depends on, and redistributes the reward-specific advantage accordingly:

$$
A_t = \sum_r m_r(x_t) \cdot A_r
$$

where $A_r$ is the group-relative advantage for reward $r$.

---

## How it works

For each sampled response and each reward $r$:

1. **Counterfactual replay.** Rerun the response with the verifier's information for $r$ *masked out* (e.g., strip the format tags from the context the verifier sees, or null the tool-schema input).
2. **Measure distribution shift.** At every token $x_t$, compute the Jensen–Shannon distance between the model's next-token distribution with and without that verifier's input:

$$
m_r(x_t) = \mathrm{JS}\!\big(\, \pi_\theta(\cdot \mid x_{<t},\, \text{full}) \,\|\, \pi_\theta(\cdot \mid x_{<t},\, \text{masked}_r) \,\big)
$$

3. **Normalize** $m_r$ across tokens so it sums to 1 (per reward, per response).
4. **Redistribute advantages.** Multiply reward $r$'s group-relative advantage by $m_r(x_t)$ at each token; sum across rewards for the final per-token advantage.
5. **Standard PPO clip** on the weighted token-level advantages.

The expensive step is (1) — one extra model forward per reward per response — but masks are shared across epochs within a step.

---

## Why it matters

- **Faster convergence.** Reported on text-to-SQL, multi-constraint instruction following, and tool calling: higher final performance *and* fewer steps to reach it than GRPO/GDPO baselines.
- **Reduces reward interference.** When two rewards disagree (format vs correctness), uniform broadcast lets each one push on tokens the other cares about. MRTPO isolates them, so adding a shaping reward no longer costs headline accuracy.
- **General pattern.** The counterfactual-ablation idea generalizes beyond RL: any time you want "which tokens does this classifier depend on?" the JS-divergence trick gives a cheap input-attribution.

---

## Gotchas & tricks

- **Cost scales with #rewards.** For $k$ rewards, you do $k$ extra counterfactual forwards per rollout. On rich RLVR stacks (6–8 verifiers) this is non-trivial — but still less than training a value network.
- **Verifier isolation must be meaningful.** The counterfactual needs to actually hide reward $r$'s information from the model's conditioning. For a format reward this is easy (strip tags); for a correctness verifier whose signal is implicit in the question, less so.
- **Group-relative baseline is kept.** MRTPO changes credit *within* a trajectory; the across-trajectory baseline is still GRPO's group mean and std.
- **Compose with KL anchor as usual.** MRTPO is orthogonal to the reference-model KL — keep $\beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ to prevent drift.

---

## Sources

- Paper: *Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL* — Wang, Liu, Song, Ding, Liang, Kang, 2026 — arXiv [2610.21439](https://arxiv.org/abs/2610.21439).
- Baseline: *GRPO* — [grpo.md](./grpo.md).
