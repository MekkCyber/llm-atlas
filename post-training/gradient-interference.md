# Gradient interference in multi-task LLM training
*Depth — why SFT tasks collide but RL tasks coexist.*

**TL;DR:** Multi-task LLM post-training suffers from cross-task gradient interference. The 2026 SFT-vs-RL analysis frames this as two regimes: SFT interference is **norm-limited** (bound scales with gradient magnitude) while on-policy RL interference is **variance-limited** (bound scales with advantage-normalized gradient variance). Variance is small under standard PPO/GRPO advantage normalization, giving near-orthogonal per-task updates in RL and dense colliding updates in SFT.

**Prereqs:** [ppo.md](ppo.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [parallel-rl.md](parallel-rl.md), [_post-training.md](_post-training.md), [fine-tuning/README.md](fine-tuning/README.md)

---

## What it is

When you fine-tune one model on tasks $\{T_1, T_2, \dots\}$, the gradient from task $i$ can hurt task $j$'s loss — that's cross-task interference, and it's why "just add more tasks" is fragile. The framing here separates *why* interference is bad by post-training paradigm.

## How it works

The upper bound on interference between per-task gradients decomposes differently for the two paradigms:

- **SFT (norm-limited).** The interference bound is proportional to $\lVert g_i \rVert \cdot \lVert g_j \rVert$. Every SFT update pushes with the full log-likelihood gradient magnitude, so unless the tasks are perfectly aligned, they overlap. Interference grows with example loss.
- **On-policy RL (variance-limited).** Under advantage normalization ($A_i = (r_i - \bar r) / \sigma_r$) and on-policy sampling, the effective per-task gradient is small in variance — the bound is proportional to $\mathrm{Var}(g_i)^{1/2} \cdot \mathrm{Var}(g_j)^{1/2}$. Variance is empirically small (advantages are order-one, most tokens contribute near-zero signal), so per-task gradient directions are approximately orthogonal.

The paper backs this up at the parameter level: RL post-training induces **sparse and approximately orthogonal** weight updates across tasks; SFT does not.

## Why it matters

- **Explains an oddly-consistent empirical fact.** Practitioners have long noticed RL post-training composes multi-task capabilities better than SFT does; this gives a mechanistic reason.
- **Motivates decoupled RL pipelines** — see [parallel-rl.md](parallel-rl.md) — where per-task RL runs merge without the mixture-tuning cost of multi-stage SFT.
- **Reframes when to use SFT vs RL.** If your goal is capability composition over multiple tasks, the variance regime buys you interference resistance that SFT can't match.

## Gotchas & tricks

- **Off-policy RL breaks the story.** Importance-weighted updates inflate gradient variance and push you back toward SFT-like interference.
- **Requires proper advantage normalization.** GRPO's per-group $\sigma_r$ (or PPO with running-mean baseline) is what shrinks the variance bound; a raw REINFORCE gradient doesn't buy you this.
- **Not zero interference — small interference.** "Approximately orthogonal" ≠ "orthogonal." At the extremes (very many tasks, very correlated tasks), interference still shows up.
- **SFT isn't hopeless.** Techniques like PCGrad, GradVac, or careful curriculum can reduce SFT interference; the paper's point is that the RL variance regime buys it for free.

## Sources

- Paper: *SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning Paradigms for LLMs* — Zhu et al., 2026 — arXiv:2608.03573.
- Prior related: *Gradient Surgery for Multi-Task Learning (PCGrad)* — Yu et al., 2020 — the classical SFT-side fix.
