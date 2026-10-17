# Multi-Session Agentic RL (Mosaic)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard agentic RL assumes a trajectory is one continuous session. Real agents break sessions via **context compaction** (compress-and-continue) and **subagent forking** (hand off to a child agent). Mosaic trains across these splits with a **critic-free, hierarchical advantage estimator**: shape the trajectory-level advantage by session-usage signals, and shape each session's advantage by its estimated marginal contribution to the final outcome.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

An agent run with context compaction or subagent forking produces trajectory
$\tau = (S_1, S_2, \ldots, S_K)$
where each $S_k$ is one *session* — a self-contained context window — and $K$ varies per trajectory. Only the final outcome $r(\tau)$ is observed.

Two existing options waste signal:
- Train only on the final session → ignores K−1 earlier sessions that did the actual work.
- Broadcast $r(\tau)$ equally to every session (naive GRPO extension) → every session gets the same credit regardless of whether it was pivotal, exploratory, or redundant.

Mosaic sits between them: broadcast *then* reshape.

## How it works

### Trajectory-level advantage, usage-aware

For a prompt $q$ with $G$ rollouts $\{\tau_i\}$ and session counts $\{K_i\}$:

$$
A_i^{\text{traj}} = \frac{r(\tau_i) - \bar r}{\sigma_r} \cdot \phi(K_i)
$$

where $\phi$ is a monotone session-usage shaping term — penalize trajectories that used many sessions for the same reward (encourages compactness). GRPO-style $\sigma_r$ normalization is kept.

### Session-level advantage, marginal-contribution shaping

Within a trajectory, each session's shared advantage is scaled by its **estimated marginal contribution** $m_k$ to the final reward. Mosaic estimates $m_k$ without a learned critic — e.g. via a leave-one-session-out ablation using the policy itself to re-score, or a sum-decomposition of the final answer attributable to each session's output. The exact estimator is architectural; the paper uses a lightweight response-based attribution.

$$
A_{i,k} = m_{i,k} \cdot A_i^{\text{traj}} \qquad \sum_k m_{i,k} = 1
$$

Standard GRPO/PPO objective is then applied with per-token advantages set to $A_{i,k}$ for tokens inside session $k$.

### Why "critic-free"

A fully learned critic over a multi-session trajectory would have to predict value across session boundaries where the context is intentionally thrown away (compaction) or forked (subagent). Both estimators above are **sample-based**, so no value network is needed — the trade is more rollouts, not more parameters.

## Why it matters

- **Matches inference.** Agent frameworks (deep research, long-horizon coding) already use compaction and subagents; training single-session and running multi-session is a direct train–test mismatch.
- **Pushes allocation to the model.** Session-usage shaping gives the policy a reason to use fewer sessions when they're wasteful — a cost-aware signal otherwise added as a hand-tuned token budget.
- **Drop-in.** No critic, no new losses, no architectural change — just a two-stage advantage shaper on top of a GRPO loop.

## Gotchas & tricks

- **Marginal-contribution estimators are the design point.** Paper uses a leave-one-out style estimate; alternatives (shapley-value sampling, learned attributions) sit in the same slot.
- **Session-usage shaping can collapse.** Too strong a $\phi$ penalty makes the policy avoid forking even when it would help. Balance via the strength / curvature of $\phi$.
- **Verified on BrowseComp.** Reported on BrowseComp with Qwen3.5-4B, both compaction and subagent-forking regimes; stability on longer-horizon code agents is open.
- **Compounds with [partial-rollouts.md](../systems/partial-rollouts.md).** The sessions are the natural chunk boundary for asynchronous rollout serving — the two abstractions line up.

## Sources

- Paper: *Mosaic: Multi-Session Agentic Reinforcement Learning* — Xu, Zhou, Long, Tan, Guo (HKUST(GZ) + Huawei Foundation Model Lab), 2026 — [arXiv:2610.22174](https://arxiv.org/abs/2610.22174).
- Code: https://github.com/LARK-AI-Lab/Mosaic
