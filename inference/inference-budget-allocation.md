# Inference-Budget Allocation
*Depth — globally allocating reasoning tokens across a stream of queries instead of per-query.*

**TL;DR:** "Think harder" is a *per-query* decision; what gets deployed is a *cluster* serving many queries under a shared compute budget. CLEAR (Zhu et al., 2026) formulates inference-time scaling as a **global constrained optimization**: maximize total accuracy subject to a token budget, model each query's reasoning utility as a *shifted-surge* function, and let the optimization's **shadow price λ** decide which queries get more thinking and which get *abandoned*. In resource-scarce regimes this beats uniform allocation by up to **3× global accuracy**.

**Prereqs:** [_rl](../post-training/_rl.md)
**Related:** [length-penalty](../post-training/reasoning/length-penalty.md), [long2short](../post-training/reasoning/long2short.md), [grpo](../post-training/grpo.md)

---

## What it is

A serving-time policy that takes a stream of incoming reasoning queries, a finite total token budget per unit time, and decides, *per query*: keep thinking, stop and emit, or abandon. The decision is made not by a per-query heuristic but by a single global equilibrium price λ that equates marginal utility across the stream.

## How it works

The economic framing:

- Each query $i$ has a utility $u_i(t_i)$ for spending $t_i$ tokens of reasoning. CLEAR fits a **shifted-surge** form: utility is near-zero until $t_i$ reaches a query-specific *emergence threshold*, then surges, then saturates.
- The global problem:

$$
\max_{\{t_i\}} \sum_i u_i(t_i) \quad \text{s.t.} \quad \sum_i t_i \le B
$$

- KKT optimality: at equilibrium, every query that's still running has the same **marginal utility** $u_i'(t_i) = \lambda$. λ is the *shadow price* — what one extra token of total budget is worth.
- Operationally, CLEAR maintains an online estimate of each query's utility curve from in-flight signal (e.g. self-consistency, verifier confidence) and an online estimate of λ. A query is:
  - **Continued** if its marginal utility exceeds λ.
  - **Reallocated to** (boosted) if it sits near its emergence threshold (a high-marginal-utility region).
  - **Abandoned** if its marginal utility falls below λ — those tokens are reinvested elsewhere ("rational abandonment").

## Why it matters

- **Pareto dominance.** Across multiple reasoning tasks, CLEAR pushes the (token-cost, accuracy) Pareto frontier outward vs. uniform-per-query budgets.
- **3× in scarce regimes.** When the cluster is heavily constrained, knowing *which queries to give up on* is the single highest-leverage decision.
- **Decouples training from serving.** The training-time levers (length-penalty, long2short) shape *each model's* reasoning; CLEAR is a serving-time lever that doesn't touch the weights.

## Gotchas & tricks

- **Utility estimation is the hard ML problem.** The shifted-surge fit needs an online estimate of "how close is this query to its emergence threshold?" — usually a small classifier on the rollout's current hidden state or self-consistency variance.
- **Abandonment is socially uncomfortable.** Users notice when their query is killed mid-reasoning. Pair with a fallback (shortest-path answer) rather than an empty response.
- **λ is global; QoS classes need their own.** If you have free and paid tiers, run two markets with separate λ's, not one.
- **Doesn't replace per-query length control.** A model that *cannot* stop reasoning when given a short budget undermines CLEAR; combine with length-penalty-trained models for best results.

## Sources

- Paper: *The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs* — Zhu et al., 2026 — [arXiv:2606.03092](https://arxiv.org/abs/2606.03092) — formulates the constrained optimization and introduces CLEAR.
