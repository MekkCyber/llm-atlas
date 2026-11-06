# Pass@(K,T) — Decomposing Agent Evaluation
*Depth — a two-axis extension of Pass@K for multi-turn agents.*

**TL;DR:** A two-dimensional extension of the standard Pass@K metric for **multi-turn agents**. The axis K is still *sampling breadth* (how many independent trajectories per task), and the new axis T is *interaction depth* (how many turns per trajectory). Pass@(K,T) = 1 − Pr[all K rollouts of depth T fail]. Reporting the whole surface — not a single number — exposes which gains come from more sampling versus more interaction, and reveals that many "multi-turn RL improves agents" claims collapse under small T budgets.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [livecodebench](./livecodebench.md) · [mmlu](./mmlu.md)

---

## What it is

The Codex Pass@K estimator measures "with K independent single-shot samples, how often does at least one succeed?" That is enough for one-shot tasks. For **agents** — which take many turns of tool use, observation, and recovery — a single number conflates two very different kinds of capability:

- *Can the agent eventually solve this with enough tries?* → axis K.
- *Can the agent recover from mistakes within one trajectory?* → axis T.

Pass@(K,T) separates them. Given a task, sample K independent trajectories of up to T turns each; the metric is the probability that at least one trajectory succeeds.

## How it works

For a task with ground-truth success indicator $y(\tau)$ on trajectory $\tau$, let $p(T)$ be the probability a single trajectory of depth $T$ succeeds under the policy. Then

$$
\text{Pass@}(K, T) \;=\; 1 - \bigl(1 - p(T)\bigr)^K
$$

In practice, the unbiased estimator (per Codex Pass@K) uses `n ≥ K` sampled trajectories per task and computes $1 - \binom{n - c}{K} / \binom{n}{K}$ where $c$ is the count of successes in `n` samples. The whole report is a **surface** — one value per $(K, T)$ cell — usually plotted as a heatmap or a family of curves.

**Diagnostic use.** Fix one axis and read the other:

- Fix small T (tight interaction budget): does raising K recover performance? If no, the policy is not just mis-planning, it's mis-stepping.
- Fix K = 1 (best single-trajectory): does raising T improve? If yes, the policy's gains are from *recovery*, not from *per-step accuracy*.

## Why it matters

The agent-RL literature frequently claims "multi-turn RL beats single-turn RL," but reports only the easiest cell of the Pass@(K,T) surface. The same RL update can yield:

- Big wins at large T (the policy learned to recover) but no wins at small T (per-step quality unchanged).
- Big wins at small K but shrinking advantage as K grows (RL concentrates mass that sampling already recovers).

Reporting Pass@(K,T) forces the author to say *where* in the budget space their method wins. Pairs naturally with the **BudgetMix RL** training recipe (randomize T during training) so the policy's Pass@(K,T) surface is flat across the deployment range.

## Gotchas & tricks

- Need `n ≥ K_max` samples per (task, T) cell to produce unbiased estimators — the full surface is sampling-expensive.
- At very small T, tasks that require any tool-use at all drop to zero; include T = 0 or T = 1 only when it's a meaningful comparator.
- "Depth T" is ambiguous when turns vary in length — fix *what counts as a turn* up front (one tool call, one action, one model response).

## Sources

- Paper: *What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?* — Zhu et al., 2026 — https://arxiv.org/abs/2611.06483
- Paper: *Evaluating Large Language Models Trained on Code* (Codex) — Chen et al., 2021 — https://arxiv.org/abs/2107.03374 (origin of the Pass@K estimator)
