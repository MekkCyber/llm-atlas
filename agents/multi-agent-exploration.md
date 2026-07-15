# Multi-Agent Exploration (MACE)
*Depth — LLM agents pick peers to interact with; naive selection is myopic, and explicit exploration structure closes the gap.*

**TL;DR:** In multi-agent systems, each agent must **probe peers to infer their capabilities** before choosing who to interact with. LLM agents fail at this — they pick peers based on surface cues (name, first response, popularity) and end up in polarized, myopic coordination patterns. MACE (Multi-Agent Contextual Exploration) adds an explicit exploration term to peer selection, and lifts downstream task performance.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md) (exploration/exploitation), [README.md](README.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Multi-agent LLM systems (debate, division-of-labor, agent society) assume agents can figure out **who to interact with and how**. The paper formalizes this as a **partially observable stochastic game (POSG)** where each agent's optimal policy requires inferring peers' latent capabilities from limited interaction traces.

Empirically, modern LLM agents fail two ways:

- **Myopic** — pick the peer whose first response looked best, then stop exploring.
- **Polarized** — quickly collapse to a small subset of peers and never probe others, even when doing so would improve regret.

MACE is a lightweight framework that fixes this by making peer-selection explicitly exploration-aware, treating it as a contextual bandit rather than a free-form LLM decision.

## How it works

### The POSG framing

- **Agents:** a pool of LLM peers with unknown, heterogeneous capabilities (differ in domain skill, quirks, biases).
- **State:** each agent's belief about peers, plus the current task context.
- **Action:** who to interact with next, plus what to ask.
- **Reward:** task-completion score (per-turn or terminal).
- **Observation:** the peer's response — noisy, partial information about their true capability.

### The MACE update

Instead of letting the LLM freely pick peers from an in-context list, MACE:

1. Maintains a **structured belief** over peer capabilities updated from past interactions (context or lightweight parametric).
2. Selects peers using an **exploration-augmented score** — expected task value + a term rewarding uncertainty reduction. In practice this looks like an upper-confidence-bound-style selection layered on top of the LLM's judgment.
3. Feeds the LLM the *selected* peer plus a rationale ("this peer is untested on X"), letting the LLM handle *how* to interact once *who* is picked.

The key move: **separate peer selection (a bandit) from interaction (an LLM)**. Free-form LLM peer selection is where the exploration failure lives.

### Theoretical result

The **value of exploration grows with agent diversity.** In a homogeneous pool, probing peers gives little new information; in a diverse pool (contextual or parametric differences), probing dominates myopic selection.

## Why it matters

- Names a concrete failure mode of multi-agent LLM systems and gives a plug-in fix. Relevant for debate-style ensembles, mixture-of-experts routing at the agent level, and any agent society with heterogeneous peers.
- Ties multi-agent LLM behavior back to well-understood bandit / POSG theory — a bridge for importing decades of exploration-vs-exploitation results.
- Motivates deliberate *peer diversity* when designing agent pools: heterogeneity is not just fault-tolerance, it's what makes exploration payoffs real.

## Gotchas & tricks

- **Structured belief overhead.** Maintaining an explicit belief state grows with pool size; for large pools, use sketches or clustering.
- **Confounding with prompt design.** If peers see different prompts, apparent capability differences may be prompt-driven, not model-driven. Standardize the interface before attributing failures to peer capability.
- **Reward attribution.** In multi-turn tasks, credit-assignment to individual peer picks is noisy. The paper uses episode-level regret, not per-turn credit.
- **Doesn't fix single-agent reasoning failures.** MACE selects better peers; it doesn't make any individual peer smarter. If all peers are weak on a task, exploration won't help.

## Sources

- Paper: *Multi-Agent LLMs Fail to Explore Each Other* — Choi, Li, Li, Wang, Li, University of Wisconsin–Madison, 2026 — arXiv:2607.11250. Code: https://github.com/deeplearning-wisc/mace
- Related classical foundation: contextual bandits and POSG formalisms (Sutton & Barto for the RL prerequisites — see [../post-training/_rl.md](../post-training/_rl.md)).
