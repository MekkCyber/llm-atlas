# Mixture-of-Agents (MoA)
*Depth — sparse routing over specialized LLM agents, with co-trained router + agents (CERA-MoA).*

**TL;DR:** Mixture-of-Agents (MoA) is the per-model analogue of MoE's per-expert routing: many specialized LLM agents behind one router that activates a sparse subset per query. Most MoA systems freeze the router while fine-tuning agents (or the reverse), so routing can't track agent-capability drift. **CERA-MoA** (Jiang et al. 2026) co-evolves the two in an iterative RL loop, using a **predictive familiarity estimator** on mid-layer hidden states to score agent competence *without* running full rollouts.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md)
**Related:** [README](README.md) · [../architectures/aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md) · [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md)

---

## What it is

A composition of $N$ LLM agents $\{\pi_1, \ldots, \pi_N\}$ (each a full model, not an FFN expert) and a **router** that decides which agents handle a given query. Contrast:

- **MoE:** router picks $K$ FFN experts per *token*. Fine-grained, in-model, integrated in one forward pass.
- **MoA:** router picks $K$ agents per *query*. Coarse-grained, cross-model, each agent runs its own forward pass.

MoA is attractive when specialized agents already exist (domain fine-tunes, safety-tuned models, tool-specific agents) and the composition cost is dominated by inference of one or two agents rather than one or two FFNs.

## How it works — CERA-MoA in particular

CERA-MoA's contribution is the *co-training* recipe:

- **Predictive familiarity estimator.** A small head reads the *mid-layer hidden states* of each candidate agent given the query prompt (no need to run the agent to completion). It predicts whether the agent will succeed on this query — a cheap competence score.
- **Cumulative-threshold adaptive routing.** Sort agents by familiarity, activate them one by one until the *cumulative* score crosses a confidence threshold. Different queries get different subset sizes.
- **Iterative RL co-training.** Alternate:
  1. Run router on a batch, collect (query, activated-subset, outcome) trajectories.
  2. Update agents with RL on the queries they were assigned (proactive sample allocation drives specialization).
  3. Update the router (and familiarity estimator) using the observed outcomes.

Because router and agents update *together*, the router tracks capability drift and agents specialize toward the queries the router sends them — a virtuous feedback loop.

## Why it matters

- **The right coordination primitive for multi-model systems.** MoE captures fine-grained expertise but requires from-scratch training. MoA composes existing agents, which is the practical reality for teams shipping mixed-model deployments.
- **Familiarity from hidden states is a general primitive.** The "cheap competence estimator without a full rollout" idea reappears wherever routing must be sample-efficient (also relevant to speculative decoding and reasoning-model selection).
- **Beats static baselines.** CERA-MoA outperforms static-router and fixed-workflow MoA baselines across domains.

## Gotchas & tricks

- **Threshold calibration.** Cumulative-threshold routing is sensitive to the threshold — set too tight and it always activates one agent; too loose and it activates all. Tune per deployment.
- **Familiarity generalizes only within the router's training distribution.** OOD queries collapse the estimator; wrap with a fallback to a strong all-agents ensemble.
- **Co-training is expensive.** Each RL iteration touches all agents whose gradient signals are non-trivial. Sequential (agent-by-agent) updates help; asynchronous updates risk router/agent decoherence.
- **Not MoE's inference-cost story.** MoA saves compute *only if* the router picks few agents on average. When routing collapses to "always run the strongest 2", MoA becomes a small ensemble and its efficiency claim vanishes.

## Sources

- Paper: *CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents* — Jiang, He, Fang, Tsinghua, 2026, arXiv 2609.18779.
- Related: *Mixture-of-Agents Enhances Large Language Model Capabilities* — Wang et al., 2024 (earlier MoA prototype).
