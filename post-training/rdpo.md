# Role-Decoupled Policy Optimization (RDPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An RL objective for multi-role agent systems: instead of training one policy that plans, uses evidence, and synthesises answers, RDPO trains **two specialised policies** — a Planner and a Synthesizer — with **role-specific advantages** derived from a terminal outcome reward plus turn-level rubric scores. Introduced in IterSynth (2026), which beats the strongest ≤8B ReAct-style deep-search agents by +4.2 average points.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md), [ppo.md](ppo.md)
**Related:** [_rewards.md](_rewards.md), [rlvr.md](rlvr.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Classical ReAct-style agents interleave *think*, *search*, and *answer* actions from a single policy. Training them with GRPO gives every step the same terminal advantage — which conflates "the plan was good" with "the synthesis was good", making credit assignment poor and role skills bleed into each other.

RDPO's move is to (a) architecturally split the agent into two roles and (b) give the RL objective a **role-conditional advantage**. Concretely:

- **Planner** $\pi_P$ emits an *information need* for the next search turn.
- **Synthesizer** $\pi_S$ ingests retrieved evidence plus the current summary state and returns a new summary (and eventually the final answer).

For a trajectory $\tau$ made of alternating planner turns $p_1, p_2, \ldots$ and synthesizer turns $s_1, s_2, \ldots$, RDPO computes two advantage streams:

$$A_t^P = R_{\text{outcome}} + \gamma\, r^P_t - b^P \qquad A_t^S = R_{\text{outcome}} + \gamma\, r^S_t - b^S$$

Turn-level rubric rewards $r^P_t$ and $r^S_t$ are LLM-judged (or verifier-scored) per role — Planner is rated on information gain, Synthesizer on evidence integration — and each role uses **its own group-relative baseline**.

## How it works

Per RL step:

1. Roll out $G$ full trajectories per query with the paired (Planner, Synthesizer) policies.
2. Score the final answer against ground truth for $R_{\text{outcome}}$ (BrowseComp / Xbench-DS style).
3. For each planner turn, ask a judge to rate it on a rubric (query specificity, coverage, novelty) → $r^P_t$. Do the same for each synthesizer turn (grounding, non-hallucination, integration) → $r^S_t$.
4. Compute role-specific group means/stds and z-score the advantages *separately* for planner and synthesizer turns.
5. Update each policy with a standard PPO clip + KL-to-reference, using its own advantage stream.

The Planner and Synthesizer can share a base backbone or be independent — IterSynth uses a shared 8B backbone with role-conditioned LoRAs.

## Why it matters

- **Role-shaped credit assignment.** A trajectory that succeeds because of great synthesis over a mediocre plan (or vice versa) is now scored correctly per role; the wrong policy stops absorbing the wrong lesson.
- **Compatible with a summary-state agent.** RDPO pairs naturally with an agent that keeps a **rolling summary** instead of a growing scratchpad, cutting context noise in long searches.
- **Model-agnostic prompting paradigm.** IterSynth reports meaningful zero-shot gains just by running frontier proprietary models in Planner+Synthesizer roles at inference time, before any RL — RDPO then trains that same structure into an 8B open model.

## Gotchas & tricks

- **Rubric-judge quality is the ceiling.** If the rubric LLM can't tell a good plan from a bad one, RDPO reduces to GRPO with extra machinery. Validate rubric agreement with humans on a held-out set before training.
- **Baselines must be per-role.** Sharing a single group mean across planner and synthesizer turns re-couples them and defeats the point.
- **Role imbalance.** In a typical trajectory the Synthesizer sees more tokens than the Planner. Weight the two losses (paper uses inverse-token-count) or the Planner underfits.
- **Terminal reward still matters.** Without $R_{\text{outcome}}$, the two policies chase rubric proxies. The outcome term anchors both roles.

## Sources

- Paper: *IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis* — Wu, Yan, Lu, et al., Zhejiang U., 2026, [arXiv 2609.29444](https://arxiv.org/abs/2609.29444) — introduces RDPO and the Planner+Synthesizer paradigm; IterSynth-8B beats ≤8B ReAct agents by +4.2 average on five deep-search benchmarks.
- Paper: *DeepSeekMath* — Shao et al., 2024 — the GRPO base.
