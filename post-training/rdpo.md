# RDPO — Role-Decoupled Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO-style RL algorithm for agents where the loop is split across **distinct roles** (e.g. Planner and Synthesizer). Instead of one blended reward over the whole trajectory, RDPO combines a **terminal outcome reward** with **turn-level rubric evaluations** and computes **role-specific advantages** — so a planning mistake doesn't credit the synthesizer and vice versa. Introduced by IterSynth for deep-search agents.

**Prereqs:** [grpo](grpo.md), [ppo](ppo.md), [_rl](_rl.md)
**Related:** [_rewards](_rewards.md), [rlvr](rlvr.md), [rl-prompt-curation](rl-prompt-curation.md), [sc-grpo](sc-grpo.md), [av-grpo](av-grpo.md)

---

## What it is

Deep-search agents (BrowseComp-style: decompose query → search → synthesize) implemented as ReAct suffer from **role coupling** — one policy handles planning, evidence use, and synthesis simultaneously — and from **context accumulation** — the growing history obscures useful information. IterSynth splits the loop into a Planner LLM (identify info needs) and a Synthesizer LLM (integrate evidence into a rolling summary). RDPO is the training algorithm that keeps their credit-assignment separate.

## How it works

For each query $q$, sample $G$ full agent trajectories $\{\tau_i\}$. Each trajectory contains alternating Planner turns $p_{i,t}$ and Synthesizer turns $s_{i,t}$.

**Rewards:**

- $r^\text{out}_i$: terminal outcome reward (did the final answer match ground truth?).
- $r^\text{rub}_{i,t,\text{role}}$: turn-level rubric reward for each turn, evaluated by a rubric-following judge specific to the role (e.g. planner rubric: "did this decomposition surface an actionable next query?"; synthesizer rubric: "did this summary integrate the new evidence?").

**Role-specific advantages:**

$$
r^{P}_i = r^\text{out}_i + \sum_t r^\text{rub}_{i,t,P}, \qquad r^{S}_i = r^\text{out}_i + \sum_t r^\text{rub}_{i,t,S}
$$

Group-normalize *separately per role*:

$$
A^{P}_i = \frac{r^{P}_i - \mu^{P}_G}{\sigma^{P}_G}, \qquad A^{S}_i = \frac{r^{S}_i - \mu^{S}_G}{\sigma^{S}_G}
$$

Update Planner tokens with $A^{P}$; Synthesizer tokens with $A^{S}$. KL to reference, clipped ratio, all standard.

## Why it matters

- **Cleaner credit assignment** than shared-reward RL on multi-role agents. When the synthesizer bombs a good plan, the planner isn't punished.
- **State-of-the-art small agent.** IterSynth-8B averages **50.7** across five long-horizon deep-search benchmarks, +4.2% over the strongest prior ≤8B ReAct-style agent.
- **Model-agnostic prompting pattern.** The Planner/Synthesizer split also boosts frontier proprietary models zero-shot.

## Gotchas & tricks

- Rubric judges must be reliable and role-specific — a generic "is this good?" LLM judge collapses back to shared-reward RL.
- Turn-level rewards can be noisy; weighting the outcome reward higher during early training is common.
- Works best when the roles have genuinely different objectives — for tasks where planning and synthesis are the same skill, coupling them may still be fine.

## Sources

- Paper: *IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis* — Wu et al., Zhejiang U. / China Mobile, 2026 — [arXiv:2609.29444](https://arxiv.org/abs/2609.29444).
- Base algorithm: [grpo](grpo.md) — DeepSeekMath, 2024.
