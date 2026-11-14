# Decision Model Roles — Actor, Judge, Teacher
*Depth — the same decision model plays three different roles in agent training, and ranks on each role don't correlate.*

**TL;DR:** For a long-horizon agent, any LLM can be deployed in three roles: as an **actor** (picks the next step given the current observation), as a **judge** (scores a complete trajectory after the fact), or as a **teacher** (emits per-step rewards used to train a smaller policy with RL). The 2026 "Actor, Judge, or Teacher?" paper shows the three roles rank models very differently — a model that acts poorly on strict-order subgoal tasks can still be a strong judge and an excellent teacher, because judging sees the whole episode.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md), [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)

---

## What it is

A decomposition of "use an LLM for agent decisions" into three distinct jobs:

- **Actor:** $a_t = \mathrm{LLM}(o_{\le t}, h_{\le t})$ — observe state, pick next action. Must anticipate consequences without having seen them.
- **Judge:** $s = \mathrm{LLM}(\text{full trajectory } \tau)$ — see the complete episode, score it. Can retroactively credit the steps that moved the task forward.
- **Teacher:** $r_t = \mathrm{LLM}(\tau, t)$ — see the full trajectory, emit a **per-step reward** for each step $a_t$. These per-step rewards train a smaller actor via RL.

The paper's claim: these three roles have **different difficulty profiles**, so top models differ by role.

## How it works

### Why actor ≠ judge

On tasks with strict-order subgoals (ALFWorld, WebShop), the actor fails because it must *predict* which action advances a subgoal before seeing the result. The judge, seeing the whole episode, works backward from the known outcome: it knows subgoal $G_i$ got completed at step $t_i$, so steps around $t_i$ get credit, and false attempts can be identified as such.

In learning terms: the actor plays a hard prediction game; the judge plays a much easier labeling game.

### Teacher = judge-on-steroids

A teacher-role LLM works like a judge but emits a per-step reward vector instead of one episode score. In the simplest form:

$$
r_t = \mathrm{Teacher}(\tau, t) = \text{"fraction of subgoal progress attributable to step } t\text{"}
$$

The small-policy actor is then trained with these per-step rewards under standard policy-gradient / GRPO — the teacher replaces a hand-designed reward function or a classical PRM.

### What the paper measures

Rank 21 LLMs under each role on two long-horizon agent benchmarks. Correlation across roles is weak: the best actor is not the best judge, and the best teacher (measured by the terminal performance of the small policy it trained) can be a middle-of-pack actor and a strong judge.

## Why it matters

- Agent-RL researchers routinely pick the reward source by "which model has the best agent score," but that's the actor-role score. The correct criterion is teacher-role score, which is different.
- Judge-role LLMs are cheap to deploy: episode-level scoring at $O(\text{episode})$ cost, not per-step. For many setups that's good enough and side-steps PRM step-definition headaches.
- Generalizes the idea that reward models should be evaluated *as reward models* (teacher quality), not as classifiers of trajectory correctness (judge accuracy), when the point is to train.

## Gotchas & tricks

- Teacher-role quality only matters relative to the student policy you're training. A teacher that's great for a 1B student may be worse for a 7B student (reward granularity mismatch).
- Judge-role scores can be gamed by trajectory length — long rambling episodes read as "detailed" to a judge that doesn't normalize.
- If the actor and the teacher are the same model, the teacher's rewards inherit the actor's biases. Prefer a *different* model, or at minimum a different sampling temperature / prompt.

## Sources

- Paper: *Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents* — Zhu et al., 2026 — https://arxiv.org/abs/2611.17167
- Related: *Let's Verify Step by Step* — Lightman et al., 2023 — judge-role evaluation of PRMs.
