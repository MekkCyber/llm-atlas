# Rubric rewards
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of hand-writing a process reward model or relying on a coarse verifier, generate an **LLM-authored rubric per task/skill** (a list of item-level checks the trajectory must satisfy), then use rubric satisfaction as the RL reward. The rubric evolves itself across training — an LLM inspects failure trajectories and adds/refines rubric items. Fills the space between rule-based verifiers (too coarse) and human-labeled PRMs (too expensive). Introduced by SkillCoach for agentic skill-use.

**Prereqs:** [prm](prm.md), [_rewards](../_rewards.md)
**Related:** [grpo](../grpo.md), [rlvr](../rlvr.md)

---

## What it is

A **rubric** is a list of yes/no or graded checks specific to a task or skill:

- "Called `read_ticket` before `refund_ticket`."
- "Passed the amount in the `currency` field, not `amount_str`."
- "Ran the validation routine after modifying state."

For an agent trace, a rubric-judging LLM verifies each item, and the reward is a function of the fraction satisfied (typically weighted, with hard-required items). Unlike a terminal verifier (agent produced the right final action), a rubric decomposes what "correct" means into observable sub-checks along the trajectory.

Rubrics are: **(a)** generated automatically from a skill's description or specification, and **(b)** *evolved* by an LLM that looks at failure trajectories and identifies missing / mis-specified checks.

---

## How it works

**Generation.** Given a skill (a chunk of documentation, a workflow, an SOP), prompt an LLM to enumerate a rubric — what a correct trace must contain. Deduplicate against sibling skills (SkillCoach's setup emphasizes distractor-skill scenarios).

**Judging.** For each rollout, an LLM judge scores each rubric item against the trace and returns a per-item pass/fail plus an aggregate score. The aggregate becomes the reward for RL.

**Self-evolution.** Periodically, feed failure trajectories back to a rubric-editor LLM: "here's the current rubric, here are traces that scored high but were actually wrong / scored low but were actually right — refine." The rubric grows tighter and less gameable over time.

**Use as RL reward.** Rubric score is the per-rollout reward in GRPO / PPO. Because it's dense across the trajectory (many items), variance is lower than terminal verifier alone.

---

## Why it matters

- **Bridges the gap.** Terminal verifiers miss reward hacking (agent stumbles onto the right answer via a wrong path). Human PRMs are expensive. LLM-authored, self-evolving rubrics sit in the middle: cheap to generate, tighter than final-answer verification.
- **Agentic skill-use is the natural fit.** Modern LLM agents call skills / tools / MCP endpoints. Rubrics are a natural way to encode "correct use" of each skill — the same shape as human-authored playbooks or QA checklists.
- **Interpretable rewards.** A rubric is human-readable. When RL blows up, you can look at *which items* the policy started gaming.
- **Empirical.** SkillCoach reports rubric-based RL improves adherence and downstream task success across their agentic skill-use suite versus terminal-verifier-only training.

---

## Gotchas & tricks

- **LLM-judge cost.** Every rollout runs the rubric judge — an LLM per item or per rubric. For $G = 16$ GRPO rollouts, this is a real bill. Batch across items; cache rubrics; reserve the expensive judge for rubric evolution and use a smaller judge for scoring.
- **Rubric hacking.** Even LLM-authored rubrics can be gamed if items are shallow ("mentions the word `refund`"). The self-evolution loop catches this if it sees enough failure examples; without it, rubrics degrade over training.
- **Skill-scope mismatch.** Rubrics designed at the skill level don't generalize to composed multi-skill tasks. Compose rubric items across skills or train a rubric selector.
- **Overlap with PRM.** A rubric applied per step is a discrete PRM; a rubric applied per trajectory is a rich ORM. The novel piece is the *self-evolving* authoring, not the shape of the reward.
- **Judge model matters.** A weak judge produces noisy rewards. Use a judge at least as capable as the policy for the scoring step.

---

## Sources

- Paper: *SkillCoach: Self-Evolving Rubrics for Evaluating and Enhancing Agentic Skill-Use* — Zhu et al., 2026 — [arXiv:2607.01874](https://arxiv.org/abs/2607.01874).
- Related: [prm](prm.md) — the human-labeled precursor.
