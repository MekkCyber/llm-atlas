# SkillCoach

*Depth — a four-axis, self-evolving rubric for evaluating and training LLM agents on skill use.*

**TL;DR:** Modern LLM agents are increasingly built on a "skills" layer — reusable SOPs, tool workflows, validation routines, scripts. SkillCoach evaluates agent trajectories along four axes — **skill selection**, **skill following**, **skill composition**, and **skill-grounded reflection** — using a judge model whose rubric *evolves* from observed rollouts rather than being frozen at write time. External verification (did the task actually succeed?) stays separate so the rubric measures process, not outcome.

**Prereqs:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../agents/_agent-memory.md](../agents/_agent-memory.md)

---

## What it is

A judge-model evaluation harness for agents that pick, follow, and compose reusable skills. Rather than a static rubric prompt, the rubric is a live artifact updated over batches of rollouts: the judge is shown recent trajectories, the failures it missed, and refines its criteria. External verification (did the tool succeed, did the code run, did the ticket close) is scored in a separate lane; SkillCoach itself is *about process quality*.

## How it works

### The four axes

1. **Skill selection** — did the agent pick the right skill for the situation? A trajectory can complete the task while using a wildly wrong skill; that's a selection failure.
2. **Skill following** — given the chosen skill, did the agent follow its steps? Skipped preconditions, out-of-order calls, missing validation are following failures.
3. **Skill composition** — for multi-skill trajectories, did the composition make sense? Redundant chains, missing hand-offs, and skill sequencing errors surface here.
4. **Skill-grounded reflection** — after failures, did the agent reason *about the skill* (misread precondition, unavailable tool) rather than about the task in the abstract?

### Rubric evolution

A batch of rollouts is scored under the current rubric. A meta-judge compares the rubric's calls with ground-truth outcomes (and human spot checks) and proposes edits — new failure categories, tightened criteria for existing axes, removed rules that never fire. The rubric itself is versioned; every training run cites which rubric version was used.

### Uses beyond eval

Because the rubric is fine-grained, its outputs double as **training-data selection signal**: rollouts that score high on all four axes are better SFT/RL data than rollouts filtered only by outcome. This makes SkillCoach a joint eval + curation tool for agentic RL pipelines.

## Why it matters

- **Judge-LM evaluations were stuck.** Static rubrics go stale as agents get stronger; free-form judges drift. Rubric evolution against ground truth is the middle path.
- **Process ≠ outcome.** A trajectory that succeeds by luck (or with terrible skill hygiene) will train future agents to be lucky, not competent. Process rubrics catch this before it becomes a training-set problem.
- **Skills are the emerging agentic abstraction.** As skills become the standard unit of agent capability, an eval that speaks the same vocabulary is what teams need.

## Gotchas & tricks

- **Keep verification separate.** If the rubric quietly rewards outcome, it collapses to an ORM. Enforce the split at the judge prompt level.
- **Rubric drift is real.** Evolution loops can chase noise — freeze the rubric periodically and re-evaluate historical trajectories to detect regressions.
- **Composition axis is the hardest to score.** Ground-truth signal for "was this the right composition?" is thin. Expect this axis to lag the other three in reliability.
- **Judge cost.** A four-axis rubric with an evolved prompt is expensive per trajectory. Sample or pre-filter with a cheap first-pass rubric.

## Sources

- Paper: *SkillCoach: Self-Evolving Rubrics for Evaluating and Enhancing Agentic Skill-Use* — Zhu et al., HKUST(GZ) / JD.COM, 2026 — [arXiv:2607.01874](https://arxiv.org/abs/2607.01874).
