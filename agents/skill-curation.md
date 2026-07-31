# Skill Curation — One Policy, One Skill Doc, Two Objectives

*Depth — cross-task agentic RL where the policy alternates between solving tasks and editing a shared skill document.*

**TL;DR:** Agentic RL usually treats tasks as IID episodes and resets between them. Skill curation, introduced in SkillRise, threads a shared **skill document** through a sequence of related tasks. The same policy alternates between solving the current task and appending / revising the skill doc. Credit is decoupled: task-solving is supervised by the current-task outcome, curation by the *discounted downstream* outcomes. No separate skill extractor, no pipeline of multiple stages.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [README.md](./README.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/reasoning/cast.md](../post-training/reasoning/cast.md)

---

## What it is

A training pattern for LLM agents that face **families of related tasks** — tasks that share reusable solution structure but are formally distinct instances (different ALFWorld layouts, different WebShop products, different ScienceWorld goals). Instead of treating each task as isolated, the agent works through a *sequence* of related tasks and edits a shared skill document between them.

## How it works

- **One policy, two roles.** The same LLM alternates between:
  - *Solving mode:* read the current task, take actions, produce an answer / outcome.
  - *Curating mode:* read the skill document, edit it (add, delete, refine bullets) before it's passed to the next task.
- **Progressive sequences.** Tasks are organized into progressively harder sequences of related instances. Easier ones seed useful skills; harder ones benefit from them.
- **Decoupled credit assignment.** Two reward channels feed one policy:
  - Solving loss: standard RLVR / GRPO on the current task outcome.
  - Curation loss: a **discounted return** over the *downstream* tasks in the same sequence — an edit to the skill doc is credited by how much it helped the tasks that came after.
- **Shared parameters.** Both modes update the same policy, so the model learns not just how to solve but how to write skills that its future self will actually use.

## Why it matters

- **Skills accumulate.** Traditional agent RL wastes the transferable structure between related tasks by resetting per-episode. Skill curation preserves it in a lightweight, inspectable artifact.
- **Simpler than multi-stage skill pipelines.** Prior skill-learning work stacks extract → retrieve → execute pipelines; skill curation is a single-policy variant with lower runtime overhead and no offline pretraining of a separate extractor.
- **Test-time scaling across tasks.** Performance improves with *longer* sequences of related tasks even when each is attempted only once — evidence of genuine skill transfer, not repeat-sampling gains.
- **Curation trained cross-task transfers to same-task.** A doc curated across distinct tasks also helps repeated attempts on the same task; the effect isn't limited to the training regime.

Reported results: +2.3 to +8.5 pp Pass@1 over the strongest baseline on ALFWorld, WebShop, ScienceWorld.

## Gotchas & tricks

- **Discounted downstream credit is fragile.** If the sequence is short, the curation signal has little to grade edits on. Longer sequences give more signal but multiply rollout cost. Tune sequence length as a hyperparameter.
- **Skill doc bloat.** Without a length cap or explicit "delete" action, docs grow unboundedly and dilute the useful bullets. The curation policy must be able to *remove* content.
- **Sequence composition matters.** "Related but distinct" is doing work — completely IID tasks give no curation signal; near-duplicates give no learning signal. The sequences that work best in the paper are graded curricula, not random draws.
- **Interference with solving.** Curation adds tokens to the context and steals attention from the task; a badly-curated doc actively hurts. Weighting the two loss channels correctly (or annealing curation reward as the doc matures) is worth the tuning cost.
- **Not a memory system.** Skill curation is training-time; the doc is a training artifact, not a run-time memory store. Runtime memory (retrieval, cache) is complementary.

## Sources

- Paper: *SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution* — Yao et al., Zhejiang U. / NUS / SJTU / Meituan, 2026 — introduces the method. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Code: https://github.com/Within-yao/SkillRise
