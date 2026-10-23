# Socratic Skill Learning (SSL)
*Depth — teaching an LLM agent reusable skills by interleaving Educator guidance with Student practice.*

**TL;DR:** A two-agent training loop for *skill learning*: an **Educator** agent guides a **Student** agent through practice on a task; the Student runs the task, emits "issues" when it gets stuck, and writes back a reusable **Skill document** when done. The Educator only intervenes on issue-flagged sub-tasks, which cuts supervision cost. Skills are inspected and loaded on later tasks that share sub-skills. The paper also introduces **SkillTransferBench** to measure skill transfer.

**Prereqs:** [_rl](../post-training/_rl.md), [_post-training](../post-training/_post-training.md)
**Related:** [rl-prompt-curation](../post-training/rl-prompt-curation.md), [rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

LLM agents today sit between two extremes for "learned behavior":

- **Prompt-engineered skills** — hard-coded instructions in the system prompt. Static, limited, not learned.
- **Full RL** — the agent updates weights on reward. Expensive, no reuse across sessions.

Socratic Skill Learning is a middle path: the agent *writes* its skills as text artifacts that persist across sessions and are reloaded on related tasks. Training is interleaved teaching + practice: the Educator intervenes only where needed; the Student does the rest.

A **Skill** is a short, named text document (name, when to use, procedure, pitfalls) that the Student authored while solving a task. It is retrieved at future task start by a lightweight matcher.

---

## How it works

The loop per training task:

1. **Student attempts the task.** Loads any relevant previously-written skills.
2. **Student emits issues.** When the Student is uncertain or hits a sub-task it can't solve, it flags a structured "issue" naming the sub-task.
3. **Educator responds *only to flagged issues*.** For each issue, the Educator provides a focused hint, correction, or demonstration — not the full solution.
4. **Student resumes.** With the hint, the Student completes the task.
5. **Student writes the Skill.** After success, the Student authors a Skill document capturing the reusable sub-procedure.

The **Issue-Guided SSL** variant limits Educator calls to the issues the Student actually raised, instead of post-hoc Educator review of every step. This is the lever that reduces expert supervision cost.

Skills are evaluated for **transfer**: a held-out task shares a sub-skill with a training task iff the same Skill document loaded on the held-out task raises success rate.

---

## Why it matters

Three reasons:

1. **Persistent, inspectable agent memory.** Skills are text, which humans can audit and edit — unlike learned weight deltas or opaque retrieval banks.
2. **Supervision cost scales with difficulty, not with task count.** The Student handles the common cases alone; the Educator pays only on hard sub-tasks.
3. **A benchmark for transfer.** SkillTransferBench measures the thing that matters — do skills written on task A help on related task B — not just single-task reward.

The approach sidesteps the recurring problem that pure RL on agents produces non-transferable policies (good on the training distribution, brittle on anything else), by externalizing *what was learned* as text.

---

## Gotchas & tricks

- **Issue granularity.** If the Student flags everything, the Educator is called constantly and the cost benefit vanishes. If it flags nothing, it writes bad skills. Thresholding issue emission by Student self-confidence is the practical knob.
- **Skill retrieval matters as much as skill writing.** A great Skill nobody loads at the right moment contributes nothing. The retriever (even a simple sub-task-name matcher) is a bottleneck.
- **Skill rot.** Early-training skills can be wrong; the loop needs a revision mechanism (Student can supersede its own prior skills on later encounters).
- **Not model-agnostic.** The Student needs to be a reasonably capable instruction-follower before SSL starts — bootstrap from an SFT'd checkpoint, not from a raw base.

---

## Sources

- Paper: *Socratic Skill Learning: Interleaving Teaching and Practice for Transferable Agent Skills* — Han Li, Yifan Yao, Xinping Lei, et al., Nanjing University / Alibaba, 2026 — [arXiv:2610.17860](https://arxiv.org/abs/2610.17860).
- Benchmark: SkillTransferBench — introduced in the same paper.
