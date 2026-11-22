# Skill-FT (AutoMaster)

*Depth — distill a teacher through a library of natural-language "skills" instead of through its unguided successes.*

**TL;DR:** Task success alone doesn't specify *which behaviors* a post-training run should reinforce — a teacher can succeed via shortcut verification, unsupported assumptions, or lucky guesses. Skill-FT steers the teacher at rollout time with a natural-language skill prompt, then SFTs a student on the resulting trajectories. AutoMaster is an LLM agent that *authors* that skill library, picks training data, and refines both from evaluation feedback. The skill prompt is kept in the training input even though it's dropped at eval.

**Prereqs:** [../rejection-sampling.md](../rejection-sampling.md), [../_post-training.md](../_post-training.md)
**Related:** [../../agents/README.md](../../agents/README.md), [README.md](README.md)

---

## What it is

A mid-point between vanilla distillation and RL. Vanilla SFT keeps whichever teacher trajectories happen to succeed; RL rewards whole trajectories with a scalar. Skill-FT instead specifies *how* the teacher should approach the task via text — "decompose before executing," "verify tool outputs before committing" — and only distills trajectories produced under that guidance.

Two moving parts:

- **Skill library** — a set of short natural-language instructions. Each skill describes a desired behavior pattern.
- **AutoMaster agent** — an LLM that (1) proposes new skills, (2) selects which prompts to train on, (3) scores student checkpoints on held-out evaluations, (4) revises the skill set and data mix on each round.

## How it works

Per round:

```
for each prompt q:
    skill = AutoMaster.select_skill(q)
    trajectory = teacher.rollout(skill || q)
    if filter(trajectory):
        dataset.add(input=(skill || q), target=trajectory)
student = SFT(dataset)
feedback = eval(student)
AutoMaster.update(skill_library, feedback)
```

Two non-obvious details:

1. **Teacher = same model.** The teacher and the student share weights; the only thing changing teacher behavior is the skill prompt.
2. **Keep the skill in the input.** The student is trained with `(skill || q) → trajectory`, but evaluated on `q` alone. Doing so still improves eval performance over dropping the skill at both train and eval time.

## Why it matters

Reframes autonomous post-training as **search over a natural-language skill library** rather than search over trajectories or weight updates. The skill prompt is a cheap, interpretable intervention and composes with rejection sampling: the filter still picks successful trajectories, but the trajectory *distribution* is reshaped by the skill.

Reported gains on Terminal, Search, and software-engineering tasks for 4B and 9B students, beating SFT on successful unguided teacher rollouts.

## Gotchas & tricks

- Keeping the skill in the training input but not at eval relies on the model treating the skill as partly-contextual framing — removing it entirely at train time collapses gains.
- The skill library can drift toward instructions that work only on the current student; AutoMaster's eval-feedback loop is what bounds this. Cap the library size and rotate out skills whose contribution drops.
- Skill prompts can leak task-specific hints; validate that gains don't disappear on genuinely held-out distributions.

## Sources

- Paper: *AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher* — Sun, Meng, Tang, Song, Zhao et al., 2026 — [arXiv:2611.26510](https://arxiv.org/abs/2611.26510).
