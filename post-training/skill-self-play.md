# Skill Self-Play
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An interaction-driven LLM self-evolution recipe that co-trains three roles — a **proposer** that generates new tasks, a **solver** that attempts them, and a **skill controller** that curates which skills the proposer targets next — all with RL. Aims to break the standard self-play dilemma: more diverse tasks are harder to verify, while easily verifiable tasks converge to a narrow band.

**Prereqs:** [_rl.md](./_rl.md), [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [rl-prompt-curation.md](./rl-prompt-curation.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

A three-role self-play framework for LLM post-training. Rather than the standard proposer/solver pair, a **skill controller** maintains an evolving library of skills and decides which the proposer should target next. This adds a curriculum layer whose objective is *coverage* of skills rather than difficulty of tasks, so training doesn't collapse into a narrow band of easily-verifiable prompts.

## How it works

- **Proposer.** Given a target skill from the controller, an LLM policy generates a task specification.
- **Solver.** A (possibly co-trained) LLM attempts the task; verifiability of the outcome is required, but can be verifier-model or rule-based.
- **Skill controller.** Maintains a running estimate of solver competence per skill; picks under-covered skills for the next proposal round. Its own policy is trained to *raise* solver competence over the whole skill set, not any single skill.
- **RL updates.** All three heads receive gradient signal — solver from task reward, proposer from a mix of "task was solvable" and "task advanced the frontier," controller from the aggregate skill-coverage change.

## Why it matters

Frontier LLM RL is bottlenecked by the supply of verifiable prompts. Human-authored suites (math, code) are exhausted quickly; naive self-play collapses to easy tasks. A working co-evolving-skills loop pushes the *frontier* rather than polishing the interior, which is one of the missing ingredients for open-ended capability growth.

## Gotchas & tricks

- **Verifier quality is the ceiling.** If the proposer can generate tasks whose verifier is wrong, the solver optimizes garbage. Constrain proposal spaces to those with reliable verifiers.
- **Skill vocabulary is where reward hacking hides.** If the controller can invent trivial "skills," it will; either constrain the vocabulary or filter with a held-out competency probe.
- **Three roles ≠ three separate models.** Papers typically share weights across roles with role-conditioning prompts to keep total compute manageable.

## Sources

- Paper: *Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills* — Huang, Cheng, Liu, Chen, Liu, Ni, Zhou, Yang, Jiang, Zhou, Cheng, Jiang, Jiang (Qwen / Alibaba), 2026 — [arXiv:2607.22529](https://arxiv.org/abs/2607.22529).
