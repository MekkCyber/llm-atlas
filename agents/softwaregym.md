# SoftwareGym
*Depth — environment synthesis for computer-use agent training.*

**TL;DR:** The bottleneck for computer-use agents is not model capacity, it is high-quality GUI task data. SoftwareGym *synthesizes entire desktop applications* and their task curricula, GUI-tests them, and collects verified trajectories — 182 synthetic apps across 12 categories at a 91.5% mean GUI-task pass rate. A model trained only on trajectories from these apps lifts OSWorld Pass@3 from 41.39% → 56.97%, with no human-written apps or demonstrations.

**Prereqs:** [grpo](../post-training/grpo.md), [rl-prompt-curation](../post-training/rl-prompt-curation.md)
**Related:** [rlvr](../post-training/rlvr.md)

---

## What it is

An automated pipeline — plus the resulting dataset and 120-task benchmark — for producing a *large, diverse training ground* for computer-use agents. Each generated artifact is a full desktop application: screens, state, interactions, plus a set of goal-conditioned tasks on top of it and automated tests that mark completion. The agent sees it as any other OS-level application.

## How it works

1. **App spec generation.** An LLM samples a category (file manager, calendar, text editor, …) and writes a spec for a plausible application.
2. **App implementation.** The spec is compiled to a working desktop app with a GUI and a programmatic test interface. State is persistent across sessions.
3. **Task + test generation.** For each app, the generator writes GUI tasks and corresponding automated tests. A task counts as valid only if the generated solution actually passes the generated test — this is the quality filter that lifts pass rate to 91.5%.
4. **Trajectory collection.** Trajectories are collected from either a solver model or human demos; verified ones go to the training set.

## Why it matters

Computer-use data is scarce and expensive to label (someone has to click and verify). If synthesizing the *environment* at this quality works, the agent-training compute curve shifts: scale the data by scaling environment generation, not human annotation. The +15.6-point gain on OSWorld Pass@3 from purely synthetic data is the signal that the approach is viable, not just a toy.

## Gotchas & tricks

- **Distribution shift.** Synthetic apps have cleaner UIs than real-world software. The 56.97% ceiling on OSWorld suggests mixing with real demos (cf. Work2Skill) is still needed for the tail.
- **Spec-to-app verification is the choke point.** Many generated specs produce apps that superficially work but have subtle bugs the test misses — those contaminate trajectories. The 91.5% pass rate reflects a hard filtering step, not raw generation quality.
- **Category coverage.** 12 categories is wide but not exhaustive; CRMs, data-entry apps, and legacy enterprise software are not represented and are where real GUI agents still fail.

## Sources

- Paper: *SoftwareGym: Scaling Computer-Use Agent Training with Diverse Synthesized Software Environments* — Jin, Wang, Kang et al., 2026 — [arXiv:2610.12611](https://arxiv.org/abs/2610.12611)
