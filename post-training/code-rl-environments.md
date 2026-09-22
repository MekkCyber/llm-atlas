# Code as an RL environment
*Depth — turning arbitrary source code into verifiable RL environments for coding agents.*

**TL;DR:** Coding-agent RL needs many *verifiable* tasks (a prompt + tests + a gold solution). Traditional pipelines mine tasks from repository *artifacts* (issues, PRs, commit history), which caps the supply and skews toward bug-fix / issue-repair. A code-as-environment pipeline instead treats the *implemented functionality itself* as the task source: an authoring agent reads a code unit, invents a behavioral spec, writes execution-grounded tests by running the original code, then filters candidates by whether repeated solution rollouts pass. CodeMidas (Ye et al., 2026) is the canonical instantiation; 5,545 tasks from 3,185 repos across 23 languages produced measurable GRPO gains on MiMo-V2.5.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [rl-prompt-curation.md](./rl-prompt-curation.md) · [rejection-sampling.md](./rejection-sampling.md)

---

## What it is

An automated pipeline that converts a corpus of open-source repositories into RL-ready `(prompt, tests, oracle solution)` triples without human labeling. It sidesteps the "issue supply" bottleneck of prior work (SWE-Gym, DeepSWE-style pipelines) by taking source code, not repository events, as the primary signal.

## How it works

The pipeline spends agentic compute at every stage of environment construction:

1. **Explore.** An agent traverses the repo, identifies self-contained units of implemented functionality, and drafts a natural-language behavioral spec describing what the code does (inputs → outputs, side effects, invariants).
2. **Ground.** The agent constructs tests by *executing the original code* on generated inputs and capturing its outputs — tests are grounded in ground-truth behavior, not in prose reasoning about it.
3. **Validate.** Candidate tasks are filtered by two checks: the tests must actually run and pass on the source code (sanity), and the task must be neither trivially nor impossibly solved by repeated rollouts of a solver model (solvability filter, in the spirit of [rejection-sampling.md](./rejection-sampling.md)).
4. **Assemble.** Surviving triples become an RL environment. Rewards are execution-based (tests pass / fail), so training is [RLVR-style](./rlvr.md).

## Why it matters

- **Environment scale is decoupled from repository events.** Any repo — even ones with zero issues or a locked commit history — becomes an RL source. CodeMidas mines 3,185 codebases across 23 languages / 15 domains, a diversity level that issue-mining can't reach.
- **Reward is free and reliable.** Because tests come from executing the original code, they define the correct behavior by construction. No hand-written oracle, no learned reward model to overfit.
- **Task shape is controllable.** Choosing which code units to spec (functions vs. modules vs. whole programs) lets the pipeline hit issue repair, whole-program construction, and terminal work with the same primitives. CodeMidas reports +11.7% (DeepSWE), +17% (ProgramBench), +8.5% (Terminal-Bench v2.1) after GRPO on the produced tasks.

## Gotchas & tricks

- **Filter aggressively.** Repeated-rollout pass rates are essential — both perfect (learn nothing) and impossible (no learning signal) tasks must be dropped. The paper's ablation confirms task-quality mass, not task count, drives gains.
- **Spec drift.** Auto-authored specs can under-specify implicit invariants ("must handle empty input"). Executing the original code catches many but not all. Trajectory analysis in the paper shows RL-trained agents actively probe for such edge cases.
- **Language and domain balance matter.** Naive mining biases toward JS/Python because those repos dominate. Explicit stratification by language and technical domain avoids overfitting to popular stacks.
- **Compute up front, not at RL time.** The authoring pipeline is expensive (many LLM calls per accepted task), but this cost is amortized: once built, the environment supports arbitrarily many RL runs. In production this looks more like data preparation than training.

## Sources

- Paper: *CodeMidas: Scaling Agentic Coding RL Environments from Code Itself* — Ye et al., Xiaomi MiMo team, 2026 — [arXiv:2609.22068](https://arxiv.org/abs/2609.22068).
- Related: DeepSWE (issue-mining baseline), Terminal-Bench v2.1 (evaluation).
