# JIT Harness Generation
*Depth — training a model whose output is an agent harness, synthesized just-in-time per task.*

**TL;DR:** Instead of hand-designing an agent scaffold once, train a *harness-generator* model that produces a task-adaptive harness (memory / planning / action / skill modules) on demand for any off-the-shelf agentic LLM. The harness is repaired live during execution and self-evolves by distilling performance signals from an archive of prior configurations.

**Prereqs:** [agent-harness.md](agent-harness.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [handoff-tax.md](handoff-tax.md)

---

## What it is

JIT-Agent (Zhang et al., 2026) is the first purpose-trained model whose *output* is an agent harness rather than a task solution. Given a task description, it emits a composable harness under the fixed four-module protocol (see [agent-harness.md](agent-harness.md)); the harness is then executed by a runtime driving an ordinary agentic LLM. The generator is trained to (a) customize a harness per task, (b) repair harnesses when execution destabilizes, and (c) self-evolve by distilling signal from an archive of past configurations.

## How it works

Three loops stack on top of each other:

1. **Task-conditioned synthesis.** The generator conditions on the task and emits a machine-parseable harness spec (memory policy, planner, action grammar, skill registry). The runtime loads it and starts executing.
2. **Live repair.** During execution, failure signals (invalid actions, timeouts, planner loops) feed back into the generator, which produces edits to the harness — swapping a memory policy, tightening the action grammar, adding a fallback skill — without restarting.
3. **Archive-driven self-evolution.** Every completed run's harness + reward becomes an entry in a growing archive. The generator is periodically retrained by distilling performance signals from that archive, so recurring failure modes get compiled into better default harness choices.

At training time, harness quality is a scalar reward derived from execution success on a task benchmark — the same shape as [RLVR](../post-training/rlvr.md), applied one level up (the object being rewarded is a harness, not a token sequence).

## Why it matters

Reported gains are large enough to matter for the "which frontier model do I pick" question. DeepSeek-V4-Flash + JIT-Agent surpasses GPT-5.6 on DeepSearchQA (+9.1) and OdysseyBench (+4.3); GLM-5.2 gains up to +20.2. Generated harnesses are competitive with mature runtimes such as OpenCode and Claude Code. The load-bearing implication: harness design is a compounding capability axis orthogonal to model scaling, and mature runtimes are prior art that can be learned from and improved on automatically for a specific task.

## Gotchas & tricks

- **Repair beats one-shot generation.** Without live repair, generated harnesses fail on distribution shift within a single run; the archive alone isn't enough.
- **Skill registry ≠ portable.** Skills that depend on host tooling need per-environment adaptation; JIT-Agent's transferability is strongest at the memory + planning + action level.
- **Reward-hacking risk.** A harness generator will find harnesses that game whatever completion criterion its training uses — evaluate against a decorrelated pass-rate signal (see [../evaluation/frontierchallenge.md](../evaluation/frontierchallenge.md) for a related "confidently-claimed completion" failure mode).

## Sources

- Paper: *JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution* — Zhang et al., 2026 — [arXiv:2608.25593](https://arxiv.org/abs/2608.25593)
