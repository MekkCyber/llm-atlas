# Trajectory2Gym — decision-grounded compositional data for tool-use agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A data recipe for SFT of tool-use agents that treats raw successful trajectories as insufficient. Trajectory2Gym rebuilds each training task as an executable Gym-style environment with explicit tool-dependency graphs, synthesizes new rollouts that target under-covered dependency structures, and masks tokens the agent could not have grounded in its visible observation history. Trained on this curated mix, five Qwen3 model sizes improve on Tau1, Tau2, and BFCL — most on stateful, multi-turn benchmarks.

**Prereqs:** [../agents/README.md](../agents/README.md), [../post-training/README.md](../post-training/README.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Modern tool-use agents are usually SFT'd on filtered successful trajectories: run a strong model/harness against a task, keep the runs that pass, imitate them. The paper argues this misses two things. First, success is cheap: a trajectory can land on the right final state without having exercised the task's real sub-dependencies (e.g. always using the same tool when several are needed in general). Second, imitation blindly copies actions whose rationale wasn't in the agent's visible observation history — "lucky" actions the agent couldn't have decided in context.

Trajectory2Gym is a two-part fix: generate *compositional* interactions that cover the dependency structure of each task, and supervise only *decision-grounded* tokens.

## How it works

1. **Rebuild each task as a Gym environment.** From the dataset's task spec, extract an executable environment with the task's tool interfaces and a dependency graph over sub-steps.
2. **Synthesize compositional rollouts.** Sample trajectories that target under-covered combinations of dependency edges — not just paths that lead to success, but paths that exercise sub-dependencies the raw data misses.
3. **Decision-grounded supervision.** For each candidate training token, check whether the action it labels is justified from the agent's visible history (observations + prior actions). Mask out tokens whose correctness isn't entailed by what the agent could see at that step.
4. **Allocate data by structural coverage.** Balance the mix so dependency-graph coverage is roughly uniform, rather than reweighting purely by raw task success.

The output is a cleaner, structurally balanced SFT dataset. The downstream training recipe is standard SFT.

## Why it matters

Tool-use agent benchmarks (Tau1/Tau2, BFCL) are dominated by stateful, multi-turn tasks where dependency structure matters more than final-answer correctness. Trajectory2Gym gives a reproducible recipe for data curation when success alone is a leaky quality signal, and reports gains across the Qwen3 family (0.5B → 32B) — concentrated exactly where the structure-of-task argument predicts.

## Gotchas & tricks

- **Need an executable environment.** The recipe assumes you can rebuild each task into something runnable. For dataset entries that are only free-form conversations, this is non-trivial.
- **Grounding filter is strict.** The paper masks tokens whose rationale isn't visible. Over-aggressive masking can starve the SFT signal; the balance depends on how dense observations are.
- **Composes with RL, not a substitute.** Trajectory2Gym is an SFT-stage data recipe. Agentic RL on top is still useful; the paper doesn't replace it.

## Sources

- Paper: *Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents* — Wen et al., LINs Lab (Westlake), 2026 — [arXiv:2610.33568](https://arxiv.org/abs/2610.33568).
