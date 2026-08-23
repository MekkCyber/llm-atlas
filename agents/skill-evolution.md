# Skill Evolution (SkillEvo)
*Depth — closing the loop on agent-skill evolution with multi-turn user simulation and a governance layer against skill regression.*

**TL;DR:** Agent *skills* — small task-specialized policies, prompts, or tool wrappers — are usually hand-authored or generated in a single LLM pass, with no feedback loop over the interactions they eventually cause. **SkillEvo** replaces single-turn evaluation with **multi-turn user simulation** that generates realistic feedback trajectories, and adds an **active governance layer** that blocks skill rewrites which would regress capability. Result across six cloud-service task categories: **+23.0** over self-reflection, **+15.4** over single-turn QA-based skill evolution.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [env-harness.md](env-harness.md)

---

## What it is

An evolutionary loop over agent skills with two novel components:

- **Multi-turn user simulation.** A dedicated simulator plays the role of the end user across multiple conversational rounds — the resulting interaction trace exposes multi-turn defects (misunderstanding recovery, context loss, incorrect tool sequencing) that single-turn evaluators miss entirely.
- **Governance layer.** A gate that inspects each proposed skill rewrite and rejects those that regress on held-out capability probes, blocking silent degradation across evolution rounds.

## How it works

1. **Rollout.** The current skill runs against the multi-turn simulator on a distribution of realistic user goals. Failure trajectories are aggregated.
2. **Propose.** An LLM rewrites the skill (prompt, tool schema, sub-policy, or heuristic) informed by the failure patterns.
3. **Govern.** The proposed rewrite runs on a probe set covering (a) known-good behaviors the current skill already handles, (b) trap cases the current skill just started passing, and (c) new failure modes surfaced this round. Regressions on (a) or (b) → reject.
4. **Commit.** Surviving rewrites replace the current skill; loop repeats.

## Why it matters

Skill / prompt / tool "evolution" is a hot subgenre of agent RL, and the field has quietly been reinventing the same broken pattern: iterate on a single-turn eval and watch multi-turn quality silently rot. SkillEvo names both problems (single-turn eval is the wrong signal; unguarded rewrites regress) and offers concrete fixes — echoing the same "close the loop, but only with a signal you trust" pattern EnvHarness applies at the environment level.

## Gotchas & tricks

- Governance probes are the failure surface: if the probe set doesn't cover the true skill contract, the governor waves through regressions on uncovered behaviors.
- Multi-turn simulators are LLMs; a weak simulator produces feedback the skill can trivially satisfy. Simulator capability must sit at or above the target user distribution.
- Cost per round grows with simulator turns × probe size — the paper's 6-category cloud-service setting is finite; scaling to open-domain skills demands smarter probe-set curation.

## Sources

- Paper: *SkillEvo: Self-Renewing Evolution Gradients from Multi-Turn Interaction Feedback* — Yan, Chen, Zhao et al., 2026 — [arXiv:2608.13120](https://arxiv.org/abs/2608.13120)
