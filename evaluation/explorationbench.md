# ExplorationBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark that measures an agent's **exploration** — hypothesize, experiment, iterate — as opposed to its recall from pretraining. Built on **verifiable Alien Worlds** whose rules are executable (so answers are exactly checkable) and whose contents *conflict with familiar knowledge* (so pattern-matching on pretraining does not help). Two sandboxes: **AlienCode** (31 discovery targets, 70 tasks) and **AlienLogic** (24 targets, 70 tasks), each with a deliberately flawed manual and a tool-call schema.

**Prereqs:** [agents README](../agents/README.md)
**Related:** [evaluation README](README.md), [wrop](wrop.md)

---

## What it is

Evaluating scientific-style exploration for AI agents is hard for two reasons: (1) verifying whether a hypothesis is "genuinely new" is subjective, and (2) it is hard to tell if the agent discovered the rule or just recalled it from training data. ExplorationBench turns both into a concrete framework by *constructing worlds* whose rules are novel by design and executable for automated grading.

## How it works

- **Alien Worlds.** Small formal environments (a synthetic programming language in AlienCode; a synthetic logic system in AlienLogic) with rules that intentionally diverge from familiar programming and logic conventions.
- **Flawed manual + tool-call schema.** The agent starts with a manual that contains known errors, and a schema for issuing tool calls into the sandbox to run experiments.
- **Discovery targets vs. tasks.** Each sandbox is structured as *discovery targets* (rules to figure out) leading into *held-out tasks* solvable only if the discovered rules are correct. This separates the exploration phase from the exploitation phase.
- **Grading.** All answers are executable, so grading is exact — no LLM judge.

## Why it matters

- **Cleanly disentangles exploration from recall.** By construction, pretraining knowledge cannot solve the tasks.
- **Concrete measurement of "exploration collapse."** The paper finds that continued exploration can stall or even *reverse* prior gains — a failure mode frontier-lab agent demos rarely show but that shows up here.
- **A benchmark aimed at the discovery-science story** frontier labs tell about their agent products.

## Gotchas & tricks

- Systems with strong tool-use loops (Claude Code, Codex, etc.) do best; agents with weak function-calling are penalized regardless of reasoning ability.
- Trajectory variance is high — evaluate with multiple seeds per task and report distributions, not just means.
- Because environments are executable, contamination is easy to check: any leakage of the exact rule strings should be caught before evaluation.
- The "flawed manual" is a load-bearing part of the setup: if the agent trusts it without experimentation, it fails; the benchmark is testing *whether it experiments at all*.
- Reward for AlienLogic tasks tends to be sparser than AlienCode — pair scores across sandboxes rather than reporting one.

## Sources

- Paper: *ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds* — Zhang et al., Fudan / Tencent, 2026 — [arXiv:2609.30199](https://arxiv.org/abs/2609.30199).
