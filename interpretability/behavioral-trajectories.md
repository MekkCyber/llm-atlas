# Behavioral trajectories (agent interpretability)
*Depth — interpret coding agents by mapping their raw execution traces to a shared behavior vocabulary, then analyzing behavior as a distribution instead of a scalar score.*

**TL;DR:** Coding-agent benchmarks score binary task success, which hides how agents actually work. A rule-based parser over 143,209 trajectories from 125 agent configurations (117 models, SWE-rebench data) maps heterogeneous harness traces to a small shared vocabulary of actions (search repo, edit file, run tests). Parser labels agree with a six-model LLM consensus on a 500-event audit. Reveals that agents with similar solve rates have substantially different behavior, and that aggregate behavioral profiles can fingerprint the agent stack on unseen repos/languages.

**Prereqs:** [./README.md](./README.md)
**Related:** [../agents/README.md](../agents/README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Behavioral-level interpretability for agents: treat the agent's execution trace as the observable, and analyze the *distribution* of actions it takes rather than just whether the task succeeded. Complementary to activation-level interpretability on the underlying model.

## How it works

1. **Shared vocabulary.** Enumerate a small set of high-level behaviors: `search_repo`, `read_file`, `edit_file`, `run_tests`, `inspect_error`, `plan`, `done`. One vocabulary for all harnesses.
2. **Rule-based parser.** Regex + AST patterns that map each harness's native tool calls (grep / ripgrep / ls / patch / pytest / write_file / …) to the shared vocabulary. Parser is deterministic and auditable.
3. **Audit.** 500 events labeled by a six-model LLM consensus; the parser agrees. Gives a credible ground-truth sample without hand-annotation at scale.
4. **Analysis.** Each agent's trajectories become a distribution over the shared behaviors. Compare agents via distances between distributions, conditional on task or repo.

## Why it matters

Two findings open new ground:
- **Behavior ≠ outcome.** Two agents with identical solve rates can trade off search vs edit vs test very differently. Benchmarks that only report pass rates miss these differences.
- **Agent-stack fingerprinting.** Aggregate behavioral profiles — without using outcomes, token counts, or tool names — identify the deployed stack on disjoint repositories and programming languages. Task composition differences may partly explain the cross-repository results, but the signal is strong.
- **Scaffold matters.** Reasoning effort and scaffold design systematically shift behavior allocations. Supports treating scaffold as a first-class variable in agent design.

Opens the door to *behavioral RL* — rewarding agents for behavior patterns, not just outcomes — and to targeted ablations that change one behavior at a time.

## Gotchas & tricks

- **Parser coverage.** Any harness introducing a new tool class needs a parser rule. Report releases the parser and an annotated trajectory subset to extend.
- **Task composition confound.** Behavioral fingerprinting generalizes across repos; some of the signal may reflect which repos a stack tends to attempt, not pure behavior. Report controls for this partially.
- **Granularity.** A vocabulary too coarse hides interesting differences; too fine turns the analysis back into tool-name comparisons. Seven behaviors is empirically a good knee.

## Sources

- Paper: *Interpreting Coding Agents Through Their Behavioral Trajectories* — Badertdinov et al. (Nebius), 2026 — [arXiv:2611.17440](https://arxiv.org/abs/2611.17440)
- Data: SWE-rebench execution trajectories
