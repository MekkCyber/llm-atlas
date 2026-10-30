# RealCLI
*Depth — paired evaluation of task execution and human–agent collaboration on realistic CLI tasks.*

**TL;DR:** **RealCLI** (Liu et al., IQuest / Beihang / Renmin University, 2026) is a benchmark that scores CLI agents on **two orthogonal axes**: can they complete the task, and how do they *collaborate with the user* while doing so? Each task appears in two modes: full-spec-upfront and interactive-with-scripted-user. Top frontier configuration reaches **49.2% average score** and peak joint-collaboration performance is **37.6%**. The headline finding: task success and collaboration quality diverge — agents that complete tasks still fail on steerability, trust calibration, and safety-boundary management.

**Prereqs:** *(none)*
**Related:** [../agents/human-agent-collaboration.md](../agents/human-agent-collaboration.md), [../agents/computer-use-agents.md](../agents/computer-use-agents.md), [_agent-benchmarks.md](_agent-benchmarks.md)

---

## What it is

A benchmark for **CLI-based agents** (shell, dev tooling, file ops) with two components:

- A **task suite** of realistic command-line tasks with ground-truth final-state checks.
- A **scripted-user agent** that supplies missing info, corrections, and authorization decisions in character during execution.

Each task is run in two modes:

| Mode | User participation | What's measured |
| --- | --- | --- |
| **Full-spec** | none during execution | task success alone |
| **Collaboration** | scripted user supplies info / corrections mid-run | task success + 5-dim collaboration score |

### Scoring

For every (task, model) pair:

```
task_score     = outcome checker on final state
collab_scores  = {steerability, trust_calibration, safety_boundary,
                  goal_state_maintenance, transparent_coordination}
joint_score    = weighted combination graded by judge + human spot-check
```

See [../agents/human-agent-collaboration.md](../agents/human-agent-collaboration.md) for the five dimensions.

---

## How it works

### Task construction

Tasks are sourced from realistic CLI usage patterns: project setup, log analysis, debugging, file reorganization, dependency management, git operations. Each has:

- A ground-truth final file-system state.
- A natural-language instruction (ranging from fully specified to deliberately underspecified).
- A scripted-user profile: what information the user has, what they will volunteer, what they'll refuse to answer.

### Scripted user

An LLM prompted to behave like a specific user:

- Rewrites the task in natural style.
- Answers clarifying questions **in character**, including sometimes saying "I don't know" or "that depends."
- Introduces constraint changes mid-run ("actually, keep the old logs").
- Grants or denies authorization for ambiguous destructive ops.

The user LLM is held constant across evaluated agents to isolate agent behavior.

### Grading

The 5 collaboration dimensions are graded via a judge LLM with a rubric per dimension. A sample of runs is also graded by human spot-checkers to measure judge accuracy. Correlation between judge and human is reported.

---

## Why it matters

- **Benchmarks the collaboration axis for CLI agents, which was previously only visible anecdotally.**
- **Headline numbers show a real gap.** Peak joint-collaboration is **37.6%**, well below task-only ceiling. The two signals are not interchangeable.
- **CLI is the right domain for first-measurement.** Tasks are complex, ambiguous, and often destructive — safety-boundary management is a real concern, not a hypothetical.
- **Composable with existing agent evals.** Combine with SWE-bench, Terminal-Bench, TauBench for a multi-axis picture.

---

## Gotchas & tricks

- **Judge-rubric bias.** The 5-dim scores reflect the rubric and the judge. The paper reports human-judge correlation; take the numbers as calibrated-but-not-perfect.
- **Scripted users are a floor.** Real users are noisier, more inconsistent, and sometimes adversarial. Treat the benchmark as a lower bound on real-world collaboration difficulty.
- **Task completion can shadow collaboration.** A model with 80% task completion and 20% collaboration might still ship for high-throughput use cases where destructive actions are rare. Report both.
- **Collaboration ≠ chatty.** Over-asking tanks the "autonomy" sub-rubric; under-asking tanks "safety-boundary management." The reward surface is non-trivial.
- **CLI-specific.** Results don't transfer verbatim to browser or GUI agents. PersonaCUABench and TauBench cover adjacent slices of the agent-benchmark space.

---

## Sources

- Paper: *Toward a Unified Evaluation of Human–Agent Collaboration for CLI Agents* — Liu et al., IQuest / Beihang / Renmin University, 2026, [arXiv 2610.22507](https://arxiv.org/abs/2610.22507).
- Related: Terminal-Bench (2024) — outcome-only CLI benchmark predecessor.
- Related: [../agents/human-agent-collaboration.md](../agents/human-agent-collaboration.md) — the five-dimension framework.
- Related: [_agent-benchmarks.md](_agent-benchmarks.md) — the broader agent-benchmark landscape.
