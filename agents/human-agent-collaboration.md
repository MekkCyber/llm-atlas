# Human–Agent Collaboration
*Depth — a five-dimensional framework for evaluating how well an LLM agent works *with* a human during execution, not just how well it finishes alone.*

**TL;DR:** Most agent evals score only task completion. RealCLI (Liu et al., 2026) proposes a **separate axis of agent quality**: how an agent handles mid-run user input — clarifications, corrections, constraint changes, authorization decisions. The paper formalizes **five dimensions** of collaboration — steerability, trust calibration, safety-boundary management, goal-state maintenance, and transparent coordination — and runs frontier agents on 100+ CLI tasks in two modes: full-spec-upfront and interactive-user. Top configuration reaches only **49.2% average score** and peak joint-collaboration performance is **37.6%**. Task success and collaboration quality are measurably different signals.

**Prereqs:** [computer-use-agents.md](computer-use-agents.md)
**Related:** [../evaluation/cli-agent-bench.md](../evaluation/cli-agent-bench.md), [../evaluation/_agent-benchmarks.md](../evaluation/_agent-benchmarks.md), [../safety/safety-case.md](../safety/safety-case.md)

---

## What it is

An agent run is normally graded on a single scalar: did the final state match the ground truth? That hides a lot. Two agents can both complete the task while behaving very differently with their user:

- One asks clarifying questions when a spec is ambiguous. The other guesses.
- One accepts a constraint change mid-run. The other ignores it.
- One asks before `rm -rf`. The other doesn't.

**Human–agent collaboration** is the quality axis that measures how well the agent participates in a working relationship with the user *during* execution.

### The RealCLI five dimensions

| Dimension | Question answered |
| --- | --- |
| **Steerability** | Does the agent adjust when the user changes a constraint mid-run? |
| **Trust calibration** | Does the agent accurately convey its own uncertainty / confidence? |
| **Safety-boundary management** | Does the agent stop and ask before destructive or ambiguous-authorization actions? |
| **Goal-state maintenance** | Does the agent remember and respect the original goal across many turns and new info? |
| **Transparent coordination** | Does the agent explain what it is doing and why, in terms the user can act on? |

Each is scored per interaction, independently of task completion.

---

## How it works

### Benchmark design (RealCLI)

Each task appears in two modes:

- **Full-spec mode.** All information the agent needs is in the opening prompt.
- **Collaboration mode.** The opening prompt is intentionally incomplete; a *scripted human user* supplies missing info, mid-run corrections, constraint changes, and authorization decisions.

The same tasks appear in both modes. Comparing them isolates collaboration quality from task capability.

### Scoring

```
score_task_success        = outcome checker over final state
score_collaboration (× 5) = per-dimension rubric graded by a judge model + human spot-checks
score_joint               = combined via a weighted rubric
```

Importantly, the *joint* score is **not** the product of task success and collaboration — a task can succeed with bad collaboration or fail with good collaboration. Both numbers are reported.

### Scripted-user agent

The user is itself an LLM, prompted to behave like a specific persona with specific information to reveal. It paraphrases the task, injects corrections, and answers the agent's questions in character. This is the same architecture pattern used in [persona-cua-bench.md](../evaluation/persona-cua-bench.md) and in TauBench, so results across the three benchmarks are commensurable in spirit.

---

## Why it matters

- **Task success is not the whole picture.** On RealCLI's frontier-agent configuration, peak joint-collaboration is **37.6%** — much lower than peak task completion. There is a real gap, and single-score benchmarks hide it.
- **Collaboration failures are safety-adjacent.** "Safety-boundary management" overlaps with the safety agenda: an agent that proceeds through `rm -rf /` without asking is both failing collaboration and failing a deployment-safety case.
- **Operationalizes a long-standing complaint.** Users intuitively know when an agent "doesn't listen" or "doesn't check in." RealCLI makes that into five measurable rubrics.
- **Composable with other agent evals.** Combine with outcome-based benchmarks (SWE-bench, OSWorld) to score the same model on both axes.

---

## Gotchas & tricks

- **Judge-model bias.** The collaboration dimensions are rubric-graded by a judge LLM; biases of the judge leak into the scores. RealCLI reports human spot-checks on a subset to calibrate.
- **Dimensions are not independent.** Trust calibration and transparent coordination overlap in practice. Factor your own reporting carefully.
- **Scripted-user agents are a floor, not a ceiling.** Real users are messier. Add real-user studies on top of benchmarks for production decisions.
- **Collaboration ≠ chattiness.** An agent that asks a clarifying question every step scores low on autonomy. The rubric rewards *appropriate* collaboration — ask when needed, act when confident.
- **Watch for Goodharting.** Once the five dimensions are known, agents can be prompted to game them (e.g., always ask one clarifying question regardless of need). Rotate rubric phrasings in deployment.

---

## Sources

- Paper: *Toward a Unified Evaluation of Human–Agent Collaboration for CLI Agents (RealCLI)* — Liu et al., IQuest / Beihang / Renmin University, 2026, [arXiv 2610.22507](https://arxiv.org/abs/2610.22507).
- Related: TauBench (2024) — early scripted-user tool-benchmark pattern.
- Related: [../evaluation/cli-agent-bench.md](../evaluation/cli-agent-bench.md), [../evaluation/_agent-benchmarks.md](../evaluation/_agent-benchmarks.md).
- Related: [../safety/safety-case.md](../safety/safety-case.md) — safety-boundary-management overlaps.
