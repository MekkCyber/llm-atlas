# SWE-Bench (and Terminal-Bench)
*Depth — the standard evaluation stack for coding agents that patch real repositories.*

**TL;DR:** SWE-Bench is the canonical eval for autonomous coding agents: given a real GitHub issue, an agent must produce a patch that makes the project's hidden test suite pass. **SWE-Bench Verified** is the human-audited 500-task subset most modern harness papers report on. **Terminal-Bench 2.x** is the sister benchmark that measures shell-driven task completion (setup, install, run). Together they form the reference eval for any coding-harness or coding-agent claim.

**Prereqs:** [../agents/agent-harness.md](../agents/agent-harness.md)
**Related:** [livecodebench.md](livecodebench.md), [../agents/_context-management.md](../agents/_context-management.md)

---

## What it is

SWE-Bench was introduced by Jimenez et al. (2023) as 2,294 real GitHub issue-and-fix pairs drawn from popular Python repositories. Each task gives the agent (a) the repository state at the parent commit and (b) the issue text. The agent produces a patch; the task passes iff the repository's own hidden test suite passes with the patch applied.

**SWE-Bench Verified** (OpenAI, 2024) is a 500-task subset filtered by human reviewers to remove ambiguous, over-specified, or environment-dependent tasks. It is the modern reporting standard.

**Terminal-Bench** (Stanford / TIL 2024, updated to 2.1) covers a complementary axis: shell tasks over Linux environments (build a project from source, set up a service, debug a failing pipeline). Where SWE-Bench measures patch quality, Terminal-Bench measures command-line agency.

---

## How it works

Per task:

```
issue text ─┐
             ├─► agent harness ─► patch ─┐
repo state ─┘                             ├─► hidden test suite ─► pass / fail
                                           │
                                gold state ┘
```

For SWE-Bench Verified, tests must (a) pass on the gold patch, and (b) fail without any patch. This filters flaky and always-passing tests. The agent has no direct access to the tests.

Terminal-Bench scores a full trajectory against a fixed goal condition (a check script that verifies the environment state). Recent papers (Fan et al. 2026, "Don't Mask the Environment" 2026) use both.

---

## Why it matters

- **Correctness signal is verifiable.** The patch either makes the tests pass or it doesn't — no LLM-as-judge.
- **Realistic scope.** Real repositories, real bugs, real dependencies — closer to production than function-level benchmarks (HumanEval, MBPP).
- **Harness discrimination.** Because the model is fixed, differences between agents come almost entirely from harness design. This makes SWE-Bench Verified the primary battleground for harness ablations (Fan et al. 2026 use it for 176 settings).
- **Terminal-Bench catches bash-native failures.** SWE-Bench-only evaluations miss agents that struggle with environment setup; Terminal-Bench 2.x closes that gap.

---

## Gotchas & tricks

- **Contamination risk.** The underlying repos and issues are public and pre-training-visible. Verified re-runs the tests on filtered tasks but the model may still have "seen" a fix. Some papers report a contamination-audited subset.
- **Env-setup dominates cost.** For agents that must set up the repo before running tests, the setup phase can account for most of the tokens; harness efficiency matters as much as reasoning.
- **Pass@k reporting is inconsistent.** Some papers report Pass@1 with a fixed seed, others report Pass@k with sampling. Compare carefully.
- **Verified vs. full.** Numbers on full SWE-Bench are systematically lower than on Verified; conflating the two inflates claimed scores.

---

## Sources

- Paper: *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* — Jimenez et al., 2023 — [arXiv:2310.06770](https://arxiv.org/abs/2310.06770).
- OpenAI Verified subset: *Introducing SWE-Bench Verified* — OpenAI, 2024 (blog).
- Paper: *An Empirical Study of Harness Design for Coding Agents* — Fan et al., 2026 — uses SWE-Bench Verified + Terminal-Bench 2.1 for 176-setting ablation.
