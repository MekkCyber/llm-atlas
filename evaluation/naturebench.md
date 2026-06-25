# NatureBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 90-task coding-agent benchmark distilled from peer-reviewed Nature-family publications. Each task pairs the original research codebase + dataset with the headline metric the authors reported; an agent passes only if its run reaches or exceeds that *published* SOTA number. Designed to test scientific result reproduction, not synthetic unit-test correctness.

**Prereqs:** [README.md](README.md)
**Related:** [livecodebench.md](livecodebench.md) · [codeforces-benchmark.md](codeforces-benchmark.md) · [_agent-benchmarks.md](_agent-benchmarks.md)

---

## What it is

SWE-bench-class benchmarks ask coding agents to fix GitHub issues against hidden unit tests. NatureBench raises the bar: instead of "make the tests pass," the verifier is *the metric the original paper reported in a Nature-family publication*. The agent must run the research code, hit (or beat) the published number, across 90 tasks spanning biology, chemistry, physics, ML, neuroscience, and more.

## How it works

Each task is a tuple `(repo, dataset, target_metric, target_value, time_budget)`:

- **Repo**: the public research codebase associated with the paper.
- **Dataset**: the data used for the headline experiment.
- **Target metric / value**: the number the paper reports (e.g. "0.847 AUC", "12.3 BLEU", "31% MoE expert utilization").
- **Time budget**: an upper bound on compute the agent may consume.

The agent is graded *pass / fail* per task: did its run produce a number that matches or exceeds the published value? Aggregate score is task-completion rate.

The contamination defense rides on two facts: (1) the verifier is the *number* published in the paper, not test outputs that might be memorized, and (2) cross-discipline scope means no single training set covers everything.

## Why it matters

- Frames "AI co-scientist" claims operationally: a system that cannot reproduce a published Nature result on a fresh run can hardly be a research collaborator.
- Removes contamination risk that has plagued SWE-bench: agents can no longer "remember" the right patch from training data — they must run the actual computation.
- Cross-discipline reach forces breadth: a coding agent over-fit to ML repos will fail the biology and physics tasks.

## Gotchas & tricks

- The "match published SOTA" verifier is binary; partial-credit grading would smooth the leaderboard but the paper opts for binary pass/fail.
- Compute budgets are necessary — some Nature-family experiments are days-long on multi-GPU. The benchmark caps time per task; very compute-heavy tasks may be unreachable for any agent within budget.
- Reproducibility varies across the source papers themselves. A few tasks are arguably unsolvable because the published number is not reliably reproducible by humans either.
- Initial release shows all ten evaluated frontier coding-agent systems falling well short of full coverage; the headroom is large.

## Sources

- Paper: *NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?* — Wang, Cheng, Zuo, et al., Tsinghua / Frontis.AI, 2026 — [arXiv:2606.24530](https://arxiv.org/abs/2606.24530).
