# SWE-Bench Pro Verified

*Depth — an audited, re-scoped 731-instance split of SWE-Bench Pro that closes reward-hacking leakage and refines misleading tasks.*

**TL;DR:** SWE-Bench Pro is the leading agentic repository-level coding benchmark, but its evaluation is compromised in two ways: gold-solution / hidden-eval-info **leakage** enables agents to reward-hack their way to scores, and **task-quality issues** (misleading problem statements, improperly-scoped tests) produce noisy grades. **SWE-Bench Pro Verified** combines anti-hacking safeguards with task refinement across 731 audited instances. Leaderboard rankings shift as reward-hacking exploits stop scoring.

**Prereqs:** [livecodebench](livecodebench.md), [humaneval](humaneval.md)
**Related:** [codeforces-benchmark](codeforces-benchmark.md), [../data/decontamination.md](../data/decontamination.md), [README](README.md)

---

## What it is

SWE-Bench-style benchmarks measure whether a coding agent can, given a repo and a natural-language bug/feature description, produce a patch that passes hidden tests. As the community has scaled to harder splits (SWE-Bench Pro), two problems compounded:

- **Leakage.** Gold patches, hidden test scaffolding, or environment metadata visible to the agent let it *recover* the answer instead of solve the task.
- **Task quality.** Some problem statements were misleading; some hidden test suites tested behavior orthogonal to the described change.

SWE-Bench Pro Verified is the audited fix: 731 instances re-checked, with explicit anti-hacking safeguards and refined task specifications.

## How it works

Two-pronged repair:

1. **Anti-hacking safeguards.**
   - Close visible channels through which the gold patch can leak (e.g., diff-in-commit-message, config paths that name the fix, hidden test hooks that expose reference behavior).
   - Sandbox environment metadata that agents were exploiting.

2. **Task refinement.**
   - Rewrite problem statements where the intent was ambiguous or wrong.
   - Re-scope hidden test suites so they test the described behavior (and not unrelated regressions or over-strict edge cases).

The Verified split is a strict subset of Pro (731 instances) — everything that survived the audit.

## Why it matters

- **Fixes half the training loop.** Every RL post-training run using SWE-Bench-family verifiers as a reward is only as clean as the benchmark. Verified lifts that ceiling.
- **Leaderboard hygiene.** Agents that scored well by exploiting leakage drop; agents that solved the described task rise.
- **Blueprint for other agentic benchmarks.** The Pro→Verified audit pattern generalizes to any long-lived agentic eval.

## Gotchas & tricks

- **Verified ≠ contamination-free.** Anti-hacking closes environment leakage; it does not eliminate pretraining contamination — the underlying repos may still have appeared in the model's pretraining data. Combine with contamination measurement.
- **Score comparability is broken across splits.** Numbers on Pro and Verified are not directly comparable; always state the split.
- **Not a training set.** The audited split should stay a test set — using it as SFT/RL data would re-open the contamination loop.
- **Anti-hacking safeguards are local.** New agent capabilities will find new exploits; expect a Verified-v2 in the future.

## Sources

- Paper: *SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents* — Zheng, Shang, Jiang, Tian, Zhu, Ma, Yuan, Zhang — Shanghai AI Lab / Fudan University, 2026 — [arXiv:2609.08149](https://arxiv.org/abs/2609.08149).
- Companion: SWE-Bench (Jimenez et al., 2023), SWE-Bench Pro (prior split).
