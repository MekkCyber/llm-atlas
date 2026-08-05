# SWE-bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark family for evaluating LLM coding agents on real GitHub bug-fix and feature tasks. Each instance provides a repository snapshot, an issue description, and a hidden held-out test suite. An agent passes if its patch makes the hidden tests go green without breaking existing ones. The de-facto standard for coding-agent evaluation and the training signal behind most SWE-tuned models.

**Prereqs:** none
**Related:** [humaneval](humaneval.md), [livecodebench.md](livecodebench.md), [codeforces-benchmark](codeforces-benchmark.md)

---

## What it is

Introduced by Jimenez et al. (Princeton, 2023) with 2,294 instances mined from 12 Python repositories. An **instance** contains:

- A repository at a specific commit.
- A GitHub issue title and body describing the bug/feature.
- A **developer patch** (ground truth, hidden from the agent).
- **FAIL_TO_PASS** tests — currently failing tests the developer patch makes pass.
- **PASS_TO_PASS** tests — currently passing tests the patch must not break.

The agent sees the issue + repo and must produce a patch. Evaluation: apply the patch, run FAIL_TO_PASS and PASS_TO_PASS, report **resolve rate** (fraction with all tests passing).

## How it works

The benchmark has grown into a **family**:

- **SWE-bench (full)** — the original 2,294 instances. Now largely deprecated as a scoring target due to noise (some instances unsolvable from the given info; some tests flaky).
- **SWE-bench Lite** — 300-instance subset for fast iteration.
- **SWE-bench Verified** (OpenAI, 2024) — 500 human-audited instances confirmed solvable from the given information, with unambiguous test intent. The current standard leaderboard.
- **SWE-bench Multimodal** — instances requiring image understanding.
- **SWE-Bench Pro** — longer-horizon, multi-file tasks with harder isolation.
- **DeepSWE** — longer, more complex instances curated for training and eval.
- **Multilingual variants** — Java, Rust, Go, TypeScript benchmarks in the same shape.

Standard evaluation harness: the agent is given tool access (shell, editor, tests), a step budget, and returns a unified diff. Community consensus is to run under a **standardised harness** so scores compare across models.

## Why it matters

- **The coding-agent leaderboard.** Almost every frontier lab reports SWE-bench Verified numbers on major releases. Progress here anchors public perception of coding progress.
- **Training signal.** Instances double as RL environments: the hidden test suite is a *verifiable reward*, ideal for RLVR-style training (used by DeepSWE, Devin, Aider, and others).
- **A large real-code target.** Unlike HumanEval (small standalone functions) or LiveCodeBench (competitive programming), SWE-bench forces multi-file understanding of real, messy repositories.

## Gotchas & tricks

- **Contamination.** Many instances have been publicly discussed since 2023; verify a model's training-cutoff-vs-benchmark-cutoff before believing published scores.
- **Harness matters as much as model.** The same base model scores very differently under different harnesses (max steps, tools, retry policy). Compare only within a single harness.
- **The "resolve = pass" metric is coarse.** A patch can pass tests while making the code worse — see [deletion-recall](deletion-recall.md) and the Guard-and-Go pattern.
- **SWE-Touch (arXiv:2608.02499)** stress-tests state-awareness: injecting plausible user edits into task-critical regions drops resolve rate by 7.7pp across leading models. A model's SWE-bench Verified number does not predict its shared-workspace robustness.
- **Prefer Verified for evaluation, full or DeepSWE for training** — Verified is small (500) so training on it risks overfitting; the noisier full benchmark and DeepSWE offer more training volume.
- **Instances vary wildly in difficulty.** Report per-instance-cluster resolve rates if you want a real quality signal, not just an aggregate.

## Sources

- Paper: *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* — Jimenez, Yang et al., Princeton, 2023.
- Blog: *Introducing SWE-bench Verified* — OpenAI, 2024.
- Paper: *SWE-Touch: Benchmarking Coding Agents When Users Touch the Code* — arXiv:2608.02499, 2026 — shared-workspace stress test.
- Paper: *To Add Is Machine, To Delete Is Human* — arXiv:2607.28887, 2026 — deletion-recall metric on top of SWE-bench Verified.
- Leaderboard: [swebench.com](https://www.swebench.com/).
