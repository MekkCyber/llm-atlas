# Φ-Bench

*Depth — a benchmark that evaluates LLMs on engineering the LLM-infrastructure stack, from kernels to end-to-end system optimization.*

**TL;DR:** Existing LLM-for-code benchmarks (HumanEval, LiveCodeBench, Codeforces) crop to isolated tasks with pre-specified objectives. **Φ-Bench** asks whether frontier LLMs can engineer the *infrastructure that powers them* — CUDA kernels, ops, schedulers, distributed primitives — up to end-to-end system optimization. Tasks are drawn from real frontier-research optimization problems and grounded in real code repositories. Frontier models are competitive on kernel-level completion but degrade sharply as scope widens toward long-horizon systems work.

**Prereqs:** [livecodebench](livecodebench.md), [codeforces-benchmark](codeforces-benchmark.md)
**Related:** [humaneval](humaneval.md), [README](README.md)

---

## What it is

The benchmark is defined at four scope levels:

- **Kernel-level function completion.** Fill in a single CUDA / Triton / low-level kernel body against unit tests.
- **Localized ops.** Compose a small library of ops (e.g. an attention variant) meeting a functional spec.
- **Long-horizon implementation.** Wire a new subsystem into an existing repo — data loader, scheduler, distributed primitive — with cross-file changes and correct integration.
- **End-to-end system optimization.** Improve a system-level metric (throughput, latency, memory) subject to correctness constraints; the agent chooses which levers to pull.

Problems are extracted from optimization work that has appeared in frontier ML research (kernel fusions, quantization, parallelism schemes) and re-anchored to public code repos so evaluation is reproducible.

## How it works

Each task ships with:

- A base repository state.
- A specification (functional for kernel/op tasks; metric for optimization tasks).
- A test harness (unit tests, perf benchmarks) that the eval runs against the agent's submission.
- Optional starter scaffolding / docs.

An agent is evaluated by giving it repo access and letting it produce a diff. The harness runs correctness tests and, for optimization tasks, perf tests; the score reflects both.

## Why it matters

- **Measures agentic systems work, not just leetcode.** Existing benchmarks reward one-shot code generation; Φ-Bench rewards long-horizon plan-and-modify behavior.
- **Ties to the auto-R&D agenda.** If LLM agents will optimize their own infrastructure (paper #1's AutoResearch angle), we need a rigorous eval for that capability.
- **Fills a specific gap.** SWE-Bench-family benchmarks are software-engineering-broad; Φ-Bench is ML-infrastructure-deep.

## Gotchas & tricks

- **Benchmark saturation risk.** ML-infra optimization problems overlap with published research — decontamination is critical (see [../data/decontamination.md](../data/decontamination.md)).
- **Perf tests are hardware-sensitive.** A submission that wins on H100 may lose on A100. The eval fixes hardware; comparability across runs relies on that.
- **Correctness is required for perf credit.** A blindingly fast wrong answer scores 0. Optimization runs must maintain functional equivalence.
- **Not a leaderboard for base models alone.** Long-horizon tasks depend heavily on the agent harness around the model; scores are model + harness, not just model.

## Sources

- Paper: *Φ-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?* — Ding, Wang, Huang, Wan, Zhang, Han, Xu, Zhang, Chu, You, Zhang, Jiang, Zhang, 2026 — [arXiv:2609.10226](https://arxiv.org/abs/2609.10226).
