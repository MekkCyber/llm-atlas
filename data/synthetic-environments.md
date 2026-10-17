# Synthetic environments for data agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Open environments for data-analysis agents are tiny (≤100 tasks) and closed training corpora can't be verified. **Specify the generative world mechanism first, then synthesize the files from it** — now ground-truth is known by construction, and an **independent audit** can check that the target finding is actually recoverable from the files. Yields a scalable, verifiable training distribution with built-in rubrics usable as RL rewards.

**Prereqs:** [_data-curation.md](./_data-curation.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md) · [decontamination.md](./decontamination.md)

---

## What it is

Training a data agent (one that reads heterogeneous files, writes code, extracts insights) needs environments shaped like real analytical workloads — spreadsheets, CSV dumps, PDFs, SQL exports — paired with ground-truth findings and gradable rubrics. The existing options:

- **Curated real datasets.** Verifiable but tiny, privacy/IP restricted.
- **Synthesized tasks from existing files.** Scalable but ground-truth is at best approximate; LLM-generated findings about files don't bind to a known mechanism.

Data World flips the generation order:

> Specify the world (a generative mechanism), then produce the files it would emit, then write tasks whose answers follow from the world — not from the files.

The files are then the agent's input; the world remains hidden. Rubrics describe how to grade a submitted finding against the (known) world's truth.

## How it works

### The engine

A multi-agent engine produces, jointly:

1. **World mechanism** — a structured description of entities, relations, dynamics, noise, and the latent causal/statistical story.
2. **Observable files** — CSVs, logs, PDFs, etc., sampled from the world mechanism's output.
3. **Task** — a question whose answer is a function of the world (not just the files).
4. **Rubric** — grading criteria that bind the submitted finding to the world's ground truth.

### Independent audit

A critical constraint: a separate agent (or a reference model independent of the generation agents) must be able to **recover the target finding from the files alone** using tools it would have in training. If it can't, the task is discarded. This stops the generator from producing tasks whose answers are *only* in the hidden world — tasks that would be unfairly ungradable.

### Rubrics as RL rewards

Rubrics are structured enough to score automatically. The same rubric that grades a benchmark submission becomes a reward signal for an RL run — enabling **RL-zero** (no distilled trajectories) training on synthesized tasks.

### Scale

Data World generates 2,135 training tasks over 1,037 **disjoint** worlds (benchmark worlds held out), across 16 domains.

### Human preference curation on question realism

Not everything that generates clean rubrics is realistic. Preference-based curation (human raters comparing synthesized questions to real workplace tasks) is layered on top to pick the final 100 benchmark tasks and the training set.

## Why it matters

- **RLVR for data work.** Rubrics give the verifiable-reward handle that RLVR has for math/code, but for open-ended data analysis.
- **Scalable without privacy risk.** The worlds are synthetic; the files are synthetic; no real data leaves anywhere.
- **Reveals the computation–insight gap.** Benchmarking 10 frontier models shows a consistent gap between "can execute the computation" and "can state the insight," repeatable across worlds.
- **RL-zero works.** No trajectory distillation needed to lift Qwen3.6-35B-A3B → DataWorld-35B-A3B (+7.06 DataWorld-Bench, +14.28 DAComp, +20.90 DABstep-Hard, +12.13 JobBench).

## Gotchas & tricks

- **The audit is load-bearing.** Without it, the generator will drift toward tasks whose answers are in the hidden world. The audit's capability tier (what tools, what model) sets the task difficulty floor.
- **World-mechanism language is open design.** The paper uses a structured domain-specific description; richer worlds (interactive, time-evolving) are open.
- **Decontamination matters.** Benchmark and training worlds must stay disjoint — easy to violate accidentally when worlds share a domain template. See [decontamination.md](./decontamination.md).
- **Human curation isn't optional.** Rubrics that grade cleanly aren't the same as realistic tasks; preference curation stops drift toward "easy to grade, uninteresting to answer."
- **Not a replacement for real data.** Synthetic worlds cover analytical structure, not real-world messiness (encoding bugs, upstream silent errors, political constraints on tooling).

## Sources

- Paper: *Data World: Scaling Data Environments from Scratch* — Li, Pang, Liu, Ji, Sun, Zhang, Xu, Yue, Luo, Zhao, Hua, Ruan, Su, Hong, Zeng, Zeng, Cai, Ye (Meituan Longcat + Nanjing U. + SJTU + CUHK + CAS), 2026 — [arXiv:2610.21656](https://arxiv.org/abs/2610.21656).
- Code: https://github.com/Zhanli-Li/Data-World
- Benchmark: DataWorld-Bench (100 tasks, 50 worlds, 16 domains); related benchmarks: DAComp, DABstep-Hard, JobBench.
