# Source-Free Environment Construction

*Depth — auto-generated training environments where the agent sees only a reference binary and its docs, never the source.*

**TL;DR:** Training coding agents on from-scratch program construction (build-a-CLI-tool-from-nothing) is bottlenecked by data: there aren't enough hand-authored full-lifecycle SE trajectories. MindForge automates the environment side: take an open-source command-line program, strip its source, expose only the compiled reference executable and its documentation, and use *that* as the training environment. A teacher agent generates program-synthesis trajectories against the source-free environment, and those trajectories fine-tune a student.

**Prereqs:** *(none — a data-pipeline pattern)*
**Related:** [README.md](./README.md), [behavioral-spec-elicitation.md](./behavioral-spec-elicitation.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A pipeline that converts open-source CLI programs into **training environments** where the LLM agent has:
- The reference binary (execute-only).
- Human-authored documentation (the tool's README / man page / --help output).

And doesn't have:
- The source code.
- Ground-truth reference implementations.

The reference binary alone serves as a *behavioral oracle* — the agent probes it, observes outputs, and must implement equivalent behavior from scratch.

## How it works

1. **Ingest OSS repos.** Any CLI program with docs qualifies; the pipeline is repo-agnostic.
2. **Build a source-free environment.** Compile the binary, strip source, expose docs.
3. **Teacher rollouts.** A strong LLM (GLM-5.2 in the paper) attempts each task, generating full program-synthesis trajectories (planning, probing, coding, testing).
4. **Curate high-quality trajectories.** Filter for successful, coherent runs.
5. **Fine-tune the student.** Standard SFT on the trajectories.

Crucially, the training-time repos are **disjoint from the eval-time repos** (ProgramBench), so the student isn't just memorizing the eval set's binaries.

## Why it matters

- **Scalable SE training data.** Any binary + docs pair becomes an environment. Doesn't depend on hand-curated tasks or gold references.
- **Whole-lifecycle coverage.** Unlike bug-fix / issue-resolution training data, source-free environments exercise the full lifecycle: reading docs, probing behavior, planning, implementing, testing.
- **Transfer to unseen benchmarks.** In the paper, fine-tuning a mid-size model on source-free trajectories improves seven unseen SE benchmarks, including RepoZero-C2Rust (+31.0), DeepSWE (+14.2), and SWE-bench Verified (+5.0). Signal is generic, not benchmark-specific.
- **The "source-free" constraint forces engineering.** Removing source-code peeking prevents the model from taking shortcuts; probing behavior is the only path to correctness.

Reported result: Qwen3.6-27B fine-tuned on source-free trajectories lifts ProgramBench from 37.98% → 49.51%, matching substantially larger frontier models.

## Gotchas & tricks

- **Binary must be behaviorally rich enough to probe.** Trivial CLIs (echo, cat) generate uninformative trajectories. The pipeline benefits from moderately-complex tools where probing actually pays off.
- **Docs quality caps trajectory quality.** Poorly-documented OSS tools produce noisy teacher trajectories. Filter aggressively.
- **Teacher-model choice matters.** GLM-5.2 in the paper; a weaker teacher generates fewer usable trajectories. Standard rejection-sampling / trajectory-filtering practice applies.
- **Doesn't teach open-ended design.** Source-free env construction is oracle-anchored — the reference binary defines correctness. It doesn't train the agent for tasks where *what to build* is genuinely up to the model.
- **Companion to [behavioral-spec-elicitation](./behavioral-spec-elicitation.md).** MindForge (this pipeline) is the training-data side; SpecFirst is the scaffold side. Same team, complementary contributions.

## Sources

- Paper: *MindForge: Teaching Small Language Models Whole-Life-Cycle Software Engineering via Source-Free Program Synthesis* — Chen et al., 2026 — introduces the pipeline. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Related: [behavioral-spec-elicitation.md](./behavioral-spec-elicitation.md).
