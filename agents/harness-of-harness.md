# Harness-of-Harness (HoH)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A framework that treats existing coding-agent *harnesses* (Codex, OpenCode, Pi) as **primitives** and orchestrates them across days-long autonomous software-development runs. Wraps each harness in an iterative plan-code-test loop that scopes work into small verifiable increments, separates implementation-time testing from independent evaluation, versions project history, and progressively exposes tools/skills. Reports +52.25% average and +82.86% max relative gain over standalone harnesses on three benchmarks; a 70+ iteration deployment autonomously produces a playable FPS.

**Prereqs:** none
**Related:** [../evaluation/livecodebench.md](../evaluation/livecodebench.md), [../evaluation/humaneval.md](../evaluation/humaneval.md), [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

Standalone coding-agent harnesses handle short-horizon tasks well but plateau on multi-day autonomous development, where the deliverable is a *complete functional system* rather than a single patch. HoH is a coordinator layer above the harness: given a high-level requirement, it drives repeated iterations of an existing harness/model pair (Codex+GPT-5.5, OpenCode+DeepSeek-V4-Pro, Pi+MiniMax-M3) toward the deliverable, injecting structure at exactly the seams where lone harnesses fail.

## How it works

The design choices are:

1. **Iterative plan–code–test loop.** Each iteration produces a small, verifiable increment on top of the versioned project.
2. **Verifiable outputs, not prescribed workflows.** The framework specifies *what* each iteration must produce (deliverables + tests), not *how* the underlying harness should get there.
3. **Balance repair vs. capability growth.** Iterations alternate between fixing regressions and adding new features so the codebase doesn't calcify around bug-fixing.
4. **Separate implementation-time testing from independent evaluation.** The harness's own tests can be Goodharted; an *independent evaluator* runs on the same deliverable. This is HoH's central anti-Goodhart move.
5. **Progressive tool/skill exposure.** The framework starts small and exposes more capability as the project matures — new file types, new role-specific tools, new integrations. Prevents the harness from being swamped by tool choice early on.
6. **Versioned project history for reuse.** Each iteration's artifact is stored so later iterations can reuse rather than recreate. Cuts hallucination of already-existing modules.

## Why it matters

- **The right primitive for autonomous SWE may be a *loop over harnesses*, not a new agent model.** HoH re-uses existing harnesses unchanged and gets order-52% gains on average.
- **Independent evaluation is the anti-Goodhart wedge.** Any long-horizon agent loop needs to separate the reward signal from the training-time test signal or it will hack the test.
- **Multi-day empirics.** The 70+ iteration FPS build is one of the first published autonomous-agent runs measuring capability at multi-day timescales rather than single-task turns.
- **Harness-agnostic template.** Works with three different harness-model pairs, suggesting the pattern rather than any specific harness is doing the work.

## Gotchas & tricks

- **"Small increments" is a hard constraint.** Letting an iteration attempt a giant change re-introduces harness-level failure modes and defeats the loop's iterative nature.
- **Independent evaluator must be *actually* independent.** If both the harness and the evaluator share the same LLM/prompt style, the anti-Goodhart property is lost.
- **Progressive tool exposure is scheduled, not adaptive.** The paper's default is a fixed exposure schedule; more adaptive variants are open work.
- **Versioned history is not the same as a git log.** HoH stores deliverables + role-specific views, so later iterations don't need to re-derive project state from raw commits.
- **Benchmarks are not routine SWE benchmarks.** GameCraft-Bench and ProgramBench measure multi-file, multi-day capability that HumanEval/LiveCodeBench don't.

## Sources

- Paper: *Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement* — Yan et al., 2026 — [arXiv:2609.01481](https://arxiv.org/abs/2609.01481).
