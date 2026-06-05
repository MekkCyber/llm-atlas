# Production-grounded agent benchmark (RAMP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RAMP (Ouyang et al., 2026) runs SWE agents through *real, dependency-staged production workflows* — compiler-construction pipelines on the YatCC platform — instead of static one-shot tasks. The setup exposes failure-propagation behavior invisible to short-horizon benchmarks: across 15 mainstream models, task completion drops from 100% in stage 1 to 20% in the final stage; *none* completes the full pipeline; compute cost varies by **2525× between comparably scoring models**.

**Prereqs:** [README.md](README.md), [../agents/README.md](../agents/README.md)
**Related:** [livecodebench.md](livecodebench.md), [humaneval.md](humaneval.md)

---

## What it is

Existing SWE-agent benchmarks (SWE-bench, etc.) are *static, isolated, short-horizon*: one task, one repo, one final patch. Real production work is *dependency-staged* — early stages produce artifacts that later stages consume, partial failures must be recovered, tool interactions and CI feedback loops drive iteration. RAMP is a benchmark whose tasks are *staged CI-like pipelines* with these properties.

## How it works

1. **YatCC platform.** Production CI infrastructure used as the substrate. Tasks are real compiler-construction workflows with serial dependencies — a later stage cannot run until the artifact from an earlier stage exists.
2. **Standardized orchestration.** Heterogeneous LLM providers and agent SDKs run against the same execution interface, so results are comparable across models.
3. **Staged recovery mechanism.** When a stage fails, the harness offers structured recovery options (retry, skip with placeholder, repair). This lets the benchmark study *how* agents handle partial workflow failure, not just whether they succeed.
4. **Utility metrics.** Beyond binary completion, RAMP scores outcome quality and process efficiency jointly: compute cost, API calls, wall-clock, recovery actions taken.

## Why it matters

- **Cascading failure is the real bottleneck.** A static benchmark scoring 70% pass can hide a model that, in a 5-stage pipeline, succeeds only 17% of the time end-to-end. RAMP makes this explicit.
- **2525× cost spread.** Models with similar accuracy can differ by three orders of magnitude in resource use — completely invisible to accuracy-only benchmarks but the central concern for production deployment.
- **Failure mode taxonomy.** Reveals systematic patterns: early-stage hallucinated dependencies poison later stages; non-deterministic tool failures cascade; some agents over-retry while others under-explore.
- **Benchmarks the agent's *recovery* behavior**, which is unmeasurable in single-shot evals.

## Gotchas & tricks

- **Compiler tasks bias the benchmark toward systems thinking.** Reasoning-heavy agents that don't do well with toolchains will underperform here even if their reasoning is strong.
- **Staged recovery introduces dependency on the harness UI.** Models with different prompting conventions for tool calls may benefit from harness-specific tuning.
- **Cost metrics are vendor-specific.** Token-cost and wall-clock vary with provider pricing and rate limits; comparing across providers is approximate.
- **Hard to game.** Unlike static benchmarks, contamination is harder — the production pipeline has stochasticity and freshness.

## Sources

- Paper: *Benchmarks are Not Enough: RAMP for Runtime Assessing of Agentic Models in Production Systems* — Ouyang et al., 2026 — [arXiv:2605.27492](https://arxiv.org/abs/2605.27492).
- Affiliation: Sun Yat-sen University (YatCC platform).
