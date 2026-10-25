# OpenDS-World
*Depth — benchmark for multi-step deep-research agents on the live open web.*

**TL;DR:** OpenDS-World is a benchmark of 200 human-verified tasks that require an agent to find, access, and combine evidence from several independent sources on the *live* open web to produce one verifiable answer. Tasks span varied media types (PDFs, interactive sites, images), file formats, and interactive-access patterns. The best of 12 frontier models reached 39.0% accuracy; humans averaged 21.1% across 1,000 attempts under a different cost budget. The benchmark exposes termination and failure-recovery as the main brittleness points of current agents.

**Prereqs:** none
**Related:** [humaneval.md](./humaneval.md), [livecodebench.md](./livecodebench.md)

---

## What it is

A deep-research evaluation suite that refuses to use a frozen corpus. Each task is a *question whose answer requires live retrieval* — agents must drive a browser or equivalent and operate across heterogeneous sites. The 200 tasks are human-verified, meaning annotators checked both the ground-truth answer and that the retrieval path still works on the open web at the time of release.

## How it works

- **Task format.** Natural-language question + verifiable answer. Verification is automatic (string / numeric match) and sometimes assisted by rubric checks.
- **Tooling.** Agents are evaluated with their native tooling (search, browser interaction, file reading, media access). The benchmark doesn't dictate a tool set.
- **Metrics beyond accuracy.** Tool-use patterns, number of search queries, termination behavior (did the agent stop at the right time?), and failure-recovery (did it try another route after a dead end?) are first-class.
- **Human baseline.** 1,000 attempts by human solvers; cost budgets differ from agents (humans allowed longer wall-clock, no API fees), so the headline numbers aren't directly comparable but *are* informative on floor difficulty.

## Why it matters

- **Live web ≠ frozen corpus.** Agents trained or evaluated on static snapshots never see broken links, interactive paywalls, cookie banners, or stale data. OpenDS-World is a step toward evaluations that match how agents will be deployed.
- **Agents outscore humans on coverage** (39% vs 21%) but fail at *termination* and *recovery* — the places where humans excel. This points at a specific research direction (self-stopping policies, dead-end detection) rather than more capability.
- **Standard harness.** Replaces or supplements BrowseComp-style evaluations with heterogeneous-media breadth.

## Gotchas & tricks

- Live-web evaluation is inherently non-stationary. Scores from different months aren't directly comparable; the benchmark needs periodic revalidation.
- Human vs agent cost budgets are not matched. Humans at 21% accuracy with unlimited time is not the same ceiling as agents at 39% with API-limited budget.
- Tasks that require interactive access (login, geo-restricted content) can fail for infrastructural reasons unrelated to agent capability — the breakdown lets you factor those out.
- Diversity of media types means a single tool failure (e.g. broken PDF reader) tanks a sub-slice.

## Sources

- Paper: *Deep Search in Real-World Web Environments* — Yu et al., 2026 — [arXiv:2610.30181](https://arxiv.org/abs/2610.30181)
