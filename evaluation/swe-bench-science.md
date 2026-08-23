# SWE-bench Science
*Depth — a repository-level benchmark for coding agents on scientific software repair, with paired ablations that decouple engineering ability from grounded scientific knowledge.*

**TL;DR:** **119 tasks across 98 GitHub repos and 20 scientific domains**, organized into three paradigms (Issue-driven, Expert-exploratory, Engineering-integration). Even the strongest agent (Claude Code with Opus-5 max) scores **below 50% pass@1**. The benchmark's headline contribution is not just difficulty but a paired ablation methodology that strips explicit scientific guidance while keeping the executable repo context — cleanly separating "agent's engineering ability" from "agent's ability to use scientific knowledge." Result: guidance quality matters — well-grounded guidance helps, poorly-aligned guidance *anchors* and hurts.

**Prereqs:** [humaneval.md](humaneval.md), [livecodebench.md](livecodebench.md)
**Related:** [../data/decontamination.md](../data/decontamination.md)

---

## What it is

A repo-level SWE-bench-style benchmark specialized to scientific software, with:

- **Three task paradigms.** *Issue-driven* (fix a filed bug), *Expert-exploratory* (an open-ended repair informed by domain knowledge), *Engineering-integration* (make components interoperate under domain constraints).
- **20 scientific domains.** Physics, biology, chemistry, geosciences, etc. — chosen so correctness has scientific consequence, not just code passing tests.
- **Paired scientific-guidance ablation.** Every task can be run with or without explicit scientific-context guidance while the executable repo remains identical.

## How it works

For each task the agent receives (a) the repo checkout, (b) failing-state description or issue text, and (c) *optionally* scientific guidance (background derivations, physical principles, known constraints). The verifier runs the repo's existing tests plus any task-specific integration tests. `pass@1` and token cost are reported per paradigm.

Failure analysis (over the run outputs) attributes each miss to one of four mechanisms:

1. **Scientific-knowledge deficit / abstraction failure** — the agent doesn't know the domain concept the fix requires.
2. **Misguided exploration / surface repair** — the agent patches the immediate symptom rather than the underlying cause.
3. **Incomplete repair coverage / integration failure** — the fix works locally but breaks integration or misses coupled sites.
4. **Failure to generalize scientific knowledge** — the agent can handle observed cases but not natural extensions.

## Why it matters

Frontier coding agents ceiling out sub-50% on a benchmark this focused says the "agents-solve-software-engineering" arc has a real ceiling in domains where surface success ≠ correctness. And the paired-ablation design gives the field a lever to measure a distinct capability — grounded use of domain knowledge — that generic SWE-bench success confounds. That distinction feeds directly into training-data curation choices for coding-agent RL.

## Gotchas & tricks

- 98 public GitHub repos raise obvious contamination risks; benchmark reporters must be explicit about training-data cutoffs and any decontamination pass.
- "Scientific guidance" is authored by humans and its quality varies; the paper's finding that *poorly-aligned* guidance hurts is a warning against uncritically piping RAG hits into a coding agent.
- Pass@1 with `max`-reasoning agents is expensive; per-domain token-budget reporting is required for fair comparison.

## Sources

- Paper: *SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?* — Xu, Lu, Zheng, Wang, Qiu (Fudan), 2026 — [arXiv:2608.19799](https://arxiv.org/abs/2608.19799)
