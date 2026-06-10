# Agents' Last Exam (ALE)
*Depth — the long-horizon, economically valuable benchmark for AI agents.*

**TL;DR:** A benchmark for **long-horizon, economically valuable, verifiable-outcome real-world tasks**, organized along the **U.S. O*NET / SOC 2018 occupational taxonomy** — 13 industry clusters, 55 sub-fields, **1K+ tasks**. Co-developed with **250+ industry experts across 88 institutions**. The framing: benchmark gains haven't translated into deployment because we benchmark the wrong things; ALE indexes evaluation directly to the units of professional work that economists already use. Introduced by Sun et al., UC Berkeley, 2026 (arXiv 2606.05405).

**Prereqs:** *(none)*
**Related:** [README.md](README.md) · [livecodebench.md](livecodebench.md) · [humaneval.md](humaneval.md)

---

## What it is

Paper: *Agents' Last Exam* — Sun et al., UC Berkeley, 2026.

- **Taxonomy:** maps tasks to **O*NET / SOC 2018**, the U.S. federal occupational classification used by the BLS for labor statistics. Each ALE task corresponds to a recognizable unit of professional work in a non-physical industry.
- **Scale:** **13 industry clusters → 55 sub-fields → 1K+ tasks**. Built with 250+ industry experts (the "execution team" plus an advisory committee of practitioners).
- **Coverage:** non-physical industries only — knowledge work, professional services, software, finance, design, scientific computing, etc.
- **Outcomes are verifiable.** Each task ships with a checkable success criterion the expert practitioner would also use.

## How it works as an LLM eval

- **Task interface.** Each task gives the agent the problem statement, the inputs, and the success criterion. The agent operates over a long-horizon trajectory (hours-scale, not single-turn) using its native tooling — code execution, browsers, file systems, model calls.
- **Verifier.** Outcome verification — produced artifacts or final answers checked against the expert-defined criterion. This is what makes ALE *gradeable* at scale, not just "ask the expert to look at it."
- **Reporting.** Per-cluster and per-subfield success rates, plus aggregate. The economic-impact framing means cluster-level rates are the headline (not a single overall number).

## Why it matters

- **Indexed to the economy.** Saturating MMLU told us almost nothing about deployment; ALE results map to *types of work models can plausibly do for pay*. Likely to become a reference benchmark in frontier-lab agent reports.
- **Verifiable + long-horizon.** Most agent benchmarks pick one. ALE's expert-defined criteria make long-horizon tasks gradeable.
- **Expert-built, not researcher-imagined.** The 250+ practitioner panel is what makes the taxonomy and tasks credible to deployers — the criterion of "would an expert pay for this output" is in the loop from the start.
- **Headline gap.** Frontier agents are far below human-expert ceilings on the long-horizon clusters — the benchmark establishes a measurable gap that wasn't visible in shorter-horizon evals.

## Gotchas & tricks

- **Expert criteria can be subjective.** Even "verifiable outcomes" sometimes leave room for grader disagreement. The paper handles this with cross-verifier protocols, but expect noise on the soft-skill clusters.
- **Non-physical only.** Robotics, manipulation, embodied tasks are out of scope. For those, look at SpatialWorld and OmniGameArena.
- **Long-horizon means expensive to grade.** A single agent run can take hours. Sample-efficient evaluation strategies (early-stopping, bandit-style cluster selection) are likely needed in practice.
- **Industry coverage is U.S.-centric.** The O*NET / SOC taxonomy is U.S.; transfer to other labor markets requires re-mapping.
- **Contamination risk.** As ALE tasks become public, future model training corpora may absorb them. Watch for revision protocols similar to MMLU-Pro.

## Sources

- Paper: *Agents' Last Exam* — Sun, Han, Zhang, Pang, Wang et al. — UC Berkeley + 88 affiliated institutions, 2026 — arXiv 2606.05405.
- Reference: O*NET / SOC 2018 occupational taxonomy — U.S. Bureau of Labor Statistics.
